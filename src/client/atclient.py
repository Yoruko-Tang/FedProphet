from stclient import ST_Client

import torch
from torch.utils.data import DataLoader
import torch.nn
import copy
import numpy as np

from utils.adversarial import Adv_Sample_Generator



class AT_Client(ST_Client):
    """
    This is a client that conducts adversarial training
    """
    def __init__(self, dataset, data_idxs, sys_info=None, 
                 local_state_preserve=False, 
                 test_adv_method='pgd',test_adv_epsilon=0.0,test_adv_alpha=0.0,
                 test_adv_T=0,test_adv_norm='inf',test_adv_bound=[0.0,1.0],
                 device=torch.device('cpu'), 
                 verbose=False, random_seed=None, **kwargs):
        super().__init__(dataset, data_idxs, sys_info, 
                         local_state_preserve, device, verbose, 
                         random_seed, **kwargs)
        
        self.test_adv_method = test_adv_method
        self.test_adv_epsilon = test_adv_epsilon
        self.test_adv_alpha = test_adv_alpha
        self.test_adv_T = test_adv_T
        self.test_adv_norm = test_adv_norm
        self.test_adv_bound = test_adv_bound
        
    def train(self,model,iteration,batchsize,lr,optimizer='sgd',
              momentum=0.0,reg=0.0, criterion=torch.nn.CrossEntropyLoss(),
              adv_method='pgd',adv_eps=0.0,adv_alpha=0.0,adv_T=0,
              adv_norm='inf',adv_bound=[0.0,1.0],adv_ratio=1.0,**kwargs):
        """train the model for one communication round."""
        model = copy.deepcopy(model) # avoid modifying global model
        model.to(self.device)
        if self.local_state_preserve and self.local_states is not None:
            model = self.load_local_state_dict(model,self.local_states)
        

        self.trainloader = DataLoader(self.trainset,batch_size=batchsize,shuffle=True)

        # Set optimizer for the local updates
        np = model.parameters()
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(np, lr=lr, momentum=momentum,weight_decay=reg)
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(np, lr=lr, weight_decay=reg)
        
        adv_criterion = lambda m,i,y:criterion(m(i),y)
        adv_data_gen = Adv_Sample_Generator(adv_criterion,adv_method,adv_eps,
                                            adv_alpha,adv_T,adv_norm,adv_bound)
        

        iters = 0
        self.batches = []
        while iters < iteration:
        #for iter in range(self.args.local_ep):
            for _, (datas, labels) in enumerate(self.trainloader):
                adv_bs = int(adv_ratio*len(datas))
                for i in range(adv_T):
                    # adv generating batches
                    self.batches.append([adv_bs]+list(datas.shape)[1:])
                # training batch
                self.batches.append(list(datas.shape))

                datas, labels = datas.to(self.device), labels.to(self.device)
                adv_datas = adv_data_gen.attack_data(model,datas,labels)
                model.train()
                model.zero_grad()
                output = model(adv_datas)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                iters += 1
                if iters == iteration:
                    break

            if self.verbose:
                print('Local Epoch : {}/{} |\tLoss: {:.4f}'.format(iters, iteration, loss.item()))
        
        self.local_states = copy.deepcopy(self.get_local_state_dict(model))
        self.final_local_accuracy,self.final_local_loss = self.validate(model)
        
        return model
        
    def adv_validate(self,model,criterion=torch.nn.CrossEntropyLoss()):
        """ Returns the validation adversarial accuracy and adversarial loss."""
        model = copy.deepcopy(model) # avoid modifying global model
        model.to(self.device)
        if self.local_state_preserve and self.local_states is not None:
            model = self.load_local_state_dict(model,self.local_states)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        self.testloader = DataLoader(self.testset,batch_size=32, shuffle=False)

        adv_data_gen = Adv_Sample_Generator(lambda m,i,y:criterion(m(i),y),
                                            self.test_adv_method,
                                            self.test_adv_epsilon,
                                            self.test_adv_alpha,
                                            self.test_adv_T,
                                            self.test_adv_norm,
                                            self.test_adv_bound)

        for batch_idx, (datas, labels) in enumerate(self.testloader):
            datas, labels = datas.to(self.device), labels.to(self.device)
            adv_datas = adv_data_gen.attack_data(model,datas,labels)
            outputs = model(adv_datas)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss/(batch_idx+1)