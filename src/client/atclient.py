from client.stclient import ST_Client

import torch
from torch.utils.data import DataLoader
import torch.nn
import copy
import numpy as np

from utils.adversarial import Adv_Sample_Generator
from hardware.sys_utils import model_summary



class AT_Client(ST_Client):
    """
    This is a client that conducts adversarial training
    """
    def __init__(self, dataset, data_idxs, sys_info=None,
                 local_state_preserve = False, 
                 test_adv_method='pgd',test_adv_epsilon=0.0,test_adv_alpha=0.0,
                 test_adv_T=0,test_adv_norm='inf',test_adv_bound=[0.0,1.0],
                 device=torch.device('cpu'), 
                 verbose=False, random_seed=None, 
                 reserved_performance = 0, reserved_memory = 0, **kwargs):
        super().__init__(dataset, data_idxs, sys_info, local_state_preserve,
                         device, verbose, random_seed, reserved_performance, 
                         reserved_memory, **kwargs)
        
        self.test_adv_method = test_adv_method
        self.test_adv_epsilon = test_adv_epsilon
        self.test_adv_alpha = test_adv_alpha
        self.test_adv_T = test_adv_T
        self.test_adv_norm = test_adv_norm
        self.test_adv_bound = test_adv_bound

        
    def train(self,init_model,local_ep,local_bs,lr,optimizer='sgd',
              momentum=0.0,reg=0.0, criterion=torch.nn.CrossEntropyLoss(),
              adv_train=True,adv_method='pgd',adv_epsilon=0.0,adv_alpha=0.0,adv_T=0,
              adv_norm='inf',adv_bound=[0.0,1.0],adv_ratio=1.0,**kwargs):
        """train the model for one communication round."""
        if not adv_train: # conduct normal training if not adversarial training
            model = super().train(init_model,local_ep,local_bs,lr,optimizer,
                                  momentum,reg,criterion)
            return model
        model = copy.deepcopy(init_model) # avoid modifying global model
        model.to(self.device)
        if self.local_states is not None:
            model = self.load_local_state_dict(model,self.local_states)
        

        self.trainloader = DataLoader(self.trainset,batch_size=local_bs,shuffle=True)

        # Set optimizer for the local updates
        np = model.parameters()
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(np, lr=lr, momentum=momentum,weight_decay=reg)
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(np, lr=lr, weight_decay=reg)
        
        adv_criterion = lambda m,i,y:criterion(m(i),y)
        adv_data_gen = Adv_Sample_Generator(adv_criterion,adv_method,adv_epsilon,
                                            adv_alpha,adv_T,adv_norm,adv_bound)
        

        iters = 0
        self.batches = []
        while iters < local_ep:
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
                if iters == local_ep:
                    break

            if self.verbose:
                print('Local Epoch : {}/{} |\tLoss: {:.4f}'.format(iters, local_ep, loss.item()))
        
        if self.local_state_preserve:
            self.local_states = copy.deepcopy(self.get_local_state_dict(model))
        self.final_local_loss = loss.item()
        
        # calculate training latency
        self.model_profile = model_summary(model,self.batches[0],len(self.batches))
        self.latency = self.model_profile.training_latency(self.batches,self.avail_perf,self.avail_mem)
        
        return model
        
    def adv_validate(self,model,criterion=torch.nn.CrossEntropyLoss()):
        """ Returns the validation adversarial accuracy and adversarial loss."""
        model = copy.deepcopy(model) # avoid modifying global model
        model.to(self.device)
        if self.local_states is not None:
            model = self.load_local_state_dict(model,self.local_states)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        self.testloader = DataLoader(self.testset,batch_size=128, shuffle=False)

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