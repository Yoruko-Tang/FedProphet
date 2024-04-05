from client.atclient import AT_Client
from torch.utils.data import DataLoader
import torch.nn
import copy
import numpy as np

from utils.adversarial import Adv_Sample_Generator
from hardware.sys_utils import model_summary

from typing import List

class Multimodel_Client(AT_Client):
    def __init__(self, dataset, data_idxs, sys_info=None,
                 model_profile:model_summary = None,
                 init_local_state:List[dict] = None,
                 local_state_preserve = False, 
                 test_adv_method='pgd',test_adv_epsilon=0.0,test_adv_alpha=0.0,
                 test_adv_T=0,test_adv_norm='inf',test_adv_bound=[0.0,1.0],
                 device=torch.device('cpu'), 
                 verbose=False, random_seed=None, 
                 reserved_performance = 0, reserved_memory = 0, **kwargs):
        super().__init__(dataset, data_idxs, sys_info, model_profile, 
                         init_local_state, local_state_preserve,
                         test_adv_method,test_adv_epsilon,test_adv_alpha,
                         test_adv_T,test_adv_norm,test_adv_bound,
                         device, verbose, random_seed, reserved_performance, 
                         reserved_memory,**kwargs)
        self.model_idx = 0
        
        
    def train(self,edge_models,model_idx,local_ep,local_bs,lr,optimizer='sgd',
              momentum=0.0,reg=0.0, criterion=torch.nn.CrossEntropyLoss(),
              adv_train=True,adv_method='pgd',adv_epsilon=0.0,adv_alpha=0.0,adv_T=0,
              adv_norm='inf',adv_bound=[0.0,1.0],adv_ratio=1.0,model_profile=None,**kwargs):
        """train the model for one communication round."""
        def standard_loss(model,input,label):
            output = model(input)
            task_loss = criterion(output,label)
            return task_loss
        
        
        model = copy.deepcopy(edge_models[model_idx]) # avoid modifying global model
        self.model_idx = model_idx
        model.to(self.device)
        if self.local_state_preserve and self.local_states[model_idx] is not None:
            model = self.load_local_state_dict(model,self.local_states[model_idx])
        
        

        trainloader = DataLoader(self.trainset,batch_size=local_bs,shuffle=True)

        # Set optimizer for the local updates
        param = model.parameters()
        if optimizer == 'sgd':
            opt = torch.optim.SGD(param, lr=lr, momentum=momentum,weight_decay=reg)
        elif optimizer == 'adam':
            opt = torch.optim.Adam(param, lr=lr, weight_decay=reg)
        

        adv_data_gen = Adv_Sample_Generator(standard_loss,adv_method,adv_epsilon,
                                            adv_alpha,adv_T,adv_norm,adv_bound)
        

        iters = 0
        self.batches = []
        while iters < local_ep:
        #for iter in range(self.args.local_ep):
            for _, (datas, labels) in enumerate(trainloader):
                # training batch
                self.batches.append(list(datas.shape))
                datas, labels = datas.to(self.device), labels.to(self.device)
                if adv_train:
                    self.adv_iters = adv_T
                    self.adv_ratio = adv_ratio
                    datas = adv_data_gen.attack_data(model,datas,labels,adv_ratio)
                else:
                    self.adv_iters = 0
                    self.adv_ratio = 0.0
                
                model.train()
                model.zero_grad()
                
                loss = standard_loss(model, datas, labels)
                loss.backward()
                opt.step()
                iters += 1
                if iters == local_ep:
                    break

            if self.verbose:
                print('Local Epoch : {}/{} |\tLoss: {:.4f}'.format(iters, local_ep, loss.item()))
        
        if self.local_state_preserve:
            self.local_states[model_idx] = copy.deepcopy(self.get_local_state_dict(model))
        self.final_local_loss = loss.item()
        
        # calculate training latency
        if model_profile is not None:
            self.model_profile = model_profile
        self.latency = self.model_profile.training_latency(performance=self.avail_perf,
                                                           memory=self.avail_mem,
                                                           eff_bandwidth=self.eff_bw,
                                                           network_bandwidth=self.network_speed,
                                                           network_latency=self.network_lag,
                                                           **self.__dict__)
        
        return {"model":model}
    
    def validate(self,model=None,edge_models=None,model_idx=None,
                 testset=None,criterion=torch.nn.CrossEntropyLoss(),
                 load_local_state = True,**kwargs):
        """ Returns the validation accuracy and loss."""
        if model_idx is None:
            model = copy.deepcopy(model)
        else:
            if model_idx == -1:# validate the model trained by this client in the last time
                model_idx = self.model_idx
            model = copy.deepcopy(edge_models[model_idx])
            if load_local_state and self.local_states[model_idx] is not None:
                model = self.load_local_state_dict(model,self.local_states[model_idx])
        model.to(self.device)
        
        
        model.eval()

        loss, total, correct = 0.0, 0.0, 0.0
        if testset is None:
            testset = self.testset
        testloader = DataLoader(testset,batch_size=128, shuffle=False)

        for batch_idx, (datas, labels) in enumerate(testloader):
            datas, labels = datas.to(self.device), labels.to(self.device)
            outputs = model(datas)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss/(batch_idx+1)
    
    def adv_validate(self,model=None,edge_models=None,model_idx=None,
                     testset=None,criterion=torch.nn.CrossEntropyLoss(),
                     load_local_state = True,**kwargs):
        """ Returns the validation adversarial accuracy and adversarial loss."""
        if model_idx is None:
            model = copy.deepcopy(model)
        else:      
            if model_idx == -1:# validate the model trained by this client in the last time
                model_idx = self.model_idx
            model = copy.deepcopy(edge_models[model_idx])
            if load_local_state and self.local_states[model_idx] is not None:
                model = self.load_local_state_dict(model,self.local_states[model_idx])
        model.to(self.device)
        
        model.eval()

        loss, total, correct = 0.0, 0.0, 0.0
        if testset is None:
            testset = self.testset
        testloader = DataLoader(testset,batch_size=128, shuffle=False)

        adv_data_gen = Adv_Sample_Generator(lambda m,i,y:criterion(m(i),y),
                                            self.test_adv_method,
                                            self.test_adv_epsilon,
                                            self.test_adv_alpha,
                                            self.test_adv_T,
                                            self.test_adv_norm,
                                            self.test_adv_bound)

        for batch_idx, (datas, labels) in enumerate(testloader):
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