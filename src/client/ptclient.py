from client.atclient import AT_Client

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
import numpy as np

from utils.adversarial import Adv_Sample_Generator
from hardware.sys_utils import model_summary



class PT_Client(AT_Client):
    """
    This is a client that conducts adversarial training with partial training
    """
    def __init__(self, dataset, data_idxs, sys_info=None,
                 model_profile:model_summary = None,
                 init_local_state = None,
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
        
        self.partial_frac = 1.0

        

        
    def train(self,model,neuron_dict,partial_frac,local_ep,local_bs,lr,optimizer='sgd',
              momentum=0.0,reg=0.0,grad_clip=None, criterion=torch.nn.CrossEntropyLoss(),
              adv_train=True,adv_method='pgd',adv_epsilon=0.0,adv_alpha=0.0,adv_T=0,
              adv_norm='inf',adv_bound=[0.0,1.0],adv_ratio=1.0,model_profile=None,**kwargs):
        """train the model for one communication round."""
        def partial_loss(model,input,label):
            output = model.partial_forward(input,neuron_dict)
            task_loss = criterion(output,label)
            return task_loss
        
        model = copy.deepcopy(model) # avoid modifying global model
        model.to(self.device)
        if self.local_state_preserve and self.local_states is not None:
            model = self.load_local_state_dict(model,self.local_states)
        
        

        trainloader = DataLoader(self.trainset,batch_size=local_bs,shuffle=True)

        # Set optimizer for the local updates
        param = model.parameters()
        if optimizer == 'sgd':
            opt = torch.optim.SGD(param, lr=lr, momentum=momentum,weight_decay=reg)
        elif optimizer == 'adam':
            opt = torch.optim.Adam(param, lr=lr, weight_decay=reg)
        

        adv_data_gen = Adv_Sample_Generator(partial_loss,adv_method,adv_epsilon,
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
                
                loss = partial_loss(model, datas, labels)
                loss.backward()
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(param, grad_clip)
                opt.step()
                
                iters += 1
                if iters == local_ep:
                    break

            if self.verbose:
                print('Local Epoch : {}/{} |\tLoss: {:.4f}'.format(iters, local_ep, loss.item()))
        
        if self.local_state_preserve:
            self.local_states = copy.deepcopy(self.get_local_state_dict(model))
        self.final_local_loss = loss.item()
        
        # calculate training latency
        if model_profile is not None:
            self.model_profile = model_profile
        self.partial_frac = partial_frac
        self.latency = self.model_profile.training_latency(performance=self.avail_perf,
                                                           memory=self.avail_mem,
                                                           eff_bandwidth=self.eff_bw,
                                                           network_bandwidth=self.network_speed,
                                                           network_latency=self.network_lag,
                                                           **self.__dict__)
        sd = model.state_dict()
        updated_partial_idxs = {}
        
        for n,m in model.named_modules():
            if n+'.weight' in sd:
                if hasattr(m,'in_retain_idx') and hasattr(m,'retain_idx'): # conv/linear
                    column_idx = m.in_retain_idx
                    row_idx = m.retain_idx
                    updated_partial_idxs[n+'.weight'] = torch.zeros_like(m.weight.data)
                    updated_partial_idxs[n+'.weight'][np.ix_(row_idx,column_idx)] = 1
                elif hasattr(m,'retain_idx'): # batchnorm
                    updated_partial_idxs[n+'.weight'] = torch.zeros_like(m.weight.data)
                    updated_partial_idxs[n+'.weight'][m.retain_idx] = 1
                elif hasattr(m,'in_retain_idx'):# last linear
                    updated_partial_idxs[n+'.weight'] = torch.zeros_like(m.weight.data)
                    updated_partial_idxs[n+'.weight'][:,m.in_retain_idx] = 1
                else:
                    updated_partial_idxs[n+'.weight'] = torch.ones_like(m.weight.data)

            if n+'.bias' in sd:
                if hasattr(m,'retain_idx'): # conv, batchnorm, linear
                    updated_partial_idxs[n+'.bias'] = torch.zeros_like(m.bias.data)
                    updated_partial_idxs[n+'.bias'][m.retain_idx] = 1
                else:# last linear
                    updated_partial_idxs[n+'.bias'] = torch.ones_like(m.bias.data)

        
        # print(model.neuron_num)
        # for k in updated_partial_idxs:
        #     print(k,torch.sum(updated_partial_idxs[k]))
        # for n,p in model.named_parameters():
        #     print(n,torch.sum(p.grad!=0))
        # input()
        return {"model":model,"updated_partial_idxs":updated_partial_idxs}
        
    