from client.atclient import AT_Client

import torch
from torch.utils.data import DataLoader,TensorDataset
import torch.nn
import copy
import numpy as np

from utils.adversarial import Adv_Sample_Generator
from hardware.sys_utils import sample_runtime_app,model_summary



class Module_Client(AT_Client):
    """
    This is a client that conducts adversarial training with fedprophet
    """
    def __init__(self, dataset, data_idxs, sys_info=None,
                 model_profile:model_summary = None,
                 local_state_preserve = False, 
                 test_adv_method='pgd',test_adv_epsilon=0.0,test_adv_alpha=0.0,
                 test_adv_T=0,test_adv_norm='inf',test_adv_bound=[0.0,1.0],
                 device=torch.device('cpu'), 
                 verbose=False, random_seed=None, 
                 reserved_performance = 0, reserved_memory = 0, **kwargs):
        super().__init__(dataset, data_idxs, sys_info, model_profile, 
                         local_state_preserve,test_adv_method,test_adv_epsilon,
                         test_adv_alpha,test_adv_T,test_adv_norm,test_adv_bound,
                         device, verbose, random_seed, reserved_performance, 
                         reserved_memory, **kwargs)

        self.feature_trainset = self.trainset
        self.feature_testset = self.testset

        self.module_list = None
        
    def forward_feature_set(self,model,module_list):
        """
        forward the current feature to the feature of the next stage
        """
        model = copy.deepcopy(model) # avoid modifying global model
        model.to(self.device)
        if self.local_states is not None:
            model = self.load_local_state_dict(model,self.local_states)
        model.eval()

        trainloader = DataLoader(self.feature_trainset,batch_size=128,shuffle=False)
        testloader = DataLoader(self.feature_testset,batch_size=128, shuffle=False)

        new_train_feature,new_test_feature = [],[]
        train_labels,test_labels = [],[]

        for datas,labels in trainloader:
            datas = datas.to(self.device)
            new_feature = model.module_forward(datas,module_list)
            new_feature = new_feature.detach().cpu()
            new_train_feature.append(new_feature)
            train_labels.append(labels)
        new_train_feature = torch.cat(new_train_feature,dim=0)
        train_labels = torch.cat(train_labels,dim=0)
        self.feature_trainset = TensorDataset(new_train_feature,train_labels)

        for datas,labels in testloader:
            datas = datas.to(self.device)
            new_feature = model.module_forward(datas,module_list)
            new_feature = new_feature.detach().cpu()
            new_test_feature.append(new_feature)
            test_labels.append(labels)
        new_test_feature = torch.cat(new_test_feature,dim=0)
        test_labels = torch.cat(test_labels,dim=0)
        self.feature_testset = TensorDataset(new_test_feature,test_labels)

        return self.feature_trainset,self.feature_testset


        
    def train(self,init_model,stage_module_list,prophet_module_list,
              stage_aux_model,prophet_aux_model,
              local_ep,local_bs,lr,optimizer='sgd',momentum=0.0,reg=0.0, 
              criterion=torch.nn.CrossEntropyLoss(),
              adv_train=True,adv_method='pgd',adv_epsilon=0.0,adv_alpha=0.0,adv_T=0,
              adv_norm='inf',adv_bound=[0.0,1.0],adv_ratio=1.0,
              mu=0.0,lamb=0.0,psi=0.0,**kwargs):
        """train the model for one communication round."""
        # model preparation
        model = copy.deepcopy(init_model) # avoid modifying global model
        model.to(self.device)
        if self.local_states is not None:
            model = self.load_local_state_dict(model,self.local_states)
        current_aux_model = copy.deepcopy(stage_aux_model)
        if current_aux_model is not None:
            current_aux_model.to(self.device)
        future_aux_model = copy.deepcopy(prophet_aux_model)
        if future_aux_model is not None:
            future_aux_model.to(self.device)

        def fedprophet_loss(model,input,label):
            output = model.module_forward(input,stage_module_list)
            if current_aux_model is not None:
                final_output = current_aux_model(output)
            else:
                final_output = output
            stage_task_loss = criterion(final_output,label)
            stage_cvx_loss = mu*torch.sum(torch.square(output))/(2*len(output))
            loss = stage_task_loss+stage_cvx_loss
            
            if len(prophet_module_list)>0:
                prophet_output = model.module_forward(output,prophet_module_list)
                if future_aux_model is not None:
                    prophet_final_output = future_aux_model(prophet_output)
                else:
                    prophet_final_output = prophet_output
                prophet_task_loss = criterion(prophet_final_output,label)
                prophet_cvx_loss = mu*torch.sum(torch.square(prophet_output))/(2*len(prophet_output))
                prophet_loss = prophet_task_loss + prophet_cvx_loss
                loss = loss*(1-psi)+prophet_loss*psi
            return loss

        
        
        

        self.trainloader = DataLoader(self.feature_trainset,batch_size=local_bs,shuffle=True)

        # Set optimizer for the local updates
        np = model.parameters()
        if current_aux_model is not None:
            anp = current_aux_model.parameters()
        else:
            anp = []
        if future_aux_model is not None:
            panp = future_aux_model.parameters()
        else:
            panp = []
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD([{'params':np,'weight_decay':reg},
                                        {'params':anp,'weight_decay':lamb},
                                        {'params':panp,'weight_decay':lamb}], 
                                        lr=lr,momentum=momentum)
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam([{'params':np,'weight_decay':reg},
                                         {'params':anp,'weight_decay':lamb},
                                         {'params':panp,'weight_decay':lamb}], 
                                         lr=lr)
        

        adv_data_gen = Adv_Sample_Generator(fedprophet_loss,adv_method,adv_epsilon,
                                            adv_alpha,adv_T,adv_norm,adv_bound)
        

        iters = 0
        self.batches = []
        while iters < local_ep:
        #for iter in range(self.args.local_ep):
            for _, (datas, labels) in enumerate(self.trainloader):
                # training batch
                self.batches.append(list(datas.shape))
                datas, labels = datas.to(self.device), labels.to(self.device)
                if adv_train:
                    self.iters_per_input = adv_T+1
                    datas = adv_data_gen.attack_data(model,datas,labels)
                else:
                    self.iters_per_input = 1
                
                model.train()
                model.zero_grad()

                loss = fedprophet_loss(model,datas,labels)
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
        self.module_list = stage_module_list + prophet_module_list
        self.latency = self.model_profile.training_latency(module_list=self.module_list,
                                                           batches=self.batches,
                                                           iters_per_input=self.iters_per_input,
                                                           performance=self.avail_perf,
                                                           memory=self.avail_mem
                                                           )
        
        return model,current_aux_model,future_aux_model
        
    def validate(self,model,module_list,auxiliary_model = None,criterion=torch.nn.CrossEntropyLoss(),**kwargs):
        """ Returns the validation accuracy and loss."""
        
        model = copy.deepcopy(model) # avoid modifying global model
        model.to(self.device)
        if self.local_states is not None:
            model = self.load_local_state_dict(model,self.local_states)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        testloader = DataLoader(self.testset,batch_size=128, shuffle=False)

        for batch_idx, (datas, labels) in enumerate(testloader):
            datas, labels = datas.to(self.device), labels.to(self.device)
            outputs = model.module_forward(input,module_list)
            if auxiliary_model is not None:
                outputs = auxiliary_model(outputs)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss/(batch_idx+1)
    
    def adv_validate(self,model,module_list=None,auxiliary_model = None,criterion=torch.nn.CrossEntropyLoss(),**kwargs):
        """ Returns the validation adversarial accuracy and adversarial loss."""
        def early_exit_loss(model,input,label):
            output = model.module_forward(input,module_list)
            if auxiliary_model is not None:
                output = auxiliary_model(output)
            loss = criterion(output,label)
            return loss
        model = copy.deepcopy(model) # avoid modifying global model
        model.to(self.device)
        if self.local_states is not None:
            model = self.load_local_state_dict(model,self.local_states)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        self.testloader = DataLoader(self.testset,batch_size=128, shuffle=False)

        adv_data_gen = Adv_Sample_Generator(early_exit_loss,
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
            if auxiliary_model is not None:
                outputs = auxiliary_model(outputs)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss/(batch_idx+1)
    
    def get_runtime_sys_stat(self):
        self.runtime_app,self.perf_degrade,self.mem_degrade=sample_runtime_app(self.rs)
        self.avail_perf = max([self.performance*self.perf_degrade,self.reserved_performance])
        self.avail_mem = max([self.memory*self.mem_degrade,self.reserved_memory])
        
        self.est_latency = self.model_profile.training_latency(module_list=self.module_list,
                                                               iters_per_input=self.iters_per_input,
                                                               batches=self.batches,
                                                               performance=self.avail_perf,
                                                               memory=self.avail_mem)
        # return the current availale performance, memory, and the training latency of the last round
        return self.runtime_app, self.avail_perf, self.avail_mem, self.latency, self.est_latency