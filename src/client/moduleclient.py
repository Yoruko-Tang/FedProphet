from client.atclient import AT_Client

import torch
from torch.utils.data import DataLoader,TensorDataset
import torch.nn
import copy
import numpy as np

from utils.adversarial import Adv_Sample_Generator
from hardware.sys_utils import sample_runtime_app,sample_networks,model_summary



class Module_Client(AT_Client):
    """
    This is a client that conducts adversarial training with fedprophet
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

        self.trainset, self.testset = self.train_test(dataset, list(data_idxs))
        self.feature_trainset = self.trainset
        # self.feature_testset = self.testset
        self.dev_name,self.performance,self.memory,self.eff_bw=sys_info
        self.model_profile = model_profile

        self.local_state_preserve = local_state_preserve
        if local_state_preserve:
            self.local_states = copy.deepcopy(init_local_state)
        else:
            self.local_states = None
        
        self.device = device
        self.final_local_loss = None
        self.verbose = verbose
        self.rs = np.random.RandomState(random_seed)

        self.reserved_performance = reserved_performance
        self.reserved_memory = reserved_memory
        
        self.iters_per_input = test_adv_T+1

        self.batches = None
        self.latency = None
        self.module_list = None
        self.get_runtime_sys_stat()

        self.test_adv_method = test_adv_method
        self.test_adv_epsilon = test_adv_epsilon
        self.test_adv_alpha = test_adv_alpha
        self.test_adv_T = test_adv_T
        self.test_adv_norm = test_adv_norm
        self.test_adv_bound = test_adv_bound
 
        
        
    def forward_feature_set(self,model,module_list,adv_train=True,
                            adv_method='pgd',adv_epsilon=0.0,adv_alpha=0.0,
                            adv_T=0,adv_norm='inf',adv_bound=[0.0,1.0]):
        """
        forward the current feature to the feature of the next stage
        """
        def feature_adv_loss(model,input,clean_output):
            output = model.module_forward(input,module_list)
            loss = torch.sum((output-clean_output)**2)/len(output)
            return loss

        model = copy.deepcopy(model) # avoid modifying global model
        model.to(self.device)
        if self.local_states is not None:
            model = self.load_local_state_dict(model,self.local_states)
        
        model.eval()

        trainloader = DataLoader(self.feature_trainset,batch_size=128,shuffle=False)
        # testloader = DataLoader(self.feature_testset,batch_size=128, shuffle=False)

        new_train_feature,new_test_feature = [],[]
        train_labels,test_labels = [],[]
        train_adv_feature_pert = []
        if adv_train:
            adv_data_gen = Adv_Sample_Generator(feature_adv_loss,adv_method,adv_epsilon,
                                            adv_alpha,adv_T,adv_norm,adv_bound)

        for datas,labels in trainloader:
            datas = datas.to(self.device)
            new_feature = model.module_forward(datas,module_list)
            new_feature = new_feature.detach()
            if adv_train:
                adv_datas = adv_data_gen.attack_data(model,datas,new_feature)
                new_adv_feature = model.module_forward(adv_datas,module_list)
                adv_feature_pert = torch.norm(new_adv_feature-new_feature,p=2,dim=list(range(1,new_feature.dim())))
                train_adv_feature_pert.append(adv_feature_pert.detach().cpu().numpy())
            new_train_feature.append(new_feature.cpu())
            train_labels.append(labels.cpu())
        new_train_feature = torch.cat(new_train_feature,dim=0)
        train_labels = torch.cat(train_labels,dim=0)
        smallest_value = torch.min(new_train_feature).item()
        largest_value = torch.max(new_train_feature).item()

        # for datas,labels in testloader:
        #     datas = datas.to(self.device)
        #     new_feature = model.module_forward(datas,module_list)
        #     new_feature = new_feature.detach()
        #     # if adv_train:
        #     #     adv_datas = adv_data_gen.attack_data(model,datas,new_feature)
        #     #     new_adv_feature = model.module_forward(adv_datas,module_list)
        #     #     adv_feature_pert = torch.norm(new_adv_feature-new_feature,p=2,dim=list(range(1,new_feature.dim()))).max()
        #     #     test_adv_feature_pert.append(adv_feature_pert.item())
        #     new_test_feature.append(new_feature.cpu())
        #     test_labels.append(labels.cpu())
        # new_test_feature = torch.cat(new_test_feature,dim=0)
        # test_labels = torch.cat(test_labels,dim=0)

        
            

        self.feature_trainset = TensorDataset(new_train_feature,train_labels)
        # self.feature_testset = TensorDataset(new_test_feature,test_labels)
        if adv_train:
            train_adv_feature_pert = np.concatenate(train_adv_feature_pert)
            # max_test_adv_feature_pert = max(test_adv_feature_pert)


        return train_adv_feature_pert, smallest_value, largest_value


        
    def train(self,model,aux_models,stage_module_list,prophet_module_list,
              stage_aux_model_name,prophet_aux_model_name,
              local_ep,local_bs,lr,optimizer='sgd',momentum=0.0,reg=0.0, 
              criterion=torch.nn.CrossEntropyLoss(),
              adv_train=True,adv_method='pgd',adv_epsilon=0.0,adv_alpha=0.0,adv_T=0,
              adv_norm='inf',adv_bound=[0.0,1.0],adv_ratio=1.0,
              mu=0.0,lamb=0.0,psi=0.0,model_profile=None,**kwargs):
        """train the model for one communication round."""

        # model preparation
        model = copy.deepcopy(model) # avoid modifying global model
        model.to(self.device)
        if self.local_states is not None:
            model = self.load_local_state_dict(model,self.local_states)
        

        if stage_aux_model_name is not None:
            current_aux_model = copy.deepcopy(aux_models[stage_aux_model_name])
            if current_aux_model is not None:
                current_aux_model.to(self.device)
        else:
            current_aux_model = None
        
        if prophet_aux_model_name is not None:
            future_aux_model = copy.deepcopy(aux_models[prophet_aux_model_name])
            if future_aux_model is not None:
                future_aux_model.to(self.device)
        else:
            future_aux_model = None

        def fedprophet_loss(model,input,label,output_dict=None):
            output = model.module_forward(input,stage_module_list)
            if current_aux_model is not None:
                final_output = current_aux_model(output)
                stage_task_loss = criterion(final_output,label)
                stage_cvx_loss = mu*torch.sum(torch.square(output))/(2*len(output))
                if output_dict is not None:
                    output_dict.update({"stage_task_loss":stage_task_loss.item(),"stage_cvx_loss":stage_cvx_loss.item()})
            else:
                final_output = output
                stage_task_loss = criterion(final_output,label)
                stage_cvx_loss = 0
                if output_dict is not None:
                    output_dict.update({"stage_task_loss":stage_task_loss.item(),"stage_cvx_loss":0})
            
            
            loss = stage_task_loss+stage_cvx_loss
            
            
            if len(prophet_module_list)>0:
                prophet_output = model.module_forward(output,prophet_module_list)
                if future_aux_model is not None:
                    prophet_final_output = future_aux_model(prophet_output)
                    prophet_task_loss = criterion(prophet_final_output,label)
                    prophet_cvx_loss = mu*torch.sum(torch.square(prophet_output))/(2*len(prophet_output))
                    if output_dict is not None:
                        output_dict.update({"prophet_task_loss":prophet_task_loss.item(),"prophet_cvx_loss":prophet_cvx_loss.item()})
                else:
                    prophet_final_output = prophet_output
                    prophet_task_loss = criterion(prophet_final_output,label)
                    prophet_cvx_loss = 0
                    if output_dict is not None:
                        output_dict.update({"prophet_task_loss":prophet_task_loss.item(),"prophet_cvx_loss":0})
                
                
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
        # normalize the step size of the stage aux model
        #alr = lr/(1-psi) if len(prophet_module_list)>0 else lr
        alr = lr
        if optimizer == 'sgd':
            opt = torch.optim.SGD([{'params':np,'lr':lr,'weight_decay':reg},
                                        {'params':anp,'lr':alr,'weight_decay':lamb},
                                        {'params':panp,'lr':lr,'weight_decay':lamb}], 
                                        momentum=momentum)
        elif optimizer == 'adam':
            # normalize the step size of the stage aux model
            opt = torch.optim.Adam([{'params':np,'lr':lr,'weight_decay':reg},
                                         {'params':anp,'lr':alr,'weight_decay':lamb},
                                         {'params':panp,'lr':lr,'weight_decay':lamb}])
        

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
                    datas = adv_data_gen.attack_data(model,datas,labels,adv_ratio)
                else:
                    self.iters_per_input = 1
                
                model.train()
                model.zero_grad()
                output_losses = {}
                loss = fedprophet_loss(model,datas,labels,output_losses)
                loss.backward()
                opt.step()
                iters += 1
                if iters == local_ep:
                    break

            if self.verbose:
                com_out = 'Local Epoch : {}/{} \t| Loss: {:.4f}'.format(iters, local_ep, loss.item())
                extra_out = ['{}: {:.4f}'.format(k,output_losses[k]) for k in output_losses.keys()]
                extra_out = "\t| ".join(extra_out)
                print(com_out + "\t| " + extra_out)
        

        
        if self.local_state_preserve:
            self.local_states = copy.deepcopy(self.get_local_state_dict(model))
        self.final_local_loss = loss.item()
        
        # calculate training latency
        if model_profile is not None:
            self.model_profile = model_profile
        self.module_list = stage_module_list + prophet_module_list
        self.latency = self.model_profile.training_latency(module_list=self.module_list,
                                                           batches=self.batches,
                                                           iters_per_input=self.iters_per_input,
                                                           performance=self.avail_perf,
                                                           memory=self.avail_mem,
                                                           eff_bandwidth=self.eff_bw,
                                                           network_bandwidth=self.network_speed,
                                                           network_latency=self.network_latency)
        
        return model,current_aux_model,future_aux_model
        
    def validate(self,model,aux_models,module_list,
                 aux_module_name = None,testset=None,
                 criterion=torch.nn.CrossEntropyLoss(),**kwargs):
        """ Returns the validation accuracy and loss."""
        
        model = copy.deepcopy(model) # avoid modifying global model
        model.to(self.device)
        if self.local_states is not None:
            model = self.load_local_state_dict(model,self.local_states)
        
        model.eval()
        
        if aux_module_name is not None:
            aux_model = aux_models[aux_module_name]
        else:
            aux_model = None

        loss, total, correct = 0.0, 0.0, 0.0
        if testset is None:
            testset = self.testset
        testloader = DataLoader(testset,batch_size=128, shuffle=False)

        for batch_idx, (datas, labels) in enumerate(testloader):
            datas, labels = datas.to(self.device), labels.to(self.device)
            outputs = model.module_forward(datas,module_list)
            if aux_model is not None:
                outputs = aux_model(outputs)
                
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss/(batch_idx+1)
    
    def adv_validate(self,model,aux_models,module_list,
                     aux_module_name = None, testset=None,
                     criterion=torch.nn.CrossEntropyLoss(),**kwargs):
        """ Returns the validation adversarial accuracy and adversarial loss."""
        
        model = copy.deepcopy(model) # avoid modifying global model
        model.to(self.device)
        if self.local_states is not None:
            model = self.load_local_state_dict(model,self.local_states)
        
        model.eval()

        if aux_module_name is not None:
            aux_model = aux_models[aux_module_name]
        else:
            aux_model = None

        def early_exit_loss(model,input,label):
            output = model.module_forward(input,module_list)
            if aux_model is not None:
                output = aux_model(output)
            loss = criterion(output,label)
            return loss
        
        loss, total, correct = 0.0, 0.0, 0.0
        if testset is None:
            testset = self.testset
        self.testloader = DataLoader(testset,batch_size=128, shuffle=False)

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
            outputs = model.module_forward(adv_datas,module_list)
            if aux_model is not None:
                outputs = aux_model(outputs)
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
        
        self.network,self.network_speed,self.network_latency = sample_networks(self.rs)

        if self.model_profile is not None:
            self.est_latency = \
                self.model_profile.training_latency(module_list=self.module_list,
                                                    iters_per_input=self.iters_per_input,
                                                    batches=self.batches,
                                                    performance=self.avail_perf,
                                                    memory=self.avail_mem,
                                                    eff_bandwidth=self.eff_bw,
                                                    network_bandwidth=self.network_speed,
                                                    network_latency=self.network_latency)
        else:
            self.est_latency = None
        # return the current availale performance, memory, and the training latency of the last round
        return self.runtime_app, self.avail_perf, self.avail_mem, self.network,self.network_speed,self.network_latency, self.latency, self.est_latency