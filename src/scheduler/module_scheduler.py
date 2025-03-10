from scheduler.base_scheduler import base_AT_scheduler
import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis,parameter_count
from math import ceil
import copy
import os.path as osp
import os
import pickle

class module_scheduler(base_AT_scheduler):
    """
    This scheduler will profile the model and partition it into modules,
    which is used in fedprophet. This scheduler will also determine the 
    training stage of each communication round, and assign each chosen 
    client the training modules according to their available resources.
    """

    def __init__(self, args, model_profile, clients,log_path=None):
        super().__init__(args)
        self.model_profile = model_profile
        #self.stage_input_size = model_profile.inputsize
        self.atom_list = self.model_profile.module_list
        self.atom_flops_dict = self.model_profile.flops_dict
        self.atom_param_dict = self.model_profile.num_parameter_dict
        self.atom_mem_dict = self.model_profile.mem_dict
        self.atom_output_dict = self.model_profile.out_feature_dict
        self.num_classes = model_profile.num_classes
        # self.datafamily = dataset_to_datafamily[self.args["dataset"]]

        self.partition_module_list,self.module_dict,self.auxiliary_model_dict,\
        self.module_flops_dict,self.module_mem_dict,self.auxiliary_model_flops_dict, \
        self.auxiliary_model_mem_dict = self.model_partition(None,args["reserved_mem"])
        
        print(f"====> Partitioned Model into {len(self.partition_module_list)} Modules.")
        print("====> Model Partitions: \n",self.partition_module_list)
        print("====> Module FLOPs: \n",self.module_flops_dict)
        print("====> Module Memory: \n", self.module_mem_dict)
        #print(self.auxiliary_model_dict)

        self.round_per_stage = ceil(self.total_round/len(self.partition_module_list))
        self.stage = 0
        self.stage_begin_round = 0
        self.best_clean_adv_ratio = None
        self.last_stage_clean_adv_ratio = 1.0
        self.clean_adv_ratios = []

        


        # print("=================Stage 1=================")
        self.clients = clients
        self.available_performance = None
        self.available_memory = None


        self.adv_epsilon = args["adv_epsilon"]
        self.adv_alpha = args["adv_alpha"]
        self.adv_norm = args["adv_norm"]
        self.adv_bound = args["adv_bound"]
        self.alpha = 1.0
        self.best_alpha = self.alpha

        self.mu = args["mu"]
        self.lamb = args["lamb"]
        self.psi = args["psi"]

        self.logs = []
        self.log_path = log_path
        if self.log_path is not None:
            if not osp.exists(self.log_path):
                os.makedirs(self.log_path)
            self.tsv_file = osp.join(self.log_path, 'scheduler.log.tsv')
            self.pkl_file = osp.join(self.log_path, 'scheduler.pkl')
            with open(self.tsv_file, 'a') as wf:
                columns = ['epoch', 'adv_epsilon', 'adv_alpha', 'adv_norm', 'adv_bound', 'mu','lamb','psi','best_weighted_acc','best_clean_adv_ratio']
                wf.write('\t'.join(columns) + '\n')

        


    def training_params(self, idx,chosen_idxs, **kwargs):
        args = super().training_params()

        stage_module = self.partition_module_list[self.stage]
        stage_module_list = self.module_dict[stage_module]


        avail_perf = self.available_performance[idx]
        avail_mem = self.available_memory[idx]

        # if args["adv_train"]:
        #     # reserve memory for adversarial training
        #     avail_mem -= np.prod(self.stage_input_size)*self.model_profile.data_Byte*args["adv_ratio"]
        # config the prophet training
        # the allowed time is the time of finishing training 
        # the current module with the smallest available performance
        allowed_flops = avail_perf/min(self.available_performance[chosen_idxs])*self.module_flops_dict[stage_module]
        #allowed_flops = np.inf
        cum_flops = self.module_flops_dict[stage_module]
        cum_mem = self.module_mem_dict[stage_module]


        prophet_module_list = []
        prophet_last_module = None

        for i in range(self.stage+1,len(self.partition_module_list)):
            next_module = self.partition_module_list[i]
            cum_flops += self.module_flops_dict[next_module]
            cum_mem += self.module_mem_dict[next_module]

            if i == self.stage + 1: # subtract the output size of the stage module since it's counted twice
                cum_mem -= self.model_profile.data_Byte*int(np.prod(self.atom_output_dict[stage_module_list[-1]]))
                        
            else:# subtract the flops and memory of the last auxiliary model
                cum_flops -= self.auxiliary_model_flops_dict[prophet_last_module]
                cum_mem -= self.auxiliary_model_mem_dict[prophet_last_module]
            
            if cum_flops < allowed_flops and cum_mem < avail_mem:
                prophet_module_list += self.module_dict[next_module]
                prophet_last_module = next_module
                
            else:
                break

        # modularization configuration
        if self.psi == 0: # only train the stage module
            args["stage_module_list"] = stage_module_list
            args["prophet_module_list"] = []
            args["stage_aux_model_name"] = stage_module
            args["prophet_aux_model_name"] = None
        else: # train both stage module and prophet module
            args["stage_module_list"] = stage_module_list
            args["prophet_module_list"] = prophet_module_list
            args["stage_aux_model_name"] = stage_module
            args["prophet_aux_model_name"] = prophet_last_module


        args["mu"] = self.mu
        args["lamb"] = self.lamb
        args["psi"] = self.psi

        # adversarial training configuration
        args["adv_epsilon"] = self.adv_epsilon*self.alpha
        args["adv_alpha"] = self.adv_alpha*self.alpha
        args["adv_norm"] = self.adv_norm
        args["adv_bound"] = self.adv_bound

        if self.args["stage_lr_decay"] is not None:
            args["lr"] *= self.args["stage_lr_decay"]**self.stage


        return args
    
    def monitor_params(self, **kwargs):
        args = super().monitor_params(**kwargs)
        module_list = []
        for i in range(self.stage+1):
            stage_module = self.partition_module_list[i]
            module_list += self.module_dict[stage_module]
        
        
        args["module_list"] = module_list
        args["aux_module_name"] = stage_module


        return args

    def stat_update(self, epoch, stat_info, sys_info, global_model, **kwargs):
        new_logs = {}

        # update systematic information
        self.available_performance = np.array(sys_info["available_perfs"])
        self.available_memory = np.array(sys_info["available_mems"])
        
        # update statistical information
        if not self.args["adv_train"] or self.round >= self.args["adv_warmup"]+int(0.05*self.round_per_stage):
            if "weighted_val_adv_acc" in stat_info:
                weighted_acc = stat_info["weighted_val_acc"] + self.last_stage_clean_adv_ratio*stat_info["weighted_val_adv_acc"]
                clean_adv_ratio = stat_info["weighted_val_acc"]/stat_info["weighted_val_adv_acc"]
                self.clean_adv_ratios.append(clean_adv_ratio)
            else:
                weighted_acc = stat_info["weighted_val_acc"]
                clean_adv_ratio = None
            
            
            if weighted_acc > self.best_weighted_acc:
                self.best_weighted_acc = weighted_acc
                self.smooth_length = 0
                self.best_model = copy.deepcopy(global_model)
                self.best_local_states = [copy.deepcopy(c.local_states) for c in self.clients]
                self.best_clean_adv_ratio = clean_adv_ratio
                self.best_alpha = self.alpha
            else:
                self.smooth_length += epoch-self.stage_begin_round-self.round
        
        # adaptive adjustment of alpha
        if self.args["adv_train"] and self.args["adapt_eps"] \
            and self.stage>0 and self.round > self.args["adv_warmup"]+int(0.05*self.round_per_stage) \
            and self.round%int(0.1*self.round_per_stage)==0:
            screen_length = min([5,len(self.clean_adv_ratios)])
            if np.mean(self.clean_adv_ratios[-screen_length:]) > 1.05*self.last_stage_clean_adv_ratio: # the adv acc is too low
                self.alpha = self.alpha + 0.1 # increase the epsilon
                # self.smooth_length = 0 # clear the smooth length for adjusting alpha
                # self.best_weighted_acc = 0
                # self.best_clean_adv_ratio = None
            elif np.mean(self.clean_adv_ratios[-screen_length:]) < 0.95*self.last_stage_clean_adv_ratio: # the clean acc is too low
                self.alpha = max([self.args["eps_quantile"],self.alpha-0.1]) # decrease the epsilon
                # self.smooth_length = 0 # clear the smooth length for adjusting alpha
                # self.best_weighted_acc = 0
                # self.best_clean_adv_ratio = None
    
        
        self.round = epoch-self.stage_begin_round
        if self.round >= self.round_per_stage or self.smooth_length >= int(0.1*self.round_per_stage): 
            # move to the next stage
            if self.stage == len(self.partition_module_list)-1:
                # stop training
                return False
            
            self.stage_begin_round = epoch
            self.round = 0
            self.smooth_length = 0
            self.best_weighted_acc = 0.0
            # if self.stage == 0:
            #     self.first_stage_clean_adv_ratio = self.best_clean_adv_ratio
            self.last_stage_clean_adv_ratio = self.best_clean_adv_ratio
            self.best_clean_adv_ratio = None
            self.clean_adv_ratios = []
            

            # recover to the best model before
            global_model.update(self.best_model) 
            for n,c in enumerate(self.clients):
                c.local_states = self.best_local_states[n]

            # forward local feature set
            current_module_list = self.module_dict[self.partition_module_list[self.stage]]
            #self.stage_input_size = self.atom_output_dict[current_module_list[-1]]
            new_adv_epsilons = []
            lowers = []
            uppers = []
            for c in self.clients:
                adv_pert,l,u \
                    = c.forward_feature_set(global_model["model"],
                                            current_module_list,
                                            adv_train = self.args["adv_train"],
                                            adv_method = self.args["adv_method"],
                                            adv_epsilon = self.adv_epsilon*self.best_alpha,
                                            adv_alpha = self.adv_alpha*self.best_alpha,
                                            adv_T = self.args["adv_T"],
                                            adv_norm = self.adv_norm,
                                            adv_bound = self.adv_bound)
                new_adv_epsilons.append(adv_pert)
                lowers.append(l)
                uppers.append(u)
            # take the largest/mean/median perturbation as the new epsilon
            
            if self.args["adv_train"]:
                new_adv_epsilons = np.concatenate(new_adv_epsilons)
                self.adv_epsilon = np.mean(new_adv_epsilons)
                if self.args["int_adv_norm"] == 'inf': # transfer the l2 norm to linf norm
                    self.adv_epsilon = np.sqrt(self.adv_epsilon**2/np.prod(self.atom_output_dict[current_module_list[-1]][1:]))
                self.adv_alpha = 2.5*self.adv_epsilon/self.args["adv_T"]
                self.adv_norm = self.args["int_adv_norm"]
                lower = min(lowers)
                upper = max(uppers)
                # usually end with relu, with bound [0.0,np.inf]
                self.adv_bound = [0 if lower>=0 else -np.inf,0 if upper<=0 else np.inf]
                self.alpha = self.args["eps_quantile"]
                self.best_alpha = self.alpha

                new_logs.update({"adv_epsilons":new_adv_epsilons,
                                 "adv_epsilon":self.adv_epsilon,
                                 "adv_alpha":self.adv_alpha,
                                 "adv_norm":self.adv_norm,
                                 "adv_bound":self.adv_bound})
        
            self.stage += 1
        
        
        print('\n | Global Training Round : {} |\t | Stage : {}\t Round : {} |\n'.format(self.round+self.stage_begin_round,self.stage,self.round))
        

        # log
        new_logs.update({"mu":self.mu,
                        "lamb":self.lamb,
                        "psi":self.psi,
                        "alpha":self.alpha,
                        "best_acc":self.best_weighted_acc})
        self.logs.append(new_logs)
        
        log_column = [epoch,self.adv_epsilon*self.alpha,self.adv_alpha*self.alpha,self.adv_norm,self.adv_bound,self.mu,self.lamb,self.psi,self.best_weighted_acc,self.best_clean_adv_ratio]
        
        with open(self.tsv_file, 'a') as af:
            af.write('\t'.join([str(c) for c in log_column]) + '\n')
        with open(self.pkl_file,'wb') as stat_f:
            pickle.dump(self.logs, stat_f)
            
        return True
        

    def model_partition(self,max_module_flops=None,max_module_mem=None,**kwargs):
        """
        partition the model in a greedy manner, with each module in the 
        max_flops and max_mem constraints.
        Returns:
        partition_module_list: list of module names, with format 'atom1+atom2+...'.
        module_dict: {module name: atom list}, e.g., 'atom1+atom2':[atom1,atom2].
        auxiliary_model_dict: {module name: auxiliary model}.
        module_flops_dict: {module name: module flops}.
        module_mem_dict: {module name: module memory requirement}.
        """
        
        

        # use the largest atom layer as the lower bound
        if max_module_flops is None:
            max_module_flops = np.inf
        else:
            assert sorted(self.atom_flops_dict.values())[-2] < max_module_flops, "Max allowed flops is too small to partition!"
        if max_module_mem is None: 
            max_module_mem = np.inf
        else:
            assert sorted(self.atom_mem_dict.values())[-2] < max_module_mem, "Max allowed memory is too small to partition!"

        
        module_dict = {} # module name: module list
        auxiliary_model_dict = {}
        module_flops_dict = {}
        module_mem_dict = {}
        aux_model_flops_dict = {}
        aux_model_mem_dict = {}


        current_partition_module_list = []

        current_sum_flops = 0
        current_sum_mem = 0
        #input_size = self.model_profile.inputsize
        #adv_reserved_mem = self.model_profile.data_Byte*np.prod(input_size) if adv_train else 0

        for atom in self.atom_list:
            output_size = self.atom_output_dict[atom]
            auxiliary_in_fea = np.prod(output_size[1:])
            auxiliary_input = torch.rand(output_size)

            if len(output_size) == 4: # CNN models, nxcxhxw
                output_width = ceil(np.sqrt(self.num_classes/output_size[1]))
                #output_width = max([ceil(np.sqrt(self.num_classes/output_size[1])),ceil(output_size[2]/max_pooling_size)])
                #output_width = max([int(np.sqrt(max_auxiliary_neuron/output_size[1])),1])
                # if the output is too large, we conduct avgpool first
                if output_size[2] > output_width:
                    auxiliary_model = nn.Sequential(nn.AdaptiveAvgPool2d((output_width,output_width)),nn.Flatten(),nn.Linear(output_size[1]*output_width**2,self.num_classes))
                else:
                    auxiliary_model = nn.Sequential(nn.Flatten(),nn.Linear(auxiliary_in_fea,self.num_classes))
            elif len(output_size) == 3: # transformer models, nxlxd
                auxiliary_model = nn.Sequential(#nn.LayerNorm(output_size[2]),
                                                nnMean(),nn.Linear(output_size[2],self.num_classes))
            else:
                auxiliary_model = nn.Sequential(nn.Linear(auxiliary_in_fea,self.num_classes))
            
            aux_flops = FlopCountAnalysis(auxiliary_model,auxiliary_input)
            aux_flops.unsupported_ops_warnings(False)
            aux_flops = aux_flops.by_module()['']
            aux_param = parameter_count(auxiliary_model)['']
            aux_output = [output_size[0],self.num_classes]
            # previous flops + flops of this atom + flops of auxiliary model
            added_flops = current_sum_flops + int(self.atom_flops_dict[atom]) + aux_flops
            # previous mem + mem of this atom + mem of input of auxiliary + mem of auxiliary model + mem of output of auxiliary model
            added_mem = current_sum_mem + int(self.atom_mem_dict[atom]) \
                    + self.model_profile.data_Byte*int(np.prod(self.atom_output_dict[atom])) \
                    + self.model_profile.param_mem_scale*self.model_profile.data_Byte*aux_param \
                    + self.model_profile.data_Byte*int(np.prod(aux_output))
            
            # this module can be added into the current module
            if added_flops <= max_module_flops and added_mem <= max_module_mem:
                current_sum_flops += (int(self.atom_flops_dict[atom]))
                current_sum_mem += int(self.atom_mem_dict[atom])
                current_partition_module_list.append(atom)

            # this module cannot be added into the current module
            else:
                # end of this module
                module_name = "+".join(current_partition_module_list)
                if len(module_name)>0:
                    # add the flops and memory of the last auxiliary model
                    module_dict[module_name] = current_partition_module_list
                    auxiliary_model_dict[module_name] = last_aux_model
                    module_flops_dict[module_name] = current_sum_flops + last_aux_flops
                    module_mem_dict[module_name] = current_sum_mem + last_aux_mem
                    aux_model_flops_dict[module_name] = last_aux_flops
                    aux_model_mem_dict[module_name] = last_aux_mem
                
                # start the next module
                current_sum_flops = int(self.atom_flops_dict[atom])
                current_sum_mem = int(self.atom_mem_dict[atom])
                # if adv_train:
                #     # reserve memory for adversarial training
                #     adv_reserved_mem = self.model_profile.data_Byte*np.prod(input_size)
                current_partition_module_list = [atom]

            last_aux_flops = aux_flops
            last_aux_mem = self.model_profile.data_Byte*int(np.prod(self.atom_output_dict[atom])) \
                        + self.model_profile.param_mem_scale*self.model_profile.data_Byte*aux_param \
                        + self.model_profile.data_Byte*int(np.prod(aux_output))
            last_aux_model = auxiliary_model
            #input_size = output_size

        # add the last module into the list
        last_module_name = "+".join(current_partition_module_list)
        module_dict[last_module_name] = current_partition_module_list
        auxiliary_model_dict[last_module_name] = None
        module_flops_dict[last_module_name] = current_sum_flops
        module_mem_dict[last_module_name] = current_sum_mem
        aux_model_flops_dict[last_module_name] = 0
        aux_model_mem_dict[last_module_name] = 0

        partition_module_list = list(module_dict.keys())

        return partition_module_list,module_dict,auxiliary_model_dict,module_flops_dict,module_mem_dict,aux_model_flops_dict,aux_model_mem_dict
    
class nnMean(nn.Module):
    """
    Use as a module for the auxiliary model of transformer models
    """
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x.mean(dim=1)
    
class nnCLS(nn.Module):
    """
    Use as a module for the auxiliary model of transformer models
    """
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x[:,0]