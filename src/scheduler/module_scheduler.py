from scheduler.base_scheduler import base_AT_scheduler
import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis,parameter_count
from datasets import dataset_to_datafamily
from math import ceil

class module_scheduler(base_AT_scheduler):
    """
    This scheduler will profile the model and partition it into modules,
    which is used in fedprophet. This scheduler will also determine the 
    training stage of each communication round, and assign each chosen 
    client the training modules according to their available resources.
    """

    def __init__(self, args, model_profile, clients):
        super().__init__(args)
        self.model_profile = model_profile
        self.atom_list = self.model_profile.module_list
        self.atom_flops_dict = self.model_profile.flops_dict
        self.atom_param_dict = self.model_profile.num_parameter_dict
        self.atom_mem_dict = self.model_profile.mem_dict
        self.atom_output_dict = self.model_profile.out_feature_dict
        self.num_classes = model_profile.num_classes
        self.datafamily = dataset_to_datafamily[self.args["dataset"]]

        self.partition_module_list,self.module_dict,self.auxiliary_model_dict,\
        self.module_flops_dict,self.module_mem_dict,self.auxiliary_model_flops_dict, \
        self.auxiliary_model_mem_dict = self.model_partition(args["max_module_flops"],
                                                            args["max_module_mem"])
        
        print(f"====> Partitioned Model into {len(self.partition_module_list)} Modules.")
        print("====> Model Partitions: \n",self.partition_module_list)
        print("====> Module FLOPs: \n",self.module_flops_dict)
        print("====> Module Memory: \n", self.module_mem_dict)

        self.total_round = args["epochs"]
        self.round_per_stage = ceil(self.total_round/len(self.partition_module_list))
        self.stage = 0
        print("=================Stage 1=================")
        self.clients = clients
        self.available_performance = np.array([c.avail_perf for c in self.clients])
        self.available_memory = np.array([c.avail_mem for c in self.clients])


        self.adv_epsilon = args["adv_epsilon"]
        self.adv_alpha = args["adv_alpha"]
        self.adv_norm = args["adv_norm"]
        self.adv_bound = args["adv_bound"]

        self.mu = args["mu"]
        self.lamb = args["lamb"]
        self.psi = args["psi"]


    def training_params(self, idx,chosen_idxs, **kwargs):
        args = super().training_params()

        stage_module = self.partition_module_list[self.stage]
        stage_module_list = self.module_dict[stage_module]


        avail_perf = self.available_performance[idx]
        avail_mem = self.available_memory[idx]

        # config the prophet training
        # the allowed time is the time of finishing training 
        # the current module with the smallest available performance
        allowed_flops = avail_perf/min(self.available_performance[chosen_idxs])*self.module_flops_dict[stage_module]
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
        if self.psi == 1.0: # only train the prophet module
            args["stage_module_list"] = stage_module_list + prophet_module_list
            args["prophet_module_list"] = []
            if len(prophet_module_list) > 0:
                args["stage_aux_model_name"] = prophet_last_module
            else:
                args["stage_aux_model_name"] = stage_module
            args["prophet_aux_model_name"] = None
        elif self.psi == 0: # only train the stage module
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
        args["adv_epsilon"] = self.adv_epsilon
        args["adv_alpha"] = self.adv_alpha
        args["adv_norm"] = self.adv_norm
        args["adv_bound"] = self.adv_bound

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
        super().stat_update(epoch%self.round_per_stage) # periodic lr/adv_train adjustment
        new_stage = epoch//self.round_per_stage
        if new_stage != self.stage: # update the adversarial training params
            print("=================Stage %d================="%(new_stage+1))
            current_module_list = self.module_dict[self.partition_module_list[self.stage]]
            new_adv_epsilons = []
            lowers = []
            uppers = []
            for c in self.clients:
                max_adv_pert,l,u \
                    = c.forward_feature_set(global_model["model"],
                                            current_module_list,
                                            adv_train = self.args["adv_train"],
                                            adv_method = self.args["adv_method"],
                                            adv_epsilon = self.adv_epsilon,
                                            adv_alpha = self.adv_alpha,
                                            adv_T = self.args["adv_T"],
                                            adv_norm = self.adv_norm,
                                            adv_bound = self.adv_bound)
                new_adv_epsilons.append(max_adv_pert)
                lowers.append(l)
                uppers.append(u)
            # take the largest perturbation as the new epsilon
            self.adv_epsilon = max(new_adv_epsilons)
            self.adv_alpha = 2*self.adv_epsilon/self.args["adv_T"]
            self.adv_norm = 'l2'
            lower = min(lowers)
            upper = max(uppers)
            # usually end with relu, with bound [0.0,np.inf]
            self.adv_bound = [0 if lower>=0 else -np.inf,0 if upper<=0 else np.inf]

        
        self.stage = new_stage

        self.available_performance = np.array(sys_info["available_perfs"])
        self.available_memory = np.array(sys_info["available_mems"])

        # Todo: add adaptive adjustment of mu, lamb and psi
        

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

        for atom in self.atom_list:
            output_size = self.atom_output_dict[atom]
            auxiliary_in_fea = np.prod(output_size[1:])
            auxiliary_input = torch.rand(output_size)

            if self.datafamily == 'imagenet' and len(output_size)>2 and output_size[-1]>=7:
                # if the output is too large, we conduct avgpool first
                auxiliary_model = nn.Sequential(nn.AvgPool2d(7),nn.Flatten(),nn.Linear(auxiliary_in_fea//49,self.num_classes))
            elif self.datafamily == 'cifar' and len(output_size)>2 and output_size[-1]>=4:
                auxiliary_model = nn.Sequential(nn.AvgPool2d(4),nn.Flatten(),nn.Linear(auxiliary_in_fea//16,self.num_classes))
            else:
                auxiliary_model = nn.Sequential(nn.Flatten(),nn.Linear(auxiliary_in_fea,self.num_classes))
            
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
                current_partition_module_list = [atom]

            last_aux_flops = aux_flops
            last_aux_mem = self.model_profile.data_Byte*int(np.prod(self.atom_output_dict[atom])) \
                        + self.model_profile.param_mem_scale*self.model_profile.data_Byte*aux_param \
                        + self.model_profile.data_Byte*int(np.prod(aux_output))
            last_aux_model = auxiliary_model

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