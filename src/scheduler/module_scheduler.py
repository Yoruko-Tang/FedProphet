from scheduler.base_scheduler import base_AT_scheduler
import numpy as np
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis,parameter_count
from datasets import dataset_to_datafamily

class module_scheduler(base_AT_scheduler):
    """
    This scheduler will profile the model and partition it into modules,
    which is used in fedprophet. This scheduler will also determine the 
    training stage of each communication round, and assign each chosen 
    client the training modules according to their available resources.
    """

    def __init__(self, args, model_profile):
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
            self.module_flops_dict,self.module_mem_dict = self.model_partition(args["max_module_flops"],args["max_module_mem"])
        

    def model_partition(self,max_module_flops=None,max_module_mem=None,**kwargs):
        """
        partition the model in a greedy manner, with each module in the 
        max_flops and max_mem constraints
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
            
            aux_flops = FlopCountAnalysis(auxiliary_model,auxiliary_input).by_module()['']
            aux_param = parameter_count(auxiliary_model)['']
            aux_output = [output_size[0],self.num_classes]
            # previous flops + flops of this atom + flops of auxiliary model
            added_flops = current_sum_flops + int(self.atom_flops_dict[atom]) + aux_flops
            # previous mem + mem of this atom + mem of input of auxiliary + mem of auxiliary model + mem of output of auxiliary model
            added_mem = current_sum_mem + int(self.atom_mem_dict[atom]) \
                        + 4*int(np.prod(self.atom_output_dict[atom])) \
                        + 3*4*aux_param + 4*int(np.prod(aux_output))
            
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
                
                # start the next module
                current_sum_flops = int(self.atom_flops_dict[atom])
                current_sum_mem = int(self.atom_mem_dict[atom])
                current_partition_module_list = [atom]

            last_aux_flops = aux_flops
            last_aux_mem = 4*int(np.prod(self.atom_output_dict[atom])) \
                            + 3*4*aux_param + 4*int(np.prod(aux_output))
            last_aux_model = auxiliary_model

        # add the last module into the list
        last_module_name = "+".join(current_partition_module_list)
        module_dict[last_module_name] = current_partition_module_list
        auxiliary_model_dict[last_module_name] = None
        module_flops_dict[last_module_name] = current_sum_flops
        module_mem_dict[last_module_name] = current_sum_mem

        partition_module_list = list(module_dict.keys())

        return partition_module_list,module_dict,auxiliary_model_dict,module_flops_dict,module_mem_dict