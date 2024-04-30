from scheduler.base_scheduler import base_AT_scheduler
import numpy as np

class kd_scheduler(base_AT_scheduler):

    def __init__(self, args, model_profiles):
        super().__init__(args)
        self.model_profiles = model_profiles
        self.memory_reqs = [mp.mem_dict['total'] for mp in self.model_profiles]
        self.mem_req_sort = np.argsort(self.memory_reqs)# sort the memory req in ascent order

        self.available_memory = None

        

    def training_params(self, idx, **kwargs):
        args = super().training_params()

        avail_mem = self.available_memory[idx]
        if args["adv_train"]:
            # reserve memory for adversarial training
            avail_mem -= np.prod(self.model_profile.inputsize)*self.model_profile.data_Byte*args["adv_ratio"]
        group = None
        for i in self.mem_req_sort: # find the largest model that can fit into the memory
            if avail_mem>self.memory_reqs[i]:
                group = i
        if group is None:
            group = self.mem_req_sort[0]
        
        
        args["model_idx"] = group
        args["model_profile"] = self.model_profiles[group]
        return args
    
    def monitor_params(self, **kwargs):
        args = super().monitor_params()

        args["model_idx"] = None
 

        return args

    def stat_update(self, epoch,stat_info, sys_info, **kwargs):
        self.available_memory = np.array(sys_info["available_mems"])
        return super().stat_update(epoch,stat_info)