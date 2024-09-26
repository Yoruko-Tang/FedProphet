from scheduler.base_scheduler import base_AT_scheduler
import numpy as np

class rbn_scheduler(base_AT_scheduler):

    def __init__(self, args, model_profile):
        super().__init__(args)
        
        self.model_profile = model_profile
        self.memory_req = model_profile.mem_dict['total']
        
        self.available_memory = None

        

    def training_params(self, idx, **kwargs):
        args = super().training_params()

        avail_mem = self.available_memory[idx]
        
        if avail_mem < self.memory_req:
            args['adv_train'] = False

        return args
    
    

    def stat_update(self, epoch, stat_info, sys_info, **kwargs):
        self.available_memory = np.array(sys_info["available_mems"])
        return super().stat_update(epoch,stat_info)