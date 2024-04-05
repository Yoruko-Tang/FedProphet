from scheduler.base_scheduler import base_AT_scheduler
import numpy as np

class pt_scheduler(base_AT_scheduler):

    def __init__(self, args, model_profile,neuron_num,sample_method):
        super().__init__(args)
        self.neuron_num = neuron_num
        self.model_profile = model_profile
        self.memory_req = model_profile.mem_dict['total']
        self.sample_method = sample_method
        self.available_memory = None

        

    def training_params(self, idx, **kwargs):
        args = super().training_params()

        avai_mem = self.available_memory[idx]
        frac = min([avai_mem/self.memory_req,1.0])
        neuron_dict = {m:None for m in self.neuron_num}
        
        for m in neuron_dict:
            if self.sample_method == 'HeteroFL': # HeteroFL
                neuron_dict[m] = np.arange(int(self.neuron_num[m]*frac))
            elif self.sample_method == 'FedDrop': # FedDropout
                neuron_dict[m] = np.random.choice(np.arange(self.neuron_num[m]),
                                                  int(self.neuron_num[m]*frac),
                                                  replace=False)
            elif self.sample_method == 'FedRolex': # FedRolex
                neuron_dict[m] = (self.round + np.arange(int(self.neuron_num[m]*frac)))%self.neuron_num[m]

        args["neuron_dict"] = neuron_dict
        args["partial_frac"] = frac

        return args
    
    

    def stat_update(self, epoch, sys_info, **kwargs):
        self.available_memory = np.array(sys_info["available_mems"])
        return super().stat_update(epoch)