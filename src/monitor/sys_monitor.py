from hardware.sys_utils import training_latency
import os.path as osp
import os
import numpy as np
import pickle

class Sys_Monitor():
    """
    collect the validation loss of each client
    """
    def __init__(self,clients,log_path=None):
        self.clients=clients
        self.chosen_clients = []
        self.chosen_devices = []
        self.runtime_apps = []
        self.avail_perfs = []
        self.avail_mems = []
        # training latency
        self.train_times = []
        self.round_times = []
        self.total_times = []

        self.epochs = []
        
        # create log file
        self.log_path = log_path
        if self.log_path is not None:
            if not osp.exists(self.log_path):
                os.makedirs(self.log_path)
            self.tsv_file = osp.join(self.log_path, 'sys.log.tsv')
            self.pkl_file = osp.join(self.log_path, 'sys.pkl')
            with open(self.tsv_file, 'w') as wf:
                columns = ['epoch', 'min_avail_perf', 'max_avail_perf', 
                           'min_avail_mem', 'max_avail_mem', 
                           'round_time', 'total_time']
                wf.write('\t'.join(columns) + '\n')
        
    def collect(self,model,epoch=None,chosen_idxs=None,log=False,**kwargs):
        runtime_app = [] # running apps of all clients
        avail_perf = [] # available performance of all clients
        avail_mem = []  # available memory of all clients
        train_times = [] # trainning latency of all clients
        for n,c in enumerate(self.clients):
            # collect performance and memory
            cruntime_app,cavail_perf,cavail_mem = c.get_runtime_sys_stat()
            runtime_app.append(cruntime_app)
            avail_perf.append(cavail_perf)
            avail_mem.append(cavail_mem)
            
            # collect training latency
            if c.batches is not None:
                if isinstance(model,list):
                    train_times.append(training_latency(model[n],c.batches,cavail_perf,cavail_mem))
                else:
                    train_times.append(training_latency(model,c.batches,cavail_perf,cavail_mem))

            else:
                train_times.append(0)
        
        
        round_time = np.max(np.array(train_times)[chosen_idxs])
        total_time = self.total_times[-1]+round_time
        
        
        if log:
            self.epochs.append(epoch)
            self.chosen_clients.append(chosen_idxs)
            if chosen_idxs is not None:
                chosen_dev = [self.clients[idx].dev_name for idx in chosen_idxs]
            else:
                chosen_dev = [c.dev_name for c in self.clients]
            self.chosen_devices.append(chosen_dev)
            self.runtime_apps.append(runtime_app)
            self.avail_perfs.append(avail_perf)
            self.avail_mems.append(avail_mem)

            self.train_times.append(train_times)
            self.round_times.append(round_time)
            self.total_times.append(total_time)

        res = {"epoch":epoch,
               "runtime_apps":runtime_app,
               "available_perfs":avail_perf,
               "available_mems":avail_mem,
               "train_times":train_times,
               "round_time":round_time,
               "total_time":total_time}
        
        return res
    
    def log(self):
        sys_column = [self.epochs[-1],
                      np.min(self.avail_perfs[-1]),np.max(self.avail_perfs[-1]),
                      np.min(self.avail_mems[-1]),np.max(self.avail_mems[-1]),
                      self.round_times[-1],self.total_times[-1]]
        
        print(f"Round:{self.epochs[-1]}\t|Round Time:{self.round_times[-1]}\t|Total Time:{self.total_times[-1]}")
        with open(self.tsv_file,'a') as af:
            af.write('\t'.join([str(c) for c in sys_column]) + '\n')

        with open(self.pkl_file,'wb') as sys_f:
            pickle.dump([self.epochs,self.chosen_clients,self.chosen_devices,
                        self.runtime_apps,self.avail_perfs,self.avail_mems,
                        self.train_times,self.round_times,self.total_times], sys_f)



    


    
    
    
        


    