import os.path as osp
import os
import numpy as np
import pickle

class Sys_Monitor():
    """
    collect the validation loss of each client
    """
    def __init__(self,clients,model_profile,log_path=None):
        self.clients=clients
        self.model_profile = model_profile
        self.chosen_clients = []
        self.chosen_devices = []
        self.runtime_apps = []
        self.avail_perfs = []
        self.avail_mems = []
        # training latency
        self.train_times = []
        self.round_times = []
        self.total_times = []

        self.estimate_times = []

        self.epochs = []
        
        # create log file
        self.log_path = log_path
        if self.log_path is not None:
            if not osp.exists(self.log_path):
                os.makedirs(self.log_path)
            self.tsv_file = osp.join(self.log_path, 'sys.log.tsv')
            self.pkl_file = osp.join(self.log_path, 'sys.pkl')
            with open(self.tsv_file, 'w') as wf:
                columns = ['epoch', 'min_avail_perf (GFLOPS)', 'max_avail_perf (GFLOPS)', 
                           'min_avail_mem (GB)', 'max_avail_mem (GB)', 
                           'round_time (s)', 'total_time (s)']
                wf.write('\t'.join(columns) + '\n')
        
    def collect(self,epoch=None,chosen_idxs=None,log=False,save=True,**kwargs):
        runtime_app = [] # running apps of all clients
        avail_perf = [] # available performance of all clients
        avail_mem = []  # available memory of all clients
        train_times = [] # trainning latency of chosen clients in this round
        est_times = [] # estimated training time of the next round
        for n,c in enumerate(self.clients):
            # collect performance, memory and training latency
            cruntime_app,cavail_perf,cavail_mem,clatency,cestlatency = c.get_runtime_sys_stat(self.model_profile)
            
            runtime_app.append(cruntime_app)
            avail_perf.append(cavail_perf)
            avail_mem.append(cavail_mem)

            if chosen_idxs is not None and n in chosen_idxs:
                train_times.append(clatency)

            est_times.append(cestlatency)

        round_time = np.max(train_times) if len(train_times)>0 else 0
        total_time = (self.total_times[-1] if len(self.total_times)>0 else 0)+round_time
        
        
        if log:
            self.epochs.append(epoch)
            self.chosen_clients.append(chosen_idxs)
            if chosen_idxs is not None:
                chosen_dev = [self.clients[idx].dev_name for idx in chosen_idxs]
            else:
                chosen_dev = None
            self.chosen_devices.append(chosen_dev)
            self.runtime_apps.append(runtime_app)
            self.avail_perfs.append(avail_perf)
            self.avail_mems.append(avail_mem)

            self.train_times.append(train_times)
            self.round_times.append(round_time)
            self.total_times.append(total_time)

            self.estimate_times.append(est_times)

            print(f"Round: {epoch}\t|Round Time: {round_time}s\t|Total Time: {total_time}s")

            if save:
                sys_column = [epoch,
                              np.min(avail_perf),np.max(avail_perf),
                              np.min(avail_mem),np.max(avail_mem),
                              round_time,total_time]
            
                
                with open(self.tsv_file,'a') as af:
                    af.write('\t'.join([str(c) for c in sys_column]) + '\n')

                with open(self.pkl_file,'wb') as sys_f:
                    pickle.dump([self.epochs,self.chosen_clients,self.chosen_devices,
                                self.runtime_apps,self.avail_perfs,self.avail_mems,
                                self.train_times,self.round_times,self.total_times,self.estimate_times], sys_f)

        res = {"epoch":epoch,
               "runtime_apps":runtime_app,
               "available_perfs":avail_perf,
               "available_mems":avail_mem,
               "train_times":train_times,
               "round_time":round_time,
               "total_time":total_time,
               "estimate_times":est_times}
        
        return res
    

        



    


    
    
    
        


    