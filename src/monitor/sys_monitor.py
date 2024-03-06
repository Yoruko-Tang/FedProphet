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
        self.networks = []
        self.network_speeds = []
        self.network_latencies = []
        # training latency
        self.train_times = []
        self.comp_times = []
        self.mem_times = []
        self.comm_times = []
        self.round_times = []
        self.round_comp_times = []
        self.round_mem_times = []
        self.round_comm_times = []

        self.total_times = []
        self.total_comp_times = []
        self.total_mem_times = []
        self.total_comm_times = []

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
                columns = ['epoch', 'min_avail_perf (FLOPS)', 'max_avail_perf (FLOPS)', 
                           'min_avail_mem (Byte)', 'max_avail_mem (Byte)', 
                           'round_time (s)', 'round_comp_time (s)', 
                           'round_mem_time (s)', 'round_comm_time (s)', 
                           'total_time (s)', 'total_comp_time (s)',
                           'total_mem_time (s)', 'total_comm_time (s)']
                wf.write('\t'.join(columns) + '\n')
        
    def collect(self,epoch=None,chosen_idxs=None,log=False,save=True,**kwargs):
        runtime_app = [] # running apps of all clients
        avail_perf = [] # available performance of all clients
        avail_mem = []  # available memory of all clients
        network = [] # network types of all clients
        network_speed = [] # network bandwidths of all clients
        network_latency = [] # network latencies of all clients
        train_times = [] # trainning latency of chosen clients in this round
        comp_times = [] # computation times of chosen clients in this round
        mem_times = [] # memory access times of chosen clients in this round
        comm_times = [] # communication times of chosen clients in this round
        est_times = [] # estimated training time of the next round
        for n,c in enumerate(self.clients):
            # collect performance, memory and training latency
            cruntime_app,cavail_perf,cavail_mem,\
            cnetwork, cnetwork_speed,cnetwork_latency,\
            clatency,cestlatency = c.get_runtime_sys_stat()
            
            runtime_app.append(cruntime_app)
            avail_perf.append(cavail_perf)
            avail_mem.append(cavail_mem)
            network.append(cnetwork)
            network_speed.append(cnetwork_speed)
            network_latency.append(cnetwork_latency)

            if chosen_idxs is not None and n in chosen_idxs:
                train_times.append(clatency["total"])
                comp_times.append(clatency["computation"])
                mem_times.append(clatency["memory"])
                comm_times.append(clatency["communication"])

            est_times.append(cestlatency["total"])

        round_time = np.max(train_times) if len(train_times)>0 else 0
        round_comp_time = np.max(comp_times) if len(train_times)>0 else 0
        round_mem_time = np.max(mem_times) if len(train_times)>0 else 0
        round_comm_time = np.max(comm_times) if len(train_times)>0 else 0
        
        total_time = (self.total_times[-1] if len(self.total_times)>0 else 0)+round_time
        total_comp_time = (self.total_comp_times[-1] if len(self.total_comp_times)>0 else 0)+round_comp_time
        total_mem_time = (self.total_mem_times[-1] if len(self.total_mem_times)>0 else 0)+round_mem_time
        total_comm_time = (self.total_comm_times[-1] if len(self.total_comm_times)>0 else 0)+round_comm_time

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
            self.networks.append(network)
            self.network_speeds.append(network_speed)
            self.network_latencies.append(network_latency)

            self.train_times.append(train_times)
            self.comp_times.append(comp_times)
            self.mem_times.append(mem_times)
            self.comm_times.append(comm_times)
            self.round_times.append(round_time)
            self.round_comp_times.append(round_comp_time)
            self.round_mem_times.append(round_mem_time)
            self.round_comm_times.append(round_comm_time)

            self.total_times.append(total_time)
            self.total_comp_times.append(total_comp_time)
            self.total_mem_times.append(total_mem_time)
            self.total_comm_times.append(total_comm_time)

            self.estimate_times.append(est_times)

            print(f"Round: {epoch}\t|Round Time: {round_time}s\t|Total Time: {total_time}s")

            if save:
                sys_column = [epoch,
                              np.min(avail_perf),np.max(avail_perf),
                              np.min(avail_mem),np.max(avail_mem),
                              round_time,round_comp_time,round_mem_time,
                              round_comm_time,total_time,total_comp_time,
                              total_mem_time,total_comm_time]
            
                
                with open(self.tsv_file,'a') as af:
                    af.write('\t'.join([str(c) for c in sys_column]) + '\n')

                with open(self.pkl_file,'wb') as sys_f:
                    pickle.dump([self.epochs,self.chosen_clients,self.chosen_devices,
                                self.runtime_apps,self.avail_perfs,self.avail_mems,
                                self.networks,self.network_speeds,self.network_latencies,
                                self.train_times,self.comp_times,self.mem_times,
                                self.comm_times,self.round_times,self.round_comp_times,
                                self.round_mem_times,self.round_comm_times,
                                self.total_times,self.total_comp_times,self.total_mem_times,
                                self.total_comm_times,self.estimate_times], sys_f)

        res = {"epoch":epoch,
               "runtime_apps":runtime_app,
               "available_perfs":avail_perf,
               "available_mems":avail_mem,
               "network_types":network,
               "network_bandwidths":network_speed,
               "network_latency":network_latency,
               "train_times":train_times,
               "round_time":round_time,
               "total_time":total_time,
               "estimate_times":est_times}
        
        return res
    

        



    


    
    
    
        


    