import numpy as np
import copy
import torch

class Avg_Server():
    def __init__(self,global_model,clients,selector,scheduler,
                 stat_monitor,sys_monitor,frac=None,
                 weights=None,test_dataset=None,
                 device=torch.device('cpu'),test_every=1,**kwargs):
        """
        Avg_Server: Average all parameters among updated models
        clients: list of clients
        selector: client selector
        scheduler: generate training hyperparameters for each client
        monitor: return the statistical and systematic information of each device
        """
        self.global_model = global_model
        self.clients = clients
        self.num_users = len(self.clients)
        self.train_frac = frac
        if weights is not None:
            self.weights = weights
        else:
            self.weights = np.ones(self.num_users)/self.num_users
        self.test_dataset = test_dataset

        self.selector = selector
        self.scheduler = scheduler
        self.stat_monitor = stat_monitor
        self.sys_monitor = sys_monitor

        self.device = device
        self.global_model.to(self.device)
        
        self.round = 0
        self.idxs_users = None
        self.test_every = test_every

        # collect the init loss and training latency
        monitor_params = self.scheduler.monitor_params()
        self.stat_info = self.stat_monitor.collect(self.global_model,epoch=0,**monitor_params)
        self.sys_info = self.sys_monitor.collect(epoch=0)




    def train(self):
        """conduct training of one communication round"""
        
        print('\n | Global Training Round : {} |\n'.format(self.round))
        
        
        
        # update the selector's information before selection
        self.selector.stat_update(epoch=self.round,
                                  selected_clients=self.idxs_users,
                                  stat_info=self.stat_info,
                                  sys_info=self.sys_info,
                                  server=self)
        
        #update the scheduler's information
        self.scheduler.stat_update(epoch =self.round,
                                   stat_info = self.stat_info,
                                   sys_info = self.sys_info,
                                   global_model = self.global_model)
        
        # select clients
        if self.train_frac is None:
            self.idxs_users = np.arange(len(self.clients))
        else:
            m = max(int(self.train_frac * self.num_users), 1)
            self.idxs_users = self.selector.select(m)
        print("Chosen Clients:",self.idxs_users)
        
        
        # train selected clients
        self.global_model = self.train_idx(self.idxs_users)

        if (self.round+1)%self.test_every == 0:
            # collect each client's statistical information
            monitor_params = self.scheduler.monitor_params()
            self.stat_info = self.stat_monitor.collect(self.global_model,
                                                       epoch=self.round,
                                                       chosen_idxs=self.idxs_users,
                                                       test_dataset=self.test_dataset,
                                                       device=self.device,log=True,
                                                       **monitor_params)
        # collect each client's systematic information
        self.sys_info = self.sys_monitor.collect(epoch=self.round,
                                                 chosen_idxs=self.idxs_users,
                                                 log=True)

        self.round += 1
        return self.global_model

    def train_idx(self,idxs_users):
        local_weights = np.array([None for _ in range(self.num_users)])
        for idx in idxs_users:
            training_hyperparameters = self.scheduler.training_params(idx=idx)
            local_model = self.clients[idx].train(self.global_model,**training_hyperparameters)
            local_model.to(self.device) # return the local model to the server's device
            local_weights[idx] = copy.deepcopy(local_model.state_dict())
            
        global_model = self.aggregate(local_weights)
        
        
        return global_model
    
    def aggregate(self,weights):
        """
        weighted average of all updated weights
        The average is conducted in a parameter-by-parameter fashion, 
        and the weights are normalized respectively.
        """
        model = copy.deepcopy(self.global_model)
        w0 = model.state_dict()
        for p in w0.keys():
            weights_sum = 0.0
            w = 0
            for n,lw in enumerate(weights):
                if lw is not None and p in lw:
                    weights_sum += self.weights[n]
                    w += self.weights[n]*lw[p]
            if weights_sum > 0:
                w0[p] = w/weights_sum
        model.load_state_dict(w0)
        return model
