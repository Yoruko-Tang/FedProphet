import numpy as np
import copy
import torch
import json

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
        self.device = device
        global_model.to(device)
        self.global_model = {"model":global_model}

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

        
        self.round = 0
        self.idxs_users = None
        self.test_every = test_every

        # collect the init loss and training latency
        # self.stat_info = self.val(self.global_model)
        # self.sys_info = self.sys_monitor.collect()




    def train(self):
        """conduct training of one communication round"""
        
        # print('\n | Global Training Round : {} |\n'.format(self.round))
        
        
        
        # update the selector's information before selection
        self.selector.stat_update(epoch=self.round,
                                  selected_clients=self.idxs_users,
                                  stat_info=self.stat_info,
                                  sys_info=self.sys_info,
                                  server=self)
        
        #update the scheduler's information
        CTN = self.scheduler.stat_update(epoch = self.round,
                                   stat_info = self.stat_info,
                                   sys_info = self.sys_info,
                                   global_model = self.global_model)
        
        if CTN:
            # select clients
            if self.train_frac is None:
                self.idxs_users = np.arange(len(self.clients))
            else:
                m = max(int(self.train_frac * self.num_users), 1)
                self.idxs_users = self.selector.select(m)
            print("Chosen Clients:",self.idxs_users)
            
            
            # train selected clients
            self.global_model = self.train_idx(self.idxs_users)
            
            # collect statistical and systematic information
            self.monitor(chosen_idxs=self.idxs_users,log=True)

            self.round += 1
        return CTN

    def train_idx(self,idxs_users):
        local_models = np.array([None for _ in range(self.num_users)])
        local_training_params = np.array([None for _ in range(self.num_users)])
        for idx in idxs_users:
            training_hyperparameters = self.scheduler.training_params(idx=idx,chosen_idxs=idxs_users)
            local_model = self.clients[idx].train(**self.global_model,**training_hyperparameters)
            local_models[idx] = local_model
            local_training_params[idx] = training_hyperparameters

        global_model = self.aggregate(local_models = local_models,
                                      training_hyperparameters = local_training_params,
                                      **self.global_model)
        
        
        return global_model
    
    def aggregate(self,local_models,model,training_hyperparameters=None,**kwargs):
        """
        weighted average of all updated weights
        The average is conducted in a parameter-by-parameter fashion, 
        and the weights are normalized respectively.
        """
        model = copy.deepcopy(model)
        w0 = model.state_dict()
        for p in w0.keys():
            weights_sum = 0.0
            w = 0
            for n,lm in enumerate(local_models):
                if lm is not None:
                    if isinstance(lm,dict):
                        local_model = lm["model"]
                    else:
                        local_model = lm
                    local_model.to(self.device)
                    lw = local_model.state_dict()
                    if p in lw:
                        weights_sum += self.weights[n]
                        w += self.weights[n]*lw[p]
            if weights_sum > 0:
                w0[p] = w/weights_sum
        model.load_state_dict(w0)
        return {"model":model}

    def val(self,model,chosen_idxs = None, log=False):
        monitor_params = self.scheduler.monitor_params()

        return self.stat_monitor.collect(model,epoch=self.round,
                                        chosen_idxs=chosen_idxs,
                                        test_dataset=self.test_dataset,
                                        log=log,**monitor_params)
    
    def monitor(self,chosen_idxs = None,log=False):
        if self.round%self.test_every == 0:
            self.stat_info = self.val(self.global_model,
                                    chosen_idxs=chosen_idxs,
                                    log=log)
        self.sys_info = self.sys_monitor.collect(epoch=self.round,
                                                 chosen_idxs=chosen_idxs,
                                                 log=log)
        return self.stat_info,self.sys_info
    
    def load(self,save_file):
        global_model,local_states = torch.load(save_file)
        # some customized methods may not be loaded with torch load. load the state dict indirectly
        self.global_model["model"].load_state_dict(global_model["model"].state_dict())
        # for other models, directly load
        global_model.pop("model")
        self.global_model.update(global_model)
        for idx,client in enumerate(self.clients):
            client.local_states = local_states[idx]
        with open(save_file.replace("best_model.pt","modelinfo.json"),'r') as info_file:
            model_info = json.load(info_file)
        self.round = model_info["round"]
        self.scheduler.round = model_info["round"]