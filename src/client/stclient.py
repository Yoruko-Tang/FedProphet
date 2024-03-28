import torch
from torch.utils.data import DataLoader, Subset
import torch.nn
import copy
import numpy as np
from hardware.sys_utils import sample_runtime_app,sample_networks,model_summary

class ST_Client():
    """
    This is the standard fl client, who uses standard local SGD for training
    Typical use case: FedAvg, FedBN
    """
    def __init__(self,dataset,data_idxs,sys_info=None,
                 model_profile:model_summary = None,
                 init_local_state = None,
                 local_state_preserve = False,
                 device = torch.device('cpu'), 
                 verbose=False, random_seed=None, 
                 reserved_performance = 0, reserved_memory = 0, 
                 **kwargs):
        self.trainset, self.testset = self.train_test(dataset, list(data_idxs))
        self.dev_name,self.peak_performance,self.peak_memory,self.eff_bw=sys_info
        self.model_profile = model_profile

        self.local_state_preserve = local_state_preserve
        if local_state_preserve:
            self.local_states = copy.deepcopy(init_local_state)
        else:
            self.local_states = None
        
        self.device = device
        self.final_local_loss = None
        
        self.verbose = verbose
        self.rs = np.random.RandomState(random_seed)

        self.reserved_performance = reserved_performance
        self.reserved_memory = reserved_memory
        
        # runtime statics
        self.batches = None
        self.runtime_app = None
        self.avail_perf = None 
        self.avail_mem = None 
        self.network = None
        self.network_speed = None
        self.network_lag = None 
        self.latency = None 
        self.est_latency = None
        
        
        
        

    def train_test(self, dataset, idxs):
        """
        Returns train and test dataloaders for a given dataset and user indexes.
        """
        # split indexes for train, and test (80, 20)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_test = idxs[int(0.8*len(idxs)):]

        trainset = Subset(dataset, idxs_train)
        testset = Subset(dataset, idxs_test)
                                
        return trainset, testset
    
    def train(self,model,local_ep,local_bs,lr,optimizer='sgd',
              momentum=0.0,reg=0.0,
              criterion=torch.nn.CrossEntropyLoss(),
              model_profile=None,
              **kwargs):
        """train the model for one communication round."""
        def standard_loss(model,input,label):
            output = model(input)
            task_loss = criterion(output,label)
            return task_loss
        

        model = copy.deepcopy(model) # avoid modifying global model
        model.to(self.device)
        if self.local_state_preserve and self.local_states is not None:
            model = self.load_local_state_dict(model,self.local_states)
        

        trainloader = DataLoader(self.trainset,batch_size=local_bs,shuffle=True)

        # Set optimizer for the local updates
        np = model.parameters()
        if optimizer == 'sgd':
            opt = torch.optim.SGD(np, lr=lr, momentum=momentum,weight_decay=reg)
        elif optimizer == 'adam':
            opt = torch.optim.Adam(np, lr=lr, weight_decay=reg)
        

        iters = 0
        self.batches = []
        while iters < local_ep:
        #for iter in range(self.args.local_ep):
            for _, (datas, labels) in enumerate(trainloader):
                self.batches.append(list(datas.shape))
                datas, labels = datas.to(self.device), labels.to(self.device)
                model.train()
                model.zero_grad()
                loss = standard_loss(model,datas,labels)
                loss.backward()
                opt.step()
                iters += 1
                if iters == local_ep:
                    break

            if self.verbose:
                print('Local Epoch : {}/{} |\tLoss: {:.4f}'.format(iters, local_ep, loss.item()))
        
        if self.local_state_preserve:
            self.local_states = copy.deepcopy(self.get_local_state_dict(model))
        self.final_local_loss = loss.item()

        # calculate training latency
        if model_profile is not None:
            self.model_profile = model_profile
        
        self.latency = self.model_profile.training_latency(performance=self.avail_perf,
                                                           memory=self.avail_mem,
                                                           eff_bandwidth=self.eff_bw,
                                                           network_bandwidth=self.network_speed,
                                                           network_latency=self.network_lag,
                                                           **self.__dict__)

        return model

    def validate(self,model,testset=None,criterion=torch.nn.CrossEntropyLoss(),
                 load_local_state = True,**kwargs):
        """ Returns the validation accuracy and loss."""
        model = copy.deepcopy(model) # avoid modifying global model
        model.to(self.device)
        if load_local_state and self.local_states is not None:
            model = self.load_local_state_dict(model,self.local_states)
        
        model.eval()

        loss, total, correct = 0.0, 0.0, 0.0
        if testset is None:
            testset = self.testset
        testloader = DataLoader(testset,batch_size=128, shuffle=False)

        for batch_idx, (datas, labels) in enumerate(testloader):
            datas, labels = datas.to(self.device), labels.to(self.device)
            outputs = model(datas)
            batch_loss = criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss/(batch_idx+1)
    
    
    def get_runtime_sys_stat(self):
        self.runtime_app,self.perf_degrade,self.mem_degrade=sample_runtime_app(self.rs)
        self.avail_perf = max([self.peak_performance*self.perf_degrade,self.reserved_performance])
        self.avail_mem = max([self.peak_memory*self.mem_degrade,self.reserved_memory])

        self.network,self.network_speed,self.network_lag = sample_networks(self.rs)
        
        if self.model_profile is not None:
            self.est_latency = \
                self.model_profile.training_latency(performance=self.avail_perf,
                                                    memory=self.avail_mem,
                                                    eff_bandwidth=self.eff_bw,
                                                    network_bandwidth=self.network_speed,
                                                    network_latency=self.network_lag,
                                                    **self.__dict__)
        else:
            self.est_latency = None
        # return the current availale performance, memory, and the training latency of the last round
        return self.runtime_app, self.avail_perf, self.avail_mem, self.network,self.network_speed,self.network_lag, self.latency, self.est_latency

    @staticmethod
    def get_local_state_dict(model):
        sd = model.state_dict()
        for name in list(sd.keys()):
            # pop out the parameters except for bn layers
            if 'weight' in name and name.replace('weight','running_mean') not in sd.keys():
                sd.pop(name)
            elif 'bias' in name and name.replace('bias','running_mean') not in sd.keys():
                sd.pop(name)
        return sd
    
    @staticmethod
    def load_local_state_dict(model,local_dict):
        sd = model.state_dict()
        sd.update(local_dict)
        model.load_state_dict(sd)
        return model
    