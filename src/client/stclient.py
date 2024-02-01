import torch
from torch.utils.data import DataLoader, Subset
import torch.nn
import copy
import numpy as np
from hardware.sys_utils import sample_runtime_app

class ST_Client():
    """
    This is the standard fl client, who uses standard local SGD for training
    Typical use case: FedAvg, FedBN
    """
    def __init__(self,dataset,data_idxs,sys_info=None,
                 local_state_preserve=False,
                 device = torch.device('cpu'), 
                 verbose=False, random_seed=None, **kwargs):
        self.trainset, self.testset = self.train_test(dataset, list(data_idxs))
        self.dev_name,self.performance,self.memory=sys_info
        self.runtime_app,self.perf_degrade,self.mem_degrade = None,None,None
        self.batches = None
        self.device = device
        self.final_local_accuracy,self.final_local_loss = 100.0,100.0
        self.local_state_preserve = local_state_preserve
        self.local_states = None
        self.verbose = verbose
        self.rs = np.random.RandomState(random_seed)


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
              criterion=torch.nn.CrossEntropyLoss(),**kwargs):
        """train the model for one communication round."""
        model = copy.deepcopy(model) # avoid modifying global model
        model.to(self.device)
        if self.local_state_preserve and self.local_states is not None:
            model = self.load_local_state_dict(model,self.local_states)
        model.train()
        
        self.trainloader = DataLoader(self.trainset,batch_size=local_bs,shuffle=True)

        # Set optimizer for the local updates
        np = model.parameters()
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(np, lr=lr, momentum=momentum,weight_decay=reg)
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(np, lr=lr, weight_decay=reg)
        

        iters = 0
        self.batches = []
        while iters < local_ep:
        #for iter in range(self.args.local_ep):
            for _, (datas, labels) in enumerate(self.trainloader):
                self.batches.append(list(datas.shape))
                datas, labels = datas.to(self.device), labels.to(self.device)
                model.zero_grad()
                output = model(datas)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                iters += 1
                if iters == local_ep:
                    break

            if self.verbose:
                print('Local Epoch : {}/{} |\tLoss: {:.4f}'.format(iters, local_ep, loss.item()))
        
        self.local_states = copy.deepcopy(self.get_local_state_dict(model))
        self.final_local_accuracy,self.final_local_loss = self.validate(model)
        
        return model

    def validate(self,model,criterion=torch.nn.CrossEntropyLoss()):
        """ Returns the validation accuracy and loss."""
        model = copy.deepcopy(model) # avoid modifying global model
        model.to(self.device)
        if self.local_state_preserve and self.local_states is not None:
            model = self.load_local_state_dict(model,self.local_states)
        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        self.testloader = DataLoader(self.testset,batch_size=32, shuffle=False)

        for batch_idx, (datas, labels) in enumerate(self.testloader):
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
        self.avail_perf = self.performance*self.perf_degrade
        self.avail_mem = self.memory*self.mem_degrade
        return self.runtime_app, self.avail_perf,self.avail_mem

    
    def get_local_state_dict(self,model):
        sd = model.state_dict()
        for name in list(sd.keys()):
            # pop out the parameters except for bn layers
            if 'weight' in name and name.replace('weight','running_mean') not in sd.keys():
                sd.pop(name)
            elif 'bias' in name and name.replace('bias','running_mean') not in sd.keys():
                sd.pop(name)
        return sd
    
    def load_local_state_dict(self,model,local_dict):
        sd = model.state_dict()
        sd.update(local_dict)
        model.load_state_dict(sd)
        return model
    