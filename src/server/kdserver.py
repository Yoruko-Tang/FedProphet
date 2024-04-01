import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset,StackDataset
import numpy as np
import copy
from server.avgserver import Avg_Server
from client import ST_Client

class FedDF_Server(Avg_Server):
    def __init__(self, global_model, edge_models, clients, selector, scheduler, 
                 stat_monitor, sys_monitor, frac=None, weights=None, 
                 test_dataset=None, device=torch.device('cpu'), test_every=1, 
                 public_dataset=None, dist_iters=128, dist_lr=5e-3, dist_batch_size=64,
                **kwargs):
        super().__init__(global_model, clients, selector, scheduler, 
                        stat_monitor, sys_monitor, frac, weights, 
                        test_dataset, device, test_every)
        
        for m in edge_models:
            m.to(device)
        self.global_model["edge_models"] = edge_models
        self.edge_model_sds = [ST_Client.get_local_state_dict(m) for m in edge_models]

        self.public_dataset = public_dataset
        self.tau_s = dist_iters
        self.lr_s = dist_lr
        self.b_s = dist_batch_size

    def aggregate(self,local_models,edge_models,training_hyperparameters=None,**kwargs):
        
        # generate KD dataset
        transfer_loader = DataLoader(self.public_dataset,batch_size=self.b_s,shuffle=False)

        # Step 1: generate the logits of each group of model
        edge_local_model = [[None]*self.num_users for _ in range(len(edge_models))]
        logits = []
        with torch.no_grad():
            for n,local_model in enumerate(local_models): # calculate logits from edge devices
                if local_model is not None:
                    group = training_hyperparameters[n]["model_idx"]
                    

                    edge_local_model[group][n] = copy.deepcopy(local_model)
                    
                    local_model.to(self.device)
                    client_preds = []
                    
                    for data,_ in transfer_loader: # ignore the original labels
                        data = data.to(self.device)
                        pred = local_model(data).cpu()
                        client_preds.append(pred)

                    logits.append(torch.cat(client_preds))
            avg_logits = 0
            for l in logits:
                avg_logits += l
            avg_logits /= len(logits)
            avg_logits = F.log_softmax(avg_logits,dim=1)

            label_dataset = TensorDataset(avg_logits)
            KD_dataset = StackDataset(self.public_dataset,label_dataset)
            KD_loader = DataLoader(KD_dataset,batch_size=self.b_s,shuffle=True)

                
        # Step 2: Train the server model with Ensemble loss
        new_edge_models = []
        for group in range(len(edge_local_model)):
            
            group_model = super().aggregate(edge_local_model[group],edge_models[group])["model"]
            group_model = ST_Client.load_local_state_dict(group_model,self.edge_model_sds[group])
            
            group_model.to(self.device)
        
            optimizer = torch.optim.SGD(group_model.parameters(),
                                        lr=self.lr_s,
                                        momentum=0.9)

            group_model.train()
            iters = 0
            print("=============Training Edge Model #%d================="%group)
            while iters < self.tau_s:
                for (samples,_),(labels,) in KD_loader:
                    samples,labels = samples.to(self.device),labels.to(self.device)
                    group_model.zero_grad()
                    output = group_model(samples)
                    loss = F.kl_div(F.log_softmax(output,dim=1),
                                    labels,reduction='batchmean',
                                    log_target=True)
                    loss.backward()
                    optimizer.step()
                    iters+=1
                    if iters == self.tau_s:
                        break
                    if iters%10 == 0:
                        print("Server Iters : {}/{}|\tLoss : {:.4f}".format(iters,self.tau_s,loss.item()))
            new_edge_models.append(group_model)
            

        
        
        return {"model":new_edge_models[0],"edge_models":new_edge_models}


class FedET_Server(FedDF_Server):
    """
    This is the server that uses knowledge distillation for aggregation, e.g., FedET and FedDF
    """
    def __init__(self, global_model, edge_models, clients, selector, scheduler, 
                 stat_monitor, sys_monitor, frac=None, weights=None, test_dataset=None, 
                 device=torch.device('cpu'), test_every=1, 
                 public_dataset=None, dist_iters=128, dist_lr=5e-3, dist_batch_size=64,
                 diver_lamb=0.05, **kwargs):
        super().__init__(global_model, edge_models, clients, selector, scheduler, 
                        stat_monitor, sys_monitor, frac, weights, 
                        test_dataset,device, test_every,public_dataset, 
                        dist_iters, dist_lr, dist_batch_size)
        
        self.lamb_s = diver_lamb

        

    def aggregate(self,local_models,model,edge_models,training_hyperparameters=None,**kwargs):
        
        # generate KD dataset
        transfer_loader = DataLoader(self.public_dataset,batch_size=self.b_s,shuffle=False)
        
        preds = []
        pred_variance = []

        
        edge_model_sd = [[] for _ in range(len(edge_models))]
        rep_weight = 0.0 
        rep_bias = 0.0
        weight_count = 0
        with torch.no_grad():
            for n,local_model in enumerate(local_models): # calculate logits from edge devices
                if local_model is not None:
                    group = training_hyperparameters[n]["model_idx"]
                    w = local_model.state_dict()
                    edge_model_sd[group].append(w)
                    rep_layers = local_model.rep_layers
                    rep_weight += w[rep_layers[0]]
                    rep_bias += w[rep_layers[1]]
                    weight_count += 1
                    local_model.to(self.device)
                    local_model.eval()

                    client_preds = []
                    client_pred_variance = []
                    for data,_ in transfer_loader: # ignore the original labels
                        data = data.to(self.device)
                        pred = local_model(data).cpu()
                        variance = torch.var(pred,dim=1)
                        client_preds.append(pred)
                        client_pred_variance.append(variance)
                    preds.append(torch.cat(client_preds))
                    pred_variance.append(torch.cat(client_pred_variance))
                    # local_model.to('cpu') # release the gpu memory
            # Step 1: initialize the last layer of the server model as the average of edge models
            rep_weight /= weight_count
            rep_bias /= weight_count
            server_model = copy.deepcopy(model)
            server_model_sd = server_model.state_dict()
            server_model_sd[server_model.rep_layers[0]]=rep_weight
            server_model_sd[server_model.rep_layers[1]]=rep_bias
            server_model.load_state_dict(server_model_sd)
            # generate consensus label
            pred_variance = torch.vstack(pred_variance)# CxB
            pred_weights = pred_variance/(torch.sum(pred_variance,dim=0,keepdim=True)+1e-6)
            logit = 0
            for i in range(len(preds)):
                logit += pred_weights[i].reshape([-1,1])*preds[i] # Bxd
            label = torch.argmax(logit,dim=1) # B
            

            # generate diversity label
            div_mask  = []
            for p in preds:
                l = torch.argmax(p,dim=1)
                div_mask.append(1-torch.eq(l,label).int()) # B
            div_mask = torch.vstack(div_mask)# CxB
            div_variance = div_mask*pred_variance # only reserve the variance with different labels
            div_weights = div_variance/(torch.sum(div_variance,dim=0,keepdim=True)+1e-6)
            div_logit = 0
            for i in range(len(preds)):
                div_logit += div_weights[i].reshape([-1,1])*preds[i] # Bxd


        label_dataset = TensorDataset(label,div_logit)
        KD_dataset = StackDataset(self.public_dataset,label_dataset)


        # Step 2: Train the server model with Ensemble loss
        KD_loader = DataLoader(KD_dataset,batch_size=self.b_s,shuffle=True)

        server_model.to(self.device)
        
        optimizer = torch.optim.SGD(server_model.parameters(),
                                    lr=self.lr_s,
                                    momentum=0.9)

        server_model.train()
        iters = 0
        print("=============Training Server Model=================")
        while iters < self.tau_s:
            for (samples,_),(labels,div_logits) in KD_loader:
                samples,labels,div_logits = samples.to(self.device),labels.to(self.device),div_logits.to(self.device)
                server_model.zero_grad()
                output = server_model(samples)
                ce_loss = F.cross_entropy(output,labels)
                kl_loss = F.kl_div(F.log_softmax(output,dim=1),
                                   F.log_softmax(div_logits,dim=1),
                                   reduction='batchmean',log_target=True)
                loss = ce_loss + self.lamb_s*kl_loss
                loss.backward()
                optimizer.step()
                iters+=1
                if iters == self.tau_s:
                    break
                if iters%10 == 0:
                    print("Server Iters : {}/{}|\tLoss : {:.4f}|\tCE Loss: {:.4f}|\tKL Loss: {:.4f}".format(iters,self.tau_s,loss.item(),ce_loss.item(),kl_loss.item()))

        # Step 3: Update Edge Models
        new_edge_models = []
        for g,group_weights in enumerate(edge_model_sd):
            if g == 0: # use the server model as the largest model directly
                new_edge_models.append(server_model)
            else:
                edge_model = copy.deepcopy(edge_models[g])
                w0 = edge_model.state_dict()
                if len(group_weights)>0:
                    for p in w0:
                        w = 0
                        for ew in group_weights:
                            w += ew[p]
                        
                        w0[p] = w/len(group_weights)
                w0[edge_model.rep_layers[0]] = copy.deepcopy(server_model.state_dict()[server_model.rep_layers[0]])
                w0[edge_model.rep_layers[1]] = copy.deepcopy(server_model.state_dict()[server_model.rep_layers[1]])
                edge_model.load_state_dict(w0)
                new_edge_models.append(edge_model)
        
        return {"model":server_model,"edge_models":new_edge_models}


            



        