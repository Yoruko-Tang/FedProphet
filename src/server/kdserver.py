import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import copy
from server.avgserver import Avg_Server

class FedET_Server(Avg_Server):
    """
    This is the server that uses knowledge distillation for aggregation, e.g., FedET and FedDF
    """
    def __init__(self, global_model, edge_models, clients, selector, scheduler, 
                 stat_monitor, sys_monitor, frac=None, weights=None, test_dataset=None, 
                 local_state_preserve=False, device=torch.device('cpu'), test_every=1, 
                 public_dataset=None, dist_iters=128, dist_lr=5e-3, dist_batch_size=64,
                 diver_lamb=0.05, **kwargs):
        super().__init__(global_model, clients, selector, scheduler, 
                        stat_monitor, sys_monitor, frac, weights, 
                        test_dataset, local_state_preserve, 
                        device, test_every)
        
        for m in edge_models:
            m.to(device)
        self.global_model["edge_models"] = edge_models

        

        self.public_dataset = public_dataset
        self.tau_s = dist_iters
        self.lr_s = dist_lr
        self.b_s = dist_batch_size
        self.lamb_s = diver_lamb

        
        

        # collect the init loss and training latency
        
        # self.stat_info = self.val(self.global_model)
        # self.sys_info = self.sys_monitor.collect()
        
    def train_idx(self,idxs_users):
        local_models = np.array([None for _ in range(self.num_users)])
        local_training_params = np.array([None for _ in range(self.num_users)])
        for idx in idxs_users:
            training_hyperparameters = self.scheduler.training_params(idx=idx,chosen_idxs=idxs_users)
            group = training_hyperparameters["model_group"]
            local_model = self.clients[idx].train(self.global_model["edge_models"][group],**training_hyperparameters)
            local_models[idx] = local_model
            local_training_params[idx] = training_hyperparameters
            
        global_model = self.aggregate(local_models = local_models,
                                      training_hyperparameters = local_training_params,
                                      **self.global_model)
        
        
        return global_model
        

    def aggregate(self,local_models,model,edge_models,training_hyperparameters=None,**kwargs):
        
        # generate KD dataset
        transfer_loader = DataLoader(self.public_dataset,batch_size=self.b_s,shuffle=False)
        samples = []
        sampled = False
        
        preds = []
        pred_variance = []

        # Step 1: initialize the last layer of the server model as the average of edge models
        edge_model_sd = [[] for _ in range(len(edge_models))]
        rep_weight = 0.0
        rep_bias = 0.0
        weight_count = 0
        for n,local_model in enumerate(local_models): # calculate logits from edge devices
            if local_model is not None:
                group = training_hyperparameters[n]["model_group"]
                local_model.to(self.device)
                w = local_model.state_dict()
                edge_model_sd[group].append(w)
                rep_layers = local_model.rep_layers
                rep_weight += w[rep_layers[0]]
                rep_bias += w[rep_layers[1]]
                weight_count += 1
                local_model.eval()

                client_preds = []
                client_pred_variance = []
                for data,_ in transfer_loader: # ignore the original labels
                    if not sampled:
                        samples.append(data.clone())
                    data = data.to(self.device)
                    pred = F.log_softmax(local_model(data),dim=1).detach()
                    variance = torch.var(pred,dim=1)
                    client_preds.append(pred)
                    client_pred_variance.append(variance)
                samples = torch.cat(samples,dim=0)
                sampled = True
                preds.append(torch.cat(client_preds))
                pred_variance.append(torch.cat(client_pred_variance))
        
        rep_weight /= weight_count
        rep_bias /= weight_count
        server_model = copy.deepcopy(model)
        server_model_sd = server_model.state_dict()
        server_model_sd[server_model.rep_layers[0]]=rep_weight
        server_model_sd[server_model.rep_layers[1]]=rep_bias
        server_model.load_state_dict(server_model_sd)
        # generate consensus label
        pred_variance = torch.vstack(pred_variance)# CxB
        pred_weights = pred_variance/torch.sum(pred_variance,dim=0,keepdim=True)
        logit = 0
        for i in range(len(preds)):
            logit += pred_weights[i]*preds[i]
        labels = torch.argmax(logit,dim=1).detach().cpu()
        

        # generate diversity label
        div_mask  = []
        for p in preds:
            l = torch.argmax(p,dim=1)
            div_mask.append(1-torch.eq(l,labels).int())
        div_mask = torch.vstack(div_mask)# CxB
        div_variance = div_mask*pred_variance
        div_weights = div_variance/torch.sum(div_variance,dim=0,keepdim=True)
        div_logit = 0
        for i in range(len(preds)):
            div_logit += div_weights[i]*preds[i]
        div_logits = div_logit.detach().cpu()
        

        KD_dataset = TensorDataset(samples,labels,div_logits)


        # Step 2: Train the server model with Ensemble loss
        KD_loader = DataLoader(KD_dataset,batch_size=self.b_s,shuffle=True)

        server_model.to(self.device)
        
        optimizer = torch.optim.SGD(server_model.parameters(),
                                    lr=self.lr_s)

        server_model.train()
        iters = 0
        print("=============Training Server Model=================")
        while iters < self.tau_s:
            for samples,labels,div_logits in KD_loader:
                samples,labels,div_logits = samples.to(self.device),labels.to(self.device),div_logits.to(self.device)
                server_model.zero_grad()
                output = server_model(samples)
                ce_loss = F.cross_entropy(output,labels)
                kl_loss = F.kl_div(F.log_softmax(output),F.softmax(div_logits))
                loss = ce_loss + self.lamb_s*kl_loss
                loss.backward()
                optimizer.step()
                iters+=1
                if iters == self.tau_s:
                    break
            print("Server Iters : {}/{}|\tLoss : {:.4f}".format(iters,self.tau_s,loss.item()))

        # Step 3: Update Edge Models
        new_edge_models = []
        for g,group_weights in enumerate(edge_model_sd):
            edge_model = copy.deepcopy(edge_models[g])
            w0 = edge_model.state_dict()
            for p in w0:
                w = 0
                for ew in group_weights:
                    w += ew[p]
                
                w0[p] = w/len(group_weights)
            w0[edge_model.rep_layers[0]] = copy.deepcopy(server_model.state_dict()[server_model.rep_layer[0]])
            w0[edge_model.rep_layers[1]] = copy.deepcopy(server_model.state_dict()[server_model.rep_layer[1]])
            edge_model.load_state_dict(w0)
            new_edge_models.append(edge_model)
        
        return {"model":server_model,"edge_models":new_edge_models}

class FedDF_Server(FedET_Server):
    pass
            



        