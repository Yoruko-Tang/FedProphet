import numpy as np
import os.path as osp
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle


class ST_Stat_Monitor():
    """
    collect the training,validation and test accuracy and loss
    """
    def __init__(self,clients,weights = None,log_path=None):
        self.clients = clients
        self.weights = weights if weights is not None else np.ones(len(self.clients))/len(self.clients)
        self.chosen_clients = []
        # training
        self.local_accs,self.local_losses = [],[]
        self.weighted_local_accs,self.weighted_local_losses = [],[]

        # validation
        self.global_accs,self.global_losses = [],[]
        self.weighted_global_accs,self.weighted_global_losses = [],[]
        
        # test
        self.test_accs,self.test_losses = [],[]

        self.epochs = []
        


        # create log files
        self.log_path = log_path
        if self.log_path is not None:
            if not osp.exists(self.log_path):
                os.makedirs(self.log_path)
            self.tsv_file = osp.join(self.log_path, 'stat.log.tsv')
            self.pkl_file = osp.join(self.log_path, 'stat.pkl')
            self.pt_file = osp.join(self.log_path,'best_model.pt')
            with open(self.tsv_file, 'w') as wf:
                columns = ['epoch', 'mode', 'loss', 'accuracy', 'best_accuracy']
                wf.write('\t'.join(columns) + '\n')

    def collect(self,global_model,epoch=None,chosen_idxs=None,test_dataset=None,device=torch.device('cpu'),log=False):
        local_accs, local_losses = [],[]
        global_accs, global_losses = [],[]
        
        # training acc and loss
        if chosen_idxs is not None:
            for idx in chosen_idxs:
                local_accs.append(self.clients[idx].final_local_accuracy)
                local_losses.append(self.clients[idx].final_local_loss)
        else: # do not collect training data
            local_accs = None
            local_losses = None
        
        # validation acc and loss
        for c in self.clients:
            acc,loss = c.validate(global_model)
            global_accs.append(acc)
            global_losses.append(loss)
        
        # test acc and loss
        if test_dataset is not None:
            test_acc,test_loss = self.test_inference(global_model,test_dataset,device)
        else:
            test_acc,test_loss = None

        # collect
        if local_accs is not None:
            local_accs, local_losses = np.array(local_accs),np.array(local_losses)
        

        global_accs, global_losses  = np.array(global_accs),np.array(global_losses)
        
        
        if log:
            self.epochs.append(epoch)
            self.chosen_clients.append(chosen_idxs)
            self.global_accs.append(global_accs)
            self.global_losses.append(global_losses)
            self.local_accs.append(local_accs)
            self.local_losses.append(local_losses)

            weighted_global_acc,weighted_global_loss = np.sum(global_accs*self.weights),np.sum(global_losses*self.weights)
            self.weighted_global_accs.append(weighted_global_acc)
            self.weighted_global_losses.append(weighted_global_loss)
            
            if local_accs is not None:
                weighted_local_acc,weighted_local_loss = np.sum(local_accs*self.weights[chosen_idxs]),np.sum(local_losses*self.weights[chosen_idxs])
            else:
                weighted_local_acc,weighted_local_loss = None, None
            self.weighted_local_accs.append(weighted_local_acc)
            self.weighted_local_losses.append(weighted_local_loss)
            
            self.test_accs.append(test_acc)
            self.test_losses.append(test_loss)
            self.log()
            # store the model if it attains the highest validation loss
            if np.argmax(self.weighted_global_accs) == len(self.weighted_global_accs)-1:
                torch.save([global_model,[c.local_states for c in self.clients]],self.pt_file)
        
        res = {"epoch":epoch,
            "train_acc": local_accs,
            "train_loss": local_losses,
            "val_acc": global_accs,
            "val_loss": global_losses,
            "test_acc": test_acc,
            "test_loss": test_loss}
        return res
    
    def log(self):
        # log the latest result into the log files
        train_column = [self.epochs[-1],'train',self.weighted_local_accs[-1],self.weighted_local_losses[-1],max(self.weighted_local_accs)]
        val_column = [self.epochs[-1],'val',self.weighted_global_accs[-1],self.weighted_global_losses[-1],max(self.weighted_global_accs)]
        test_column = [self.epochs[-1],'test',self.test_accs[-1],self.test_losses[-1],max(self.test_accs)]
        print(f"Round:{self.epochs[-1]}\t|Val Acc:{self.weighted_global_accs[-1]}\t|Test Acc:{self.test_accs[-1]}")
        with open(self.tsv_file, 'a') as af:
            af.write('\t'.join([str(c) for c in train_column]) + '\n')
            af.write('\t'.join([str(c) for c in val_column]) + '\n')
            af.write('\t'.join([str(c) for c in test_column]) + '\n')
        
        with open(self.pkl_file,'wb') as stat_f:
            pickle.dump([self.weights,self.epochs,self.chosen_clients, 
                        self.local_accs,self.local_losses,self.weighted_local_accs,self.weighted_local_losses,
                        self.global_accs,self.global_losses,self.weighted_global_accs,self.weighted_global_losses,
                        self.test_accs,self.test_losses], stat_f)
        
        
            
    def test_inference(self, model, test_dataset,device = torch.device('cpu')):
        """ Returns the test accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        model.to(device)
        criterion = F.cross_entropy
        testloader = DataLoader(test_dataset, batch_size=128,
                                shuffle=False)

        for batch_idx, (datas, labels) in enumerate(testloader):
            datas, labels = datas.to(device), labels.to(device)

            # Inference
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

        
class AT_Stat_Monitor(ST_Stat_Monitor):
    """
    collect the training,validation and test accuracy and loss
    """
    def __init__(self,clients,weights = None,log_path=None):
        super().__init__(clients,weights,log_path)
        
        # validation
        self.global_adv_accs,self.global_adv_losses = [],[]
        self.weighted_global_adv_accs,self.weighted_global_adv_losses = [],[]
        
        # test
        self.test_adv_accs,self.test_adv_losses = [],[]


        # create log files
        with open(self.tsv_file, 'w') as wf:
            columns = ['epoch', 'mode', 'clean_loss', 'clean_accuracy', 'best_clean_accuracy','adv_loss', 'adv_accuracy', 'best_adv_accuracy']
            wf.write('\t'.join(columns) + '\n')

    def collect(self,global_model,epoch=None,chosen_idxs=None,test_dataset=None,device=torch.device('cpu'),log=False):
        res = super().collect(global_model,epoch,chosen_idxs,test_dataset,device,log=False)
        
        # val acc and loss
        global_clean_accs = res["val_acc"]
        global_clean_losses = res["val_loss"]

        global_adv_accs, global_adv_losses = [],[]
        for c in self.clients:
            adv_acc,adv_loss = c.adv_validate(global_model)
            global_adv_accs.append(adv_acc)
            global_adv_losses.append(adv_loss)
        global_adv_accs = np.array(global_adv_accs)
        global_adv_losses = np.array(global_adv_losses)

        # test acc and loss
        test_clean_acc = res["test_acc"]
        test_clean_loss = res["test_loss"]


        if test_dataset is not None:
            test_adv_acc,test_adv_loss = self.test_adv_inference(global_model,test_dataset,device)
        else:
            test_adv_acc,test_adv_loss = None

        if log:
            self.epochs.append(epoch)
            self.chosen_clients.append(chosen_idxs)
            self.global_accs.append(global_clean_accs)
            self.global_losses.append(global_clean_losses)
            self.global_adv_accs.append(global_adv_accs)
            self.global_adv_losses.append(global_adv_losses)
            self.local_accs.append(res["train_acc"])
            self.local_losses.append(res["train_loss"])
            
            weighted_global_clean_acc,weighted_global_clean_loss = np.sum(global_clean_accs*self.weights),np.sum(global_clean_losses*self.weights)
            self.weighted_global_accs.append(weighted_global_clean_acc)
            self.weighted_global_losses.append(weighted_global_clean_loss)
            
            weighted_global_adv_acc,weighted_global_adv_loss = np.sum(global_adv_accs*self.weights),np.sum(global_adv_losses*self.weights)
            self.weighted_global_adv_accs.append(weighted_global_adv_acc)
            self.weighted_global_adv_losses.append(weighted_global_adv_loss)

            if res["train_acc"] is not None:
                weighted_local_acc,weighted_local_loss = np.sum(res["train_acc"]*self.weights[chosen_idxs]),np.sum(res["train_loss"]*self.weights[chosen_idxs])
            else:
                weighted_local_acc,weighted_local_loss = None, None
            self.weighted_local_accs.append(weighted_local_acc)
            self.weighted_local_losses.append(weighted_local_loss)
            
            self.test_accs.append(test_clean_acc)
            self.test_losses.append(test_clean_loss)
            self.test_adv_accs.append(test_adv_acc)
            self.test_adv_losses.append(test_adv_loss)
            self.log()
        
        adv_res = {"epoch":epoch,
            "train_acc": res["train_acc"],
            "train_loss": res["train_loss"],
            "val_acc": global_clean_accs,
            "val_loss": global_clean_losses,
            "val_adv_acc": global_adv_accs,
            "val_adv_loss": global_adv_losses,
            "test_acc": test_clean_acc,
            "test_loss": test_clean_loss,
            "test_adv_acc": test_adv_acc,
            "test_adv_loss": test_adv_loss}
        return adv_res
    
    def log(self):
        # log the latest result into the log files
        train_column = [self.epochs[-1],'train',self.weighted_local_accs[-1],self.weighted_local_losses[-1],max(self.weighted_local_accs),"n/a","n/a","n/a"]
        val_column = [self.epochs[-1],'val',self.weighted_global_accs[-1],self.weighted_global_losses[-1],max(self.weighted_global_accs),self.weighted_global_adv_accs[-1],self.weighted_global_adv_losses[-1],max(self.weighted_global_adv_accs)]
        test_column = [self.epochs[-1],'test',self.test_accs[-1],self.test_losses[-1],max(self.test_accs),self.test_adv_accs[-1],self.test_adv_losses[-1],max(self.test_adv_accs)]
        print(f"Round:{self.epochs[-1]}\t|Val Clean Acc:{self.weighted_global_accs[-1]}\t|Val Adv Acc:{self.weighted_global_adv_accs[-1]}\t|Test Clean Acc:{self.test_accs[-1]}\t|Test Adv Acc:{self.test_adv_accs[-1]}")
        with open(self.tsv_file, 'a') as af:
            af.write('\t'.join([str(c) for c in train_column]) + '\n')
            af.write('\t'.join([str(c) for c in val_column]) + '\n')
            af.write('\t'.join([str(c) for c in test_column]) + '\n')
        
        with open(self.pkl_file,'wb') as stat_f:
            pickle.dump([self.weights,self.epochs,self.chosen_clients, 
                        self.local_accs,self.local_losses,self.weighted_local_accs,self.weighted_local_losses,
                        self.global_accs,self.global_losses,self.weighted_global_accs,self.weighted_global_losses,
                        self.global_adv_accs,self.global_adv_losses,self.weighted_global_adv_accs,self.weighted_global_adv_losses,
                        self.test_accs,self.test_losses,self.test_adv_accs,self.test_adv_losses], stat_f)
            

    def test_adv_inference(self,model, test_dataset,device = torch.device('cpu')):
        pass