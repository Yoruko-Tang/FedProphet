import numpy as np
import os.path as osp
import os
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
from utils.adversarial import Adv_Sample_Generator
import json

class ST_Stat_Monitor():
    """
    collect the training,validation and test accuracy and loss
    """
    def __init__(self,clients,weights = None,log_path=None,
                 criterion=F.cross_entropy):
        self.clients = clients
        self.weights = weights if weights is not None else np.ones(len(self.clients))/len(self.clients)
        self.chosen_clients = []
        # training
        self.local_losses = []
        self.weighted_local_losses = []

        # validation
        self.global_accs,self.global_losses = [],[]
        self.weighted_global_accs,self.weighted_global_losses = [],[]
        
        # test
        self.test_accs,self.test_losses = [],[]

        self.epochs = []
        
        self.criterion = criterion

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

    def collect(self,global_model,epoch=None,chosen_idxs=None,test_dataset=None,device=torch.device('cpu'),log=False,save=True,**validation_kwargs):
        local_losses = []
        global_accs, global_losses = [],[]
        
        # training acc and loss
        if chosen_idxs is not None:
            for idx in chosen_idxs:
                local_losses.append(self.clients[idx].final_local_loss)
        else: 
            local_losses = None
        
        # validation acc and loss
        for n,c in enumerate(self.clients):
            if isinstance(global_model,list):
                acc,loss = c.validate(global_model[n],**validation_kwargs)
            else:
                acc,loss = c.validate(global_model,**validation_kwargs)
            global_accs.append(acc)
            global_losses.append(loss)
        
        # test acc and loss
        if test_dataset is not None:
            
            test_acc,test_loss = self.test_inference(global_model,test_dataset,device)
            # else:
            #     test_accs,test_losses = [],[]
            #     for n in range(len(global_model)):
            #         test_acc_n,test_loss_n = self.test_inference(global_model[n],test_dataset,device)
            #         test_accs.append(test_acc_n)
            #         test_losses.append(test_loss_n)
            #     test_acc,test_loss = np.mean(test_accs),np.mean(test_losses)
        
        else:
            test_acc,test_loss = 0,None

        # collect
        if local_losses is not None:
            local_losses = np.array(local_losses)
            weighted_local_loss = np.sum(local_losses*self.weights[chosen_idxs])/np.sum(self.weights[chosen_idxs])
        else:
            weighted_local_loss = None
    
        

        global_accs, global_losses  = np.array(global_accs),np.array(global_losses)
        weighted_global_acc,weighted_global_loss = np.sum(global_accs*self.weights),np.sum(global_losses*self.weights)
        
        
        if log:
            self.epochs.append(epoch)
            self.chosen_clients.append(chosen_idxs)

            self.local_losses.append(local_losses)
            self.global_accs.append(global_accs)
            self.global_losses.append(global_losses)
            self.test_accs.append(test_acc)
            self.test_losses.append(test_loss)

            self.weighted_local_losses.append(weighted_local_loss)
            self.weighted_global_accs.append(weighted_global_acc)
            self.weighted_global_losses.append(weighted_global_loss)

            print(f"Round: {epoch}\t|Validation Accuracy: {weighted_global_acc*100:.2f}%\t|Test Accuracy: {test_acc*100:.2f}%")
            # log the latest result into the log files
            if save:
                train_column = [epoch,'train',weighted_local_loss,'n/a','n/a']
                val_column = [epoch,'val',weighted_global_loss,weighted_global_acc,
                            max(self.weighted_global_accs)]
                test_column = [epoch,'test',test_loss,test_acc,max(self.test_accs)]
                with open(self.tsv_file, 'a') as af:
                    af.write('\t'.join([str(c) for c in train_column]) + '\n')
                    af.write('\t'.join([str(c) for c in val_column]) + '\n')
                    if test_dataset is not None:
                        af.write('\t'.join([str(c) for c in test_column]) + '\n')
                
                with open(self.pkl_file,'wb') as stat_f:
                    pickle.dump([self.weights,self.epochs,self.chosen_clients, 
                                self.local_losses,self.weighted_local_losses,
                                self.global_accs,self.global_losses,
                                self.weighted_global_accs,self.weighted_global_losses,
                                self.test_accs,self.test_losses], stat_f)
                # store the model if it attains the highest validation loss
                if np.argmax(self.weighted_global_accs) == len(self.weighted_global_accs)-1:
                    torch.save([global_model,[c.local_states for c in self.clients]],self.pt_file)
                    model_info = {"round":epoch,
                                  "validation clean accuracy":weighted_global_acc,
                                  "test clean accuracy":test_acc}
                    with open(self.pt_file.replace('best_model.pt','modelinfo.json'),'w') as pf:
                        json.dump(model_info,pf,indent=True)
        
        res = {"epoch":epoch,
            "train_loss": local_losses,
            "val_acc": global_accs,
            "val_loss": global_losses,
            "test_acc": test_acc,
            "test_loss": test_loss}
        return res
    
      
            
    def test_inference(self, model, test_dataset,device = torch.device('cpu')):
        """ Returns the test accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        model.to(device)
        
        testloader = DataLoader(test_dataset, batch_size=128,
                                shuffle=False)

        for batch_idx, (datas, labels) in enumerate(testloader):
            datas, labels = datas.to(device), labels.to(device)

            # Inference
            outputs = model(datas)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss/(batch_idx+1)

    # def validation(self,model):
    #     val_accs = []
    #     val_losses = []
    #     for n,c in enumerate(self.clients):
    #         if isinstance(model,list):
    #             acc,loss = c.validate(model[n])
    #         else:
    #             acc,loss = c.validate(model)
    #         val_accs.append(acc)
    #         val_losses.append(loss)
    #     return val_accs, val_losses
        
class AT_Stat_Monitor(ST_Stat_Monitor):
    """
    collect the training,validation and test accuracy and loss
    """
    def __init__(self,clients,weights = None,log_path=None,
                 criterion = F.cross_entropy,
                 test_adv_method='PGD',test_adv_eps=8/255,
                 test_adv_alpha=2/255,test_adv_T=10,test_adv_norm='inf',
                 test_adv_bound=[0.0,1.0]):
        super().__init__(clients,weights,log_path,criterion)
        self.test_adv_criterion = lambda m,i,y:self.criterion(m(i),y)
        self.test_adv_method=test_adv_method
        self.test_adv_eps = test_adv_eps
        self.test_adv_alpha = test_adv_alpha
        self.test_adv_T = test_adv_T
        self.test_adv_norm = test_adv_norm
        self.test_adv_bound = test_adv_bound
        self.adv_sample_gen = Adv_Sample_Generator(criterion=self.test_adv_criterion,
                                                   attack_method=self.test_adv_method,
                                                   epsilon=self.test_adv_eps,
                                                   alpha=self.test_adv_alpha,
                                                   T=self.test_adv_T,
                                                   norm=self.test_adv_norm,
                                                   bound=self.test_adv_bound)
        # validation
        self.global_adv_accs,self.global_adv_losses = [],[]
        self.weighted_global_adv_accs,self.weighted_global_adv_losses = [],[]
        
        # test
        self.test_adv_accs,self.test_adv_losses = [],[]


        # create log files
        with open(self.tsv_file, 'w') as wf:
            columns = ['epoch', 'mode', 'clean_loss', 'clean_accuracy', 'best_clean_accuracy','adv_loss', 'adv_accuracy', 'best_adv_accuracy']
            wf.write('\t'.join(columns) + '\n')

    def collect(self,global_model,adv_test=True,epoch=None,chosen_idxs=None,test_dataset=None,device=torch.device('cpu'),log=False,save=True,**validation_kwargs):
        if not adv_test:
            if log:
                self.global_adv_accs.append(np.zeros(len(self.clients)))
                self.global_adv_losses.append(np.zeros(len(self.clients)))

                self.weighted_global_adv_accs.append(0)
                self.weighted_global_adv_losses.append(0)

                self.test_adv_accs.append(0)
                self.test_adv_losses.append(None)
            return super().collect(global_model,epoch,chosen_idxs,test_dataset,device,log,save)

        res = super().collect(global_model,epoch,chosen_idxs,test_dataset,device,log,save=False)


        
        # val acc and loss
        global_adv_accs, global_adv_losses = [],[]
        for n,c in enumerate(self.clients):
            if isinstance(global_model,list):
                adv_acc,adv_loss = c.adv_validate(global_model[n],**validation_kwargs)
            else:
                adv_acc,adv_loss = c.adv_validate(global_model,**validation_kwargs)
            global_adv_accs.append(adv_acc)
            global_adv_losses.append(adv_loss)
        global_adv_accs = np.array(global_adv_accs)
        global_adv_losses = np.array(global_adv_losses)

        if test_dataset is not None:
            test_adv_acc,test_adv_loss = self.test_adv_inference(global_model,test_dataset,device)
        else:
            test_adv_acc,test_adv_loss = 0,None

        # collect
        weighted_global_adv_acc,weighted_global_adv_loss = np.sum(global_adv_accs*self.weights),np.sum(global_adv_losses*self.weights)

        res["val_adv_acc"] = global_adv_accs
        res["val_adv_loss"] = global_adv_losses
        res["test_adv_acc"] = test_adv_acc
        res["test_adv_loss"] = test_adv_loss

        if log:
            self.global_adv_accs.append(global_adv_accs)
            self.global_adv_losses.append(global_adv_losses)
            
            self.weighted_global_adv_accs.append(weighted_global_adv_acc)
            self.weighted_global_adv_losses.append(weighted_global_adv_loss)

            self.test_adv_accs.append(test_adv_acc)
            self.test_adv_losses.append(test_adv_loss)

            print(f"Round: {epoch}\t|Validation Adversarial Accuracy: {weighted_global_adv_acc*100:.2f}%\t|Test Adversarial Accuracy: {test_adv_acc*100:.2f}%")
            
            if save:
                # log the latest result into the log files
                train_column = [epoch,'train',self.weighted_local_losses[-1],"n/a","n/a","n/a","n/a","n/a"]
                val_column = [epoch,'val',self.weighted_global_losses[-1],self.weighted_global_accs[-1],max(self.weighted_global_accs),weighted_global_adv_loss,weighted_global_adv_acc,max(self.weighted_global_adv_accs)]
                test_column = [epoch,'test',self.test_losses[-1],self.test_accs[-1],max(self.test_accs),test_adv_loss,test_adv_acc,max(self.test_adv_accs)]
                with open(self.tsv_file, 'a') as af:
                    af.write('\t'.join([str(c) for c in train_column]) + '\n')
                    af.write('\t'.join([str(c) for c in val_column]) + '\n')
                    if test_dataset is not None:
                        af.write('\t'.join([str(c) for c in test_column]) + '\n')
                
                with open(self.pkl_file,'wb') as stat_f:
                    pickle.dump([self.weights,self.epochs,self.chosen_clients, 
                                self.local_losses,self.weighted_local_losses,
                                self.global_accs,self.global_losses,
                                self.weighted_global_accs,self.weighted_global_losses,
                                self.global_adv_accs,self.global_adv_losses,
                                self.weighted_global_adv_accs,self.weighted_global_adv_losses,
                                self.test_accs,self.test_losses,self.test_adv_accs,self.test_adv_losses], stat_f)
                # store the model if it attains the highest validation loss
                weighted_global_clean_adv_accs = np.array(self.weighted_global_accs) + np.array(self.weighted_global_adv_accs)
                if np.argmax(weighted_global_clean_adv_accs) == len(weighted_global_clean_adv_accs)-1:
                    torch.save([global_model,[c.local_states for c in self.clients]],self.pt_file)
                    model_info = {"round":epoch,
                                  "validation clean accuracy":self.weighted_global_accs[-1],
                                  "validation adversarial accuracy":weighted_global_adv_acc,
                                  "test clean accuracy":self.test_accs[-1],
                                  "test adversarial accuracy":test_adv_acc}
                    with open(self.pt_file.replace('best_model.pt','modelinfo.json'),'w') as pf:
                        json.dump(model_info,pf,indent=True)
        
        
        return res
    
        
            

    def test_adv_inference(self,model, test_dataset,device = torch.device('cpu')):
        """ Returns the adversarial test accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        model.to(device)
        
        testloader = DataLoader(test_dataset, batch_size=128,
                                shuffle=False)

        for batch_idx, (datas, labels) in enumerate(testloader):
            datas, labels = datas.to(device), labels.to(device)
            adv_datas = self.adv_sample_gen.attack_data(model,datas,labels)
            # Inference
            outputs = model(adv_datas)
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)
            correct += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)

        accuracy = correct/total
        return accuracy, loss/(batch_idx+1)
