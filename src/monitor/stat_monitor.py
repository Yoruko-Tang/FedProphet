import numpy as np
import os.path as osp
import os
import torch
import pickle
import json

class ST_Stat_Monitor():
    """
    collect the training,validation and test accuracy and loss
    """
    def __init__(self,clients,weights = None,log_path=None):
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

    def collect(self,global_model,epoch=None,chosen_idxs=None,test_dataset=None,log=False,save=True,**validation_kwargs):
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
                # valid_param = dict(validation_kwargs,**global_model[n])
                acc,loss = c.validate(**global_model[n],**validation_kwargs)
            else:
                # valid_param = dict(validation_kwargs,**global_model)
                acc,loss = c.validate(**global_model,**validation_kwargs)
            global_accs.append(acc)
            global_losses.append(loss)
        
        # test acc and loss
        if test_dataset is not None:
            if isinstance(global_model,list):
                test_accs,test_losses = [],[]
                for n in range(len(global_model)):
                    # test_param = dict(validation_kwargs,**global_model[n])
                    test_acc_n,test_loss_n = self.clients[n].validate(testset=test_dataset,
                                                                      **global_model[n],
                                                                      **validation_kwargs)
                    test_accs.append(test_acc_n)
                    test_losses.append(test_loss_n)
                test_acc,test_loss = np.mean(test_accs),np.mean(test_losses)
                
            else:
                # test_param = dict(validation_kwargs,**global_model)
                test_acc,test_loss = self.clients[0].validate(testset=test_dataset,
                                                              **global_model,
                                                              **validation_kwargs)
        
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
                    pickle.dump({"weights":self.weights,
                                "epoch":self.epochs,
                                "chosen_clients":self.chosen_clients, 
                                "train_loss":self.local_losses,
                                "weighted_train_loss":self.weighted_local_losses,
                                "val_acc":self.global_accs,
                                "val_loss":self.global_losses,
                                "weighted_val_acc":self.weighted_global_accs, 
                                "weighted_val_loss":self.weighted_global_losses,
                                "test_acc":self.test_accs,
                                "test_loss":self.test_losses}, stat_f)
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
            "weighted_train_loss": weighted_local_loss,
            "val_acc": global_accs,
            "val_loss": global_losses,
            "weighted_val_acc": weighted_global_acc,
            "weighted_val_loss": weighted_global_loss,
            "test_acc": test_acc,
            "test_loss": test_loss}
        return res
    
        
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

    def collect(self,global_model,adv_test=True,epoch=None,chosen_idxs=None,test_dataset=None,log=False,save=True,**validation_kwargs):
        res = super().collect(global_model,epoch,chosen_idxs,test_dataset,log,save=False,**validation_kwargs)
        if not adv_test:
            global_adv_accs = np.zeros(len(self.clients))
            global_adv_losses = np.array([None]*len(self.clients))
            weighted_global_adv_acc = 0
            weighted_global_adv_loss = None
            test_adv_acc = 0
            test_adv_loss = None
            # if log:
            #     self.global_adv_accs.append(np.zeros(len(self.clients)))
            #     self.global_adv_losses.append(np.zeros(len(self.clients)))

            #     self.weighted_global_adv_accs.append(0)
            #     self.weighted_global_adv_losses.append(0)

            #     self.test_adv_accs.append(0)
            #     self.test_adv_losses.append(None)
            # return super().collect(global_model,epoch,chosen_idxs,test_dataset,log,save,**validation_kwargs)

        


        else:
            # val acc and loss
            global_adv_accs, global_adv_losses = [],[]
            for n,c in enumerate(self.clients):
                if isinstance(global_model,list):
                    # valid_param = dict(validation_kwargs,**global_model[n])
                    adv_acc,adv_loss = c.adv_validate(**global_model[n],**validation_kwargs)
                else:
                    # valid_param = dict(validation_kwargs,**global_model)
                    adv_acc,adv_loss = c.adv_validate(**global_model,**validation_kwargs)
                global_adv_accs.append(adv_acc)
                global_adv_losses.append(adv_loss)
            global_adv_accs = np.array(global_adv_accs)
            global_adv_losses = np.array(global_adv_losses)

            if test_dataset is not None:
                if isinstance(global_model,list):
                    test_adv_accs,test_adv_losses = [],[]
                    for n in range(len(global_model)):
                        # test_param = dict(validation_kwargs,**global_model[n])
                        test_adv_acc_n,test_adv_loss_n = self.clients[n].adv_validate(testset=test_dataset,**global_model[n],**validation_kwargs)
                        test_adv_accs.append(test_adv_acc_n)
                        test_adv_losses.append(test_adv_loss_n)
                    test_adv_acc,test_adv_loss = np.mean(test_adv_accs),np.mean(test_adv_losses)
                    
                else:
                    # test_param = dict(validation_kwargs,**global_model)
                    test_adv_acc,test_adv_loss = self.clients[0].adv_validate(testset=test_dataset,**global_model,**validation_kwargs)
            else:
                test_adv_acc,test_adv_loss = 0,None

            # collect
            weighted_global_adv_acc,weighted_global_adv_loss = np.sum(global_adv_accs*self.weights),np.sum(global_adv_losses*self.weights)

        res["val_adv_acc"] = global_adv_accs
        res["val_adv_loss"] = global_adv_losses
        res["weighted_val_adv_acc"] = weighted_global_adv_acc
        res["weighted_val_adv_loss"] = weighted_global_adv_loss
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
                    pickle.dump({"weights":self.weights,
                                "epoch":self.epochs,
                                "chosen_clients":self.chosen_clients, 
                                "train_loss":self.local_losses,
                                "weighted_train_loss":self.weighted_local_losses,
                                "val_acc":self.global_accs,
                                "val_loss":self.global_losses,
                                "weighted_val_acc":self.weighted_global_accs, 
                                "weighted_val_loss":self.weighted_global_losses,
                                "val_adv_acc":self.global_adv_accs,
                                "val_adv_loss":self.global_adv_losses,
                                "weighted_val_adv_acc":self.weighted_global_adv_accs,
                                "weighted_val_adv_loss":self.weighted_global_adv_losses,
                                "test_acc":self.test_accs,
                                "test_loss":self.test_losses,
                                "test_adv_acc":self.test_adv_accs,
                                "test_adv_loss":self.test_adv_losses}, stat_f)
                # store the model if it attains the highest validation loss
                weighted_global_clean_adv_accs = 0.4*np.array(self.weighted_global_accs) + 0.6*np.array(self.weighted_global_adv_accs)
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
    