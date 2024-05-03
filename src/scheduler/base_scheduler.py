import copy
class base_scheduler():
    """
    a base scheduler for standard training, 
    which determines the learning rate according to the round only. 
    """
    def __init__(self,args):
        self.round = 0
        self.total_round = args["epochs"]
        self.args = args
        self.best_weighted_acc = 0.0
        self.smooth_length = 0

    def training_params(self,**kwargs):
        lr = self.args["lr"]
        if self.args["lr_decay"] is not None:
            if self.args["lr_schedule"] is None:
                lr = lr*(self.args["lr_decay"]**self.round)
            elif self.round in self.args["lr_schedule"]:
                loc = self.args["lr_schedule"].index(self.round)
                lr = lr*(self.args["lr_decay"]**(loc+1))
        
                
        args = copy.deepcopy(self.args)
        args["lr"] = lr

        return args
    
    def monitor_params(self,**kwargs):
        return {}
    
    def stat_update(self,epoch,stat_info,**kwargs):
        # update statistical information
        
        weighted_acc = stat_info["weighted_val_acc"]
   
        if weighted_acc > self.best_weighted_acc:
            self.best_weighted_acc = weighted_acc
            self.smooth_length = 0
        else:
            self.smooth_length += epoch-self.round
        self.round = epoch
        if self.round >= self.total_round: #or self.smooth_length >= int(0.1*self.total_round):
            return False
        else:
            print('\n | Global Training Round : {} |\n'.format(self.round))
            return True

class base_AT_scheduler(base_scheduler):
    """
    a base scheduler which determines the training hyperparameter according 
    to the round only. For example, it can adjust the learning rate 
    and the adversarial training phase according to the training round.
    """
    def __init__(self,args):
        super().__init__(args)
        self.target_clean_adv_ratio = args["target_clean_adv_ratio"]


    def training_params(self,**kwargs):
        args = super().training_params()
        if self.round < self.args["adv_warmup"]:
            args["adv_ratio"] = self.args["warmup_adv_ratio"]
            if args["adv_ratio"] == 0:
                args["adv_train"] = False
            #args["lr"] = self.args["lr"] # do not decay the lr until adv_warmup

        
        return args
    
    def monitor_params(self,**kwargs):
        
        if self.round < self.args["adv_warmup"]:
            adv_test = False
        else:
            adv_test = self.args["adv_test"]

        return {"adv_test":adv_test,"clean_adv_ratio":self.target_clean_adv_ratio}
    
    def stat_update(self,epoch,stat_info,**kwargs):
        # update statistical information
        if not self.args["adv_train"] or self.round >= self.args["adv_warmup"]:
            if "weighted_val_adv_acc" in stat_info:
                weighted_acc = stat_info["weighted_val_acc"] + self.target_clean_adv_ratio*stat_info["weighted_val_adv_acc"]
                
            else:
                weighted_acc = stat_info["weighted_val_acc"]
                
   
            if weighted_acc > self.best_weighted_acc:
                self.best_weighted_acc = weighted_acc
                self.smooth_length = 0
            else:
                self.smooth_length += epoch-self.round
        self.round = epoch
        if self.round >= self.total_round: #or self.smooth_length >= int(0.1*self.total_round):
            return False
        else:
            print('\n | Global Training Round : {} |\n'.format(self.round))
            return True