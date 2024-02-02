import copy
class base_scheduler():
    """
    a base scheduler which determines the training hyperparameter according 
    to the round only. For example, it can adjust the learning rate 
    and the adversarial training phase according to the training round.
    """
    def __init__(self,args):
        self.round = 0

        self.args = args

    def set(self,**kwargs):
        lr = self.args["lr"]
        if self.args["lr_decay"] is not None:
            if self.args["lr_schedule"] is None:
                lr = lr*(self.args["lr_decay"]**self.round)
            elif self.round in self.args["lr_schedule"]:
                loc = self.args["lr_schedule"].index(self.round)
                lr = lr*(self.args["lr_decay"]**(loc+1))
        adv_train = self.args["adv_train"]
        if self.round < self.args["adv_warmup"]:
            adv_train = False
        
                
        args = copy.deepcopy(self.args)
        args["lr"] = lr
        args["adv_train"] = adv_train

        
        return args
    
    def stat_update(self,epoch,**kwargs):
        self.round = epoch

        