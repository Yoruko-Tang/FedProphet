class base_scheduler():
    """
    a base scheduler returns the local training hyperparameters only based on the training round
    """
    def __init__(self,optimizer,local_ep,local_bs,lr,lr_decay=None,
                 lr_schedule=None,momentum=0.0,reg=0.0,**kwargs):
        self.optimizer = optimizer
        self.local_ep = local_ep
        self.local_bs = local_bs
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_schedule = lr_schedule
        self.momentum = momentum
        self.reg = reg
        self.round = 0

    def set(self,idx):
        if self.lr_decay is not None:
            if self.lr_schedule is None:
                lr = self.lr*(self.lr_decay**self.round)
            elif self.round in self.lr_schedule:
                loc = self.lr_schedule.index(self.round)
                lr = self.lr*(self.lr_decay**(loc+1))
        res = {
            "iteration":self.local_ep,
            "batchsize":self.local_bs,
            "lr":lr,
            "optimizer":self.optimizer,
            "momentum":self.momentum,
            "reg":self.reg
        }
        return res
    
    def stat_update(self,epoch,stat_info,sys_info,**kwargs):
        self.round = epoch
        