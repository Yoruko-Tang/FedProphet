import copy
class base_scheduler():
    """
    a base scheduler that can only adjust the learning rate
    """
    def __init__(self,args):
        self.round = 0

        self.args = args

    def set(self,**kwargs):
        if self.args["lr_decay"] is not None:
            if self.args["lr_schedule"] is None:
                lr = self.args["lr"]*(self.args["lr_decay"]**self.round)
            elif self.round in self.args["lr_schedule"]:
                loc = self.args["lr_schedule"].index(self.round)
                lr = self.args["lr"]*(self.args["lr_decay"]**(loc+1))
        
        args = copy.deepcopy(self.args)
        args["lr"]=lr
        
        return args
    
    def stat_update(self,epoch,**kwargs):
        self.round = epoch

        