import numpy as np


class Selector():
    def __init__(self,total_client_num,weights=None) -> None:
        self.total_client_num=total_client_num
        self.weights = weights if weights is not None else np.ones(total_client_num)/total_client_num

    def select(self,select_num=0,**kwargs):
        raise NotImplementedError("Selection strategy has not been defined!")

    def stat_update(self,**kwargs):
        pass



class Random_Selector(Selector):
    def __init__(self, total_client_num, weights=None,**kwargs) -> None:
        super().__init__(total_client_num, weights)
    
    def select(self,select_num=0,**kwargs):
        candidate_list = np.arange(self.total_client_num)
        return np.random.choice(candidate_list, select_num, replace=False)












    

