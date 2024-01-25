from selection.selector import Selector
import numpy as np

class PowerD_Selector(Selector):
    """
    Client Selection in Federated Learning: Convergence Analysis and Power-of-Choice Selection Strategies
    """
    def __init__(self, total_client_num, weights=None, loss_value=None, d=0, **kwargs) -> None:
        super().__init__(total_client_num, weights)
        if loss_value is not None:
            self.loss_value = np.array(loss_value)
        else:
            self.loss_value = np.ones(self.total_client_num)*100.0 # init as inf if not given the loss
        self.d = d

    def select(self,select_num=0,**kwargs):
        candidate_list = np.arange(self.total_client_num)
        A = np.random.choice(candidate_list, self.d, replace=False,p=self.weights)
        idxs_users = A[np.argsort(self.loss_value[A])[-select_num:]]
        return idxs_users
    
    def stat_update(self, stat_info = None,**kwargs):
        """
        stat_info: should be the global loss of all clients at the end of this round
        """
        if stat_info is not None:
            stat_info = stat_info["val_loss"]
            self.loss_value = np.array(stat_info)
