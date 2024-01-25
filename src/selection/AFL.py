from selection.selector import Selector
import numpy as np

class AFL_Selector(Selector):
    """
    Active Federated Learning
    """
    def __init__(self,total_client_num,weights=None,loss_value=None,alpha1=0.75,alpha2=0.01,alpha3=0.1,**kwargs) -> None:
        super().__init__(total_client_num,weights)
        if loss_value is not None:
            self.loss_value = np.array(loss_value)
        else:
            self.loss_value = np.ones(self.total_client_num)*100.0 # init as a large value if not given the loss
        self.alpha1=alpha1
        self.alpha2=alpha2
        self.alpha3=alpha3
        

    def select(self,select_num=0,**kwargs):
        num_users = self.total_client_num
        candidate_list = np.arange(self.total_client_num)
        # AFL
        delete_num = int(self.alpha1*num_users)
        sel_num = int((1-self.alpha3)*select_num)
        tmp_value = np.vstack([candidate_list,self.loss_value[candidate_list]])
        tmp_value = tmp_value[:,tmp_value[1,:].argsort()]
        prob = np.exp(self.alpha2*tmp_value[1,delete_num:])
        prob = prob/np.sum(prob)
        sel1 = np.random.choice(np.array(tmp_value[0,delete_num:],dtype=np.int64),sel_num,replace=False,p=prob)
        remain = set(candidate_list)-set(sel1)
        sel2 = np.random.choice(list(remain),select_num-sel_num,replace = False)
        idxs_users = np.append(sel1,sel2)
        return idxs_users
    
    def stat_update(self, selected_clients=None, stat_info=None, **kwargs):
        """
        stat_info: should be the global loss of selected clients at the beginning of this round
        """
        if stat_info is not None:
            stat_info = stat_info["val_loss"][selected_clients]
            self.loss_value[selected_clients] = np.array(stat_info)*np.sqrt(self.weights[selected_clients])
