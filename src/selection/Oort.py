from selection.selector import Selector
import numpy as np

class Oort_Selector(Selector):
    """
    Oort: Efficient Federated Learning via Guided Participant Selection
    """
    def __init__(self,total_client_num,weights=None,pacer_step=0.001,step_window=20,epsilon_range=[0.9,0.2],epsilon_decay=0.98,oort_alpha=2.0,oort_c=0.95,loss_init=None,sys_init=None,**kwargs) -> None:
        super().__init__(total_client_num,weights)
        self.delta = pacer_step
        self.T = self.delta
        self.W = step_window
        self.init_epsilon = epsilon_range[0]
        self.min_epsilon = epsilon_range[1]
        self.epsilon = self.init_epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha = oort_alpha
        self.c = oort_c

        if loss_init is not None:
            self.stat_utils = np.array(loss_init)*self.weights
        else:
            self.stat_utils = np.array([100.0]*self.total_client_num) # init as a large number
        if sys_init is not None:
            self.sys_utils = np.array(sys_init)
        else:
            self.sys_utils = np.array([self.T]*self.total_client_num) # init as T

        self.last_inv_round = np.array([None]*self.total_client_num)
        self.explored_clients = []

        self.round = 0

        self.utils_hist = []

    def exploitation_utility(self,idx_users):
        utils = []
        for i in idx_users:
            assert i in self.explored_clients and self.last_inv_round[i] is not None, "Client %d has not been explored!"%i
            u = self.stat_utils[i]+np.sqrt(0.1*np.log(self.round)/self.last_inv_round[i])
            if self.T < self.sys_utils[i]:
                u *= ((self.T/self.sys_utils[i])**self.alpha)
            utils.append(u)
        return utils
    
    def exploration_utility(self,idx_users):
        utils = []
        for i in idx_users:
            u = self.stat_utils[i]
            if self.T < self.sys_utils[i]:
                u *= ((self.T/self.sys_utils[i])**self.alpha)
            utils.append(u)
        return utils
        
    def select(self,select_num,**kwargs):
        exploitation_selection, exploration_selection = np.array([],dtype=np.int64),np.array([],dtype=np.int64)
        # exploitation
        exploitation_num = min([int((1-self.epsilon)*select_num),len(self.explored_clients)])
        if exploitation_num > 0:
            utils = self.exploitation_utility(self.explored_clients)
            sorted_utils = sorted(utils,reverse=True)
            utils_th = sorted_utils[exploitation_num-1]*self.c
            utils = np.array(utils)
            admit_clients = np.array(self.explored_clients)[utils>=utils_th]
            admit_utils = utils[utils>=utils_th]
            # select clients with large utils
            exploitation_selection = np.random.choice(admit_clients,exploitation_num,p=admit_utils/np.sum(admit_utils),replace=False).tolist()
        
        # eploration
        exploration_num = min([select_num-exploitation_num,self.total_client_num-len(self.explored_clients)])
        if exploration_num>0:
            unexplored_clients = list(set(list(range(self.total_client_num)))-set(self.explored_clients))
            utils = self.exploration_utility(unexplored_clients)
            utils = np.array(utils)
            # select clients with large utils
            exploration_selection = np.random.choice(unexplored_clients,exploration_num,p=utils/np.sum(utils),replace=False).tolist()
        
        
        selection = np.append(exploitation_selection,exploration_selection)
        return selection
    
    def stat_update(self, epoch=None, selected_clients=None, stat_info=None, sys_info=None, **kwargs):
        """
        stat_info: should be the local (training) loss of selected clients at the end of this round
        sys_info: should be the true training + communication time of selected clients
        """
        if epoch is not None:
            self.round = epoch
        # update statistcal and systematic utility
        if stat_info is not None:
            if selected_clients is not None:
                stat_info = stat_info["train_loss"]
                self.explored_clients = list(set(self.explored_clients+selected_clients.tolist()))
                self.last_inv_round[selected_clients] = self.round
            else:
                stat_info = stat_info["val_loss"]
            self.stat_utils[selected_clients] = np.array(stat_info)*self.weights[selected_clients]
            # update T
            if len(self.explored_clients)>0:
                self.utils_hist.append(np.mean(self.stat_utils[self.explored_clients]))
                if self.round >= 2 * self.W and self.round % self.W == 0:

                    utilLastPacerRounds = sum(self.utils_hist[-2*self.W:-self.W])
                    utilCurrentPacerRounds = sum(self.utils_hist[-self.W:])

                    # Cumulated statistical utility becomes flat, so we need a bump by relaxing the pacer
                    if abs(utilCurrentPacerRounds - utilLastPacerRounds) <= utilLastPacerRounds * 0.1:
                        self.T = self.T + self.delta

                    # change sharply -> we decrease the pacer step
                    elif abs(utilCurrentPacerRounds - utilLastPacerRounds) >= utilLastPacerRounds * 5:
                        self.T = max(self.delta, self.T - self.delta)

        if sys_info is not None:
            self.sys_utils[selected_clients] = np.array(sys_info)
    
        if len(self.explored_clients)==self.total_client_num:
            self.epsilon = 0.0
        elif epoch is not None:
            self.epsilon = max([self.epsilon*self.epsilon_decay,self.min_epsilon])

        
