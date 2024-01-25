from selection.selector import Selector
import numpy as np
from copy import deepcopy

class Harmony_Selector(Selector):
    def __init__(self, total_client_num, data_matrix, weights=None, omega = 1.0, xi=np.sqrt(2), epsilon=0.5, runtime = None, **kwargs) -> None:
        super().__init__(total_client_num, weights)
        self.omega = omega
        self.xi = xi
        self.epsilon = epsilon
        
        self.data_matrix = data_matrix # the number of each class on each client, N x C
        self.dist_client = self.data_matrix/np.sum(self.data_matrix,axis=1,keepdims=True) # [P_i]
        self.dist_class = np.sum(self.data_matrix,axis=0)/np.sum(self.data_matrix) # P_exp
        self.stat_utils = np.array([Harmony_Selector.KL_Div(self.dist_client[i],self.dist_class) for i in range(self.total_client_num)])

        # store the systematic statistics
        if runtime is not None:
            self.sys_utils = np.array(runtime)
        else:
            self.sys_utils = np.zeros(total_client_num)
        
        self.explored_clients = []
        self.client_select_time = np.zeros(self.total_client_num,dtype=np.int64)
        self.round = 0

    @staticmethod
    def KL_Div(p,q):
        return np.sum(p*(np.log((p+1e-4)/(q+1e-4))))
    
    def exploitation_utility(self,idx_users):
        utils = []
        for i in idx_users:
            assert i in self.explored_clients and self.client_select_time[i] >= 1, "Client %d has not been explored!"%i
            u = -(self.stat_utils[i]+self.omega*self.sys_utils[i])+self.xi*np.sqrt(np.log(self.round)/(self.client_select_time[i]))
            utils.append(u)
        return utils
    
    def exploration_utility(self,idx_users):
        utils = []
        for i in idx_users:
            u = -(self.stat_utils[i]+self.omega*self.sys_utils[i])
            utils.append(u)
        return utils

    def select(self,select_num,**kwargs):
        exploitation_selection, exploration_selection = np.array([],dtype=np.int64),np.array([],dtype=np.int64)
        # exploitation
        exploitation_num = min([int((1-self.epsilon)*select_num),len(self.explored_clients)])
        if exploitation_num > 0:
            utils = self.exploitation_utility(self.explored_clients)
            utils = np.array(utils)
            explored_clients = np.array(self.explored_clients)
            exploitation_selection = explored_clients[np.argsort(utils)[-exploitation_num:]] # select clients with largest util

        
        # eploration
        exploration_num = min([select_num-exploitation_num,self.total_client_num-len(self.explored_clients)])
        if exploration_num>0:
            unexplored_clients = list(set(list(range(self.total_client_num)))-set(self.explored_clients))
            utils = self.exploration_utility(unexplored_clients)
            utils = np.array(utils)
            unexplored_clients = np.array(unexplored_clients)
            exploration_selection = unexplored_clients[np.argsort(utils)[-exploration_num:]] # select clients with largest util
        
        selection = np.append(exploitation_selection,exploration_selection)
        return selection
    
    def data_shaper(self,idx_users):
        # given the selected users in idx_users, 
        # datashaper make sure that their overall data distribution is close to P_exp
        selected_data_matrix = self.data_matrix[idx_users,:] # number of data of each selected client of each class
        selected_data_client = np.sum(selected_data_matrix,axis=1) # total number of data of each selected client
        selected_data_class = np.sum(selected_data_matrix,axis=0) # total number of selected data of each class
        dist_selected_client = self.dist_client[idx_users,:] # distribution of data of each selected client of each class
        star = np.min(selected_data_class) # the smallest number of data in a class
        B = deepcopy(self.data_matrix)
        for i in range(len(selected_data_class)):
            if selected_data_class[i] > star:
                delta = selected_data_class[i]-star # number of data that needs to be removed from this class
                sorted_client_class_idx = np.argsort(dist_selected_client[:,i])[::-1] # sort P_j and get the order

                sorted_client_class_dist = dist_selected_client[sorted_client_class_idx,i] # sorted P_j
                sorted_client_data_num = selected_data_client[sorted_client_class_idx] # C_i for each client i in the sorted order as P_j
                sorted_client_idx = np.array(idx_users)[sorted_client_class_idx] # The ID order of the sorted clients

                # reducing from the client with the highest p_{i,j}
                X = [sorted_client_class_dist[0]]
                CX = [sorted_client_data_num[0]]
                X_idx = [sorted_client_idx[0]]
                for j in range(1,len(sorted_client_class_dist)):
                    Y = sorted_client_class_dist[j]
                    CY = sorted_client_data_num[j]
                    Y_idx = sorted_client_idx[j]
                    Omega = np.sum([(X[k]-Y)*CX[k] for k in range(len(X))]) # the reduced number
                    if Omega < delta: # the reduced number is not enough
                        X.append(Y)
                        CX.append(CY)
                        X_idx.append(Y_idx)
                    else: # the reduced number is equal or larger than the required number
                        for k in range(len(X)):
                            # calculate the remaining data number for each client
                            B[X_idx[k],i] = B[X_idx[k],i]-int((delta/Omega)*CX[k]*(X[k]-Y))# shrink the reducing number by delta/omega
                        break

        return B

                    
    def stat_update(self, epoch=None, selected_clients=None, data_matrix=None, sys_info=None,**kwargs):
        """
        stat_info: should be the data matrix of all clients in this round
        sys_info: should be the estimated runtime of all clients 
        """
        if epoch is not None:
            self.round = epoch
        if selected_clients is not None:
            self.client_select_time[selected_clients] = self.client_select_time[selected_clients] + 1
            self.explored_clients = list(set(self.explored_clients+selected_clients.tolist()))

        if len(self.explored_clients)==self.total_client_num:
            self.epsilon = 0.0
        
        if data_matrix is not None: # update the data matrix 
            self.data_matrix =  data_matrix# the number of each class on each client, N x C
            self.dist_client = self.data_matrix/np.sum(self.data_matrix,axis=1,keepdims=True) # [P_i]
            self.dist_class = np.sum(self.data_matrix,axis=0)/np.sum(self.data_matrix) # P_exp
            self.stat_utils = np.array([Harmony_Selector.KL_Div(self.dist_client[i],self.dist_class) for i in range(self.total_client_num)])
        if sys_info is not None: # update the training time
            self.sys_utils = np.array(sys_info)

        
