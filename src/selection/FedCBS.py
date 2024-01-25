from selection.selector import Selector
import numpy as np
from copy import deepcopy
import pdb
class FedCBS_Selector(Selector):
    def __init__(self, total_client_num, data_matrix, weights=None, probability_power_init=1, probability_power_final=10, **kwargs) -> None:
        super().__init__(total_client_num, weights)
        self.probability_power_init = probability_power_init
        self.probability_power_final = probability_power_final
        
        
        self.data_matrix = data_matrix # the number of each class on each client, N x C
        self.num_classes = self.data_matrix.shape[1]


        self.data_size_client = np.sum(self.data_matrix,axis=1)


        self.S = np.matmul(self.data_matrix,self.data_matrix.transpose())


        
        self.T_pull = np.ones(self.total_client_num,dtype=np.int64)
        self.epoch = 0

    def select(self,select_num,**kwargs):
        single_client_select_ls_dict  = dict()
        select_measure_ls =[]

        terminate_loop=False

        repeat_times=0


        while repeat_times<=5 or not terminate_loop:
            continue_loop = False
            single_client_select_ls = []
            
            clients_ls_for_selection = list(range(self.total_client_num))

            probability_power = self.probability_power_init
            probability_power_last = self.probability_power_init - (
                    self.probability_power_final - self.probability_power_init) / int(select_num -1)

            for i in range(select_num):
                probability_ls_for_selection = []
                if i== 0:
                    for j in clients_ls_for_selection:
                        single_client_potential_ls_last = deepcopy(single_client_select_ls)
                        single_client_potential_ls = single_client_select_ls + [j]
                        probability_ls_for_selection.append(
                            1 / ((self.S[
                                    np.ix_(single_client_potential_ls, single_client_potential_ls)].sum()) / (
                                    self.data_size_client[np.ix_(
                                        single_client_potential_ls)].sum()) ** 2 - 1 / self.num_classes) ** probability_power + np.sqrt(
                                3 * np.log(self.epoch + 1) / (2 * self.T_pull[j])))

                    max_idx = probability_ls_for_selection.index(max(probability_ls_for_selection))
                    client_selected_this_epoch = clients_ls_for_selection[max_idx]
                    single_client_select_ls.append(client_selected_this_epoch)
                    clients_ls_for_selection.remove(client_selected_this_epoch)

                if i >= 1:
                    probability_power = probability_power + (self.probability_power_final - self.probability_power_init) / int(select_num - 1)
                    probability_power_last = probability_power_last + (
                                self.probability_power_final - self.probability_power_init) / int(select_num - 1)

                    for j in clients_ls_for_selection:
                        single_client_potential_ls_last = deepcopy(single_client_select_ls)
                        single_client_potential_ls = single_client_select_ls + [j]

                        if (self.S[
                                np.ix_(single_client_potential_ls,
                                    single_client_potential_ls)].sum() / (self.data_size_client[
                            np.ix_(single_client_potential_ls)].sum()) ** 2 - 1 / self.num_classes) ** probability_power <= 1e-3:
                            client_probablity_Denominator = 1e-3
                        else:
                            client_probablity_Denominator = (self.S[
                                                                np.ix_(single_client_potential_ls,
                                                                        single_client_potential_ls)].sum() / (
                                                                self.data_size_client[np.ix_(
                                                                    single_client_potential_ls)].sum()) ** 2 - 1 / self.num_classes) ** probability_power
                        probability_ls_for_selection.append(
                            (self.S[np.ix_(single_client_potential_ls_last,single_client_potential_ls_last)].sum() / (self.data_size_client[np.ix_(single_client_potential_ls_last)].sum()) ** 2 - 1 / self.num_classes) ** probability_power_last / client_probablity_Denominator)

                    probability_ls_sum = sum(probability_ls_for_selection)
                    probability_ls_for_selection = [prob / probability_ls_sum for prob in
                                                    probability_ls_for_selection]


                    try:
                        client_selected_this_epoch = \
                            np.random.choice(clients_ls_for_selection, 1, p=probability_ls_for_selection)[0]

                    except ValueError as e:
                        if "probabilities contain NaN" in str(e):
                            print("Encountered NaN in probabilities, trying again...")
                            continue_loop=True
                            break

                    single_client_select_ls.append(client_selected_this_epoch)
                    clients_ls_for_selection.remove(client_selected_this_epoch)

            if not continue_loop:
                single_client_select_ls_dict[repeat_times] = single_client_select_ls
                select_measure_ls.append(
                    self.S[np.ix_(single_client_select_ls, single_client_select_ls)].sum() / (
                        self.data_size_client[np.ix_(single_client_select_ls)].sum()) ** 2 - 1 / self.num_classes)
                terminate_loop=True
                repeat_times+=1

            else:
                terminate_loop = False






        min_idx = select_measure_ls.index(min(select_measure_ls))
        final_single_client_select_ls = single_client_select_ls_dict[min_idx]


        return  final_single_client_select_ls
    
    def stat_update(self, epoch=None, selected_clients=None, data_matrix = None, **kwargs):
        """
        stat_info: should be the local loss of selected clients at the end of this round
        sys_info: should be the true training + communication time of selected clients
        """
        if epoch is not None:
            self.epoch = epoch
        # update statistcal and systematic utility
        if selected_clients is not None:
            self.T_pull[selected_clients] = self.T_pull[selected_clients] + 1
        
        if data_matrix is not None:
            self.data_matrix = data_matrix # the number of each class on each client, N x C
            self.num_classes = self.data_matrix.shape[1]
            self.data_size_client = np.sum(self.data_matrix,axis=1)
            self.S = np.matmul(self.data_matrix,self.data_matrix.transpose())