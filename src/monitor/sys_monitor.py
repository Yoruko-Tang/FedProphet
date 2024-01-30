import random
from scipy.stats import truncnorm

class Sys_Monitor:
    """
    collect the validation loss of each client
    """
    def __init__(self,client_num,flsys_profile_info,device_random_seed,scaling_factor):

        self.unique_client_device_dic = self.read_client_device_info(flsys_profile_info)
        self.device_name_list, self.device_perf_list, self.device_mem_list \
            = self.sample_devices(client_num,device_random_seed,self.unique_client_device_dic,scaling_factor)
        self.unique_runtime_app_list = ['idle', '1080p', '4k', 'inference', 'detection', 'web']
        self.unique_perf_degrade_dic = {'idle': 1, '1080p': 0.735, '4k': 0.459, 'inference': 0.524, 'detection': 0.167 , 'web': 0.231}
        self.unique_mem_avail_dic = {'idle': 1, '1080p': 0.5, '4k': 0.25, 'inference': 0.75, 'detection': 0.0625, 'web': 0.125}

        self.device_perf_degrade_factor_list, self.device_mem_avail_factor_list, self.device_runtime_app_list \
            = self.sample_runtime_app(client_num,self.unique_runtime_app_list,self.unique_perf_degrade_dic,self.unique_mem_avail_dic)
        self.device_runtime_perf_list = self.get_runtime_value(self.device_perf_list, self.device_perf_degrade_factor_list)
        self.device_runtime_mem_list = self.get_runtime_value(self.device_mem_list, self.device_mem_avail_factor_list)
        
        # self.model_stat 

        
    
    def model_stat(model):

        """
        return module-required mem;  flops
        """
        
    def collect(self):
        self.device_runtime_perf_list = self.get_runtime_value(self.device_perf_list, self.device_perf_degrade_factor_list)
        self.device_runtime_mem_list = self.get_runtime_value(self.device_mem_list, self.device_mem_avail_factor_list)
        return None
    
    def profile(self,id,layer):
        return None

    def get_runtime_value(self, value_list, factor_list):
        runtime_value_list =[]

        for i in range(len(value_list)):
            runtime_value_list.append(float(value_list[i])*factor_list[i])

        return runtime_value_list


    def sample_runtime_app(self, num_clients, runtime_app_list, perf_degrade_dic, mem_avail_dic):
        # generate the specific runtime applications for each client
        # runtime application for each client is dynamic - different random seed per epoch
        num_runtime_status = len(runtime_app_list)
        # print("num of runtime status: ", num_runtime_status)

        runtime_app_for_each_client_list = []
        degrade_factor_for_each_client_list = []
        mem_avail_factor_for_each_client_list = []

        runtime_app_id_list = [random.randint(0,num_runtime_status-1) for x in range(num_clients)]
        # print(runtime_app_id_list)
        for id in runtime_app_id_list:
            runtime_app_for_each_client_list.append(runtime_app_list[id])
        # print(runtime_app_for_each_client_list)

        for key in runtime_app_for_each_client_list:
            degrade_factor_for_each_client_list.append(perf_degrade_dic[key])
            mem_avail_factor_for_each_client_list.append(mem_avail_dic[key])
            
        # print(degrade_factor_for_each_client_list)

        return degrade_factor_for_each_client_list, mem_avail_factor_for_each_client_list, runtime_app_id_list
    
    def sample_devices(self,client_num,random_seed,device_dic,scaling_factor):
        """
        sample the device with its theoretical performance (GFLOPS) and 
        maximal available memory (GB) for each client
        """
        num_unique_device = len(device_dic)
        unique_id_list = [id for id in range(num_unique_device)]
        unique_name_list = []
        unique_perf_list = []
        unique_mem_list  = []
        mul_perf_mem_list = []
        

        for k in device_dic.keys():
            unique_name_list.append(k)
            unique_perf_list.append(device_dic[k][0])
            unique_mem_list.append(device_dic[k][1])
            mul_perf_mem_list.append(float(device_dic[k][0])*float(device_dic[k][1]))
        
        scaled_mul_perf_mem_list = [v ** scaling_factor for v in mul_perf_mem_list]
        total = sum(scaled_mul_perf_mem_list)
        prob_list = [scaled_v / total for scaled_v in scaled_mul_perf_mem_list]


        client_device_name_list = []
        client_device_perf_list = [] #GFLOPS
        client_device_mem_list  = [] #GB

        random.seed(random_seed)
        device_id_list = random.choices(unique_id_list, weights=prob_list, k=client_num)
        # device_id_list = [random.randint(0,num_unique_device-1) for _ in range(client_num)]
        for id in device_id_list:
            client_device_name_list.append(unique_name_list[id])
            client_device_perf_list.append(unique_perf_list[id])
            client_device_mem_list.append(unique_mem_list[id])
        
        return client_device_name_list, client_device_perf_list, client_device_mem_list
        


    def read_client_device_info(self,flsys_profile_info):

        """
        arg: flsys_profile_info

        return: unique_client_device_dic - {'client_device_name': GFLOPS, GB}
        """

        unique_client_device_dic = {}
        file = open(flsys_profile_info, 'r')

        while True:
            line = file.readline()
            if not line: break
            line = line.replace('\n', ' ').replace('\t', ' ')
            splitline = line.split(" ")
            splitline = splitline[:-1]
            compact_line =[]
            for item in splitline:
                if item != '':
                    compact_line.append(item)

            client_device_name = compact_line[0]
            if client_device_name != 'Client':
                if client_device_name not in unique_client_device_dic.keys():
                    unique_client_device_dic[client_device_name] = [compact_line[1]] #GFLOPS
                    unique_client_device_dic[client_device_name].append(compact_line[2]) #GB
        
        return unique_client_device_dic