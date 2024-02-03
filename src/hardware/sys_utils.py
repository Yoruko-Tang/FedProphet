import numpy as np
from numpy.random import RandomState
from fvcore.nn import FlopCountAnalysis, parameter_count
import torch

unique_runtime_app_list = ['idle', '1080p', '4k', 'inference', 'detection', 'web']
unique_perf_degrade_dic = {'idle': 1, '1080p': 0.735, '4k': 0.459, 'inference': 0.524, 'detection': 0.167 , 'web': 0.231}
unique_mem_avail_dic = {'idle': 1, '1080p': 0.5, '4k': 0.25, 'inference': 0.75, 'detection': 0.0625, 'web': 0.125}

def get_devices(args,seed=None):
    """
    Returns the devices of each client, which is a dict: idx -> (device_name,perf,mem)
    The available device list is defined in args.flsys_profile_info, and the 
    sampling probability is determined by args.sys_scaling_factor.
    """
    rs = RandomState(seed)
    unique_client_device_dic = read_client_device_info(args.flsys_profile_info)
    device_name_list, device_perf_list, device_mem_list \
        = sample_devices(args.num_users,rs,unique_client_device_dic,
                         args.sys_scaling_factor)
    
    user_devices = {i:(device_name_list[i],device_perf_list[i],device_mem_list[i]) for i in range(args.num_users)}
    return user_devices
    


def read_client_device_info(flsys_profile_info):

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

def sample_devices(num_users,rs,device_dic,sys_scaling_factor):
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
        unique_perf_list.append(float(device_dic[k][0]))
        unique_mem_list.append(float(device_dic[k][1]))
        mul_perf_mem_list.append(float(device_dic[k][0])*float(device_dic[k][1]))
    
    scaled_mul_perf_mem_list = np.array([v ** sys_scaling_factor for v in mul_perf_mem_list])
    prob_list = scaled_mul_perf_mem_list/np.sum(scaled_mul_perf_mem_list)


    client_device_name_list = []
    client_device_perf_list = [] #GFLOPS
    client_device_mem_list  = [] #GB


    device_id_list = rs.choice(unique_id_list, p=prob_list, size=num_users)
    # device_id_list = [random.randint(0,num_unique_device-1) for _ in range(num_users)]
    for id in device_id_list:
        client_device_name_list.append(unique_name_list[id])
        client_device_perf_list.append(unique_perf_list[id])
        client_device_mem_list.append(unique_mem_list[id])
    
    return client_device_name_list, client_device_perf_list, client_device_mem_list


def sample_runtime_app(rs):
    # generate the specific runtime applications for each client
    # runtime application for each client is dynamic - different random seed per epoch

    runtime_app = rs.choice(unique_runtime_app_list)
    return runtime_app, unique_perf_degrade_dic[runtime_app],unique_mem_avail_dic[runtime_app]


def profile_model(model, inputsize):
    """
    get the flops and paramenters of each layer (block) in a model 

    return: flops_per_module and params_per_module
    *** note that the last item in the returned dictionaries is ('total', flops/params) ***
    """
    x = torch.rand(inputsize)
    flops = FlopCountAnalysis(model,x)
    _flops_per_module = flops.by_module()
    _params_per_module = parameter_count(model)
    flops_per_module = {}
    params_per_module = {}
    dot_count = 0
    for key in _flops_per_module.keys():
        for c in key:
            if c == '.':
                dot_count += 1
        if (dot_count == 0 and 'layer' not in key and 'classifier' not in key and 'features' not in key) \
            or (dot_count == 1 and 'weight' not in key and 'bias' not in key):
            flops_per_module[key] = _flops_per_module[key]
            params_per_module[key] = _params_per_module[key]
        dot_count = 0

    flops_per_module['total'] = flops_per_module.pop('')
    params_per_module['total'] = params_per_module.pop('')

    def validate(dic):
        sum = 0
        for key in dic:
            if key != 'total':
                sum += dic[key]
        assert sum == dic['total'], "Wrong profiling data!"
    
    validate(flops_per_module)
    validate(params_per_module)

    return flops_per_module, params_per_module

def training_latency(module_dic,inputsizes,performance,memory):
    """
    Calculate the training latency given a model, device performance and device memory.
    The inputsizes can be a list of sizes, one for each minibatch.
    The total training latency should be calculated as the sum of all minibatches.
    """
    # [[10,3,32,32],[]]
    return 0

def model_partition(model,inputsize,max_flops,max_mem,num_classes):
    """
    partition the model in a greedy manner, with each module in the 
    max_flops and max_mem constraints
    """

    def greedy_partition(flops_per_module,params_per_module,inputsize,max_flops,max_mem):
        """
        max_flops: flops
        max_mem: bytes
        memory consumption relates to input size...
        """
        module_id_list = [i for i in range(len(flops_per_module))]
        partition_module_id_list =[]
        current_partition_module_id_list = []

        partition_module_list = []
        current_partition_module_list = []
        current_sum_flops = 0
        current_sum_mem = 0

        # assert max(flops) or max(params) > constraints 


        pos = 0
        for key in flops_per_module.keys():
            # get the intermidiate features
            # intmd_feature = get_feature_size(current_partition)
            if current_sum_flops + flops_per_module[key] <= max_flops and \
            current_sum_mem + (params_per_module[key]+intmd_feature) * 3 * 4 <= max_mem:
                current_sum_flops += flops_per_module[key]
                current_sum_mem += (params_per_module[key]+intmd_feature) * 3 * 4
                current_partition_module_list.append(key)
                current_partition_module_id_list.append(pos)
            
            else:
                partition_module_list.append(current_partition_module_list)
                partition_module_id_list.append(current_partition_module_id_list)
                
                
            
            
            

            

        
    


class feature_summary():

    in_feature_list = {}


    @staticmethod
    def register_feature_hook(model,layer_list):
        for n,m in model.named_modules():
            if n in layer_list:
                m.called_name = n
                m.register_forward_hook(feature_summary.in_feature_hook)
                
    
    @staticmethod
    def in_feature_hook(module,fea_in,fea_out):
        feature_summary.in_feature_list[module.called_name] = fea_in[0].size()
        return None
    
    @staticmethod
    def get_total_feature_num():
        total = 0
        for fs in feature_summary.in_feature_list.values():
            volumn = 1
            for s in fs:
                volumn*=s
            total += volumn
        return total.item()
    

    