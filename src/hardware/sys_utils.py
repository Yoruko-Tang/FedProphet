import numpy as np
from numpy.random import RandomState
from fvcore.nn import FlopCountAnalysis, parameter_count
import torch
from copy import deepcopy

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


def training_latency(module_dic,inputsizes,performance,memory):
    """
    Calculate the training latency given a model, device performance and device memory.
    The inputsizes can be a list of sizes, one for each minibatch.
    The total training latency should be calculated as the sum of all minibatches.
    """
    # [[10,3,32,32],[]]
    return 0

       
    
class model_summary():
    """
    This class will profile a given model and get its number of parameters,
    flops of each layer, and sizes of intermediate features. 
    The granulty of the model is defined by the model itself.
    """
    def __init__(self,model,inputsize):
        self.in_feature_dict = {}
        self.out_feature_dict = {}
        self.module_list, self.flops_dict, self.num_parameter_dict, self.mem_dict = self.profile_model(model,inputsize) 


    def register_feature_hook(self,model):
        # out_feature_layer_list = []
        # for module in model.module_list:
        #     if isinstance(module,list):
        #         out_feature_layer_list.append(module[-1])
        #     elif isinstance(module,str):
        #         out_feature_layer_list.append(module)
        for n,m in model.named_modules():
            m.called_name = n
            if n in model.feature_layer_list:
                m.register_forward_hook(self.in_feature_hook)
            # elif n in out_feature_layer_list:
                # m.regisiter_forward_hook(self.out_feature_hook)
        

    def in_feature_hook(self,module,fea_in,fea_out):
        self.in_feature_dict[module.called_name] = fea_in[0].size()
        # self.out_feature_dict[module.called_name] = fea_out.size()
        return None
    
    # def out_feature_hook(self,module,fea_in,fea_out):
    #     self.out_feature_dict[module.called_name] = fea_out.size()
    #     return None
    

    
    def profile_model(self, model, inputsize):
        """
        get the flops and paramenters of each layer (block) in a model 

        return: flops_per_module, params_per_module and mem_per_module
        *** note that the last item in the returned dictionaries is ('total', flops/params) ***
        """
        self.register_feature_hook(model)
        x = torch.rand(inputsize)
        flops = FlopCountAnalysis(model,x)
        _flops_per_module = flops.by_module()
        _params_per_module = parameter_count(model)
        flops_per_module = {}
        params_per_module = {}
        mem_per_module = {}

        for key in model.module_list:
            if isinstance(key,str):
                flops_per_module[key] = _flops_per_module[key]
                params_per_module[key] = _params_per_module[key]
                mem_per_module[key] = 3*4*params_per_module[key] # memory for parameter
                if key in self.in_feature_dict.keys():# this module is a layer
                    mem_per_module[key] += 4*self.mul(self.in_feature_dict[key])

                else: # this module is a block
                    for layer in self.in_feature_dict.keys(): # calculate every layer in this block
                        if (key+'.') in layer:
                            mem_per_module[key] += 4*self.mul(self.in_feature_dict[layer])
                            
            else: # this module is a combination of multiple sub-modules
                name = "+".join(key)
                flops_per_module[name] = 0
                params_per_module[name] = 0
                mem_per_module[name] = 0
                for l in key:# calculate the flops and parameters in every sub-module
                    flops_per_module[name] += _flops_per_module[l]
                    params_per_module[name] += _params_per_module[l]                  
                    if l in self.in_feature_dict.keys(): # this sub-module is a layer
                        mem_per_module[name] += 4*self.mul(self.in_feature_dict[l])
                    
                    else: # this sub-module is a block
                        for layer in self.in_feature_dict.keys(): # calculate every layer in this block
                            if (l+'.') in layer:
                                mem_per_module[name] += 4*self.mul(self.in_feature_dict[layer])
                    
                mem_per_module[name] += 3*4*params_per_module[name] # memory for parameter
                

        module_name_list = list(flops_per_module.keys())
        flops_per_module['total'] = _flops_per_module['']
        params_per_module['total'] = _params_per_module['']
        mem_per_module['total'] = sum(mem_per_module.values())

        # calculate the output size of each module
        for n in range(len(module_name_list)-1):
            self.out_feature_dict[module_name_list[n]] = None
            nidx = n+1
            while self.out_feature_dict[module_name_list[n]] == None and nidx < len(module_name_list):
                # find the first layer in the next module
                for nl in module_name_list[nidx].split('+'):
                    if nl in self.in_feature_dict.keys():
                        self.out_feature_dict[module_name_list[n]] = self.in_feature_dict[nl]
                        break
                    else:
                        for layer in self.in_feature_dict.keys():
                            if (nl+'.') in layer:
                                self.out_feature_dict[module_name_list[n]] = self.in_feature_dict[layer]
                                break
                        if self.out_feature_dict[module_name_list[n]] is not None:
                            break
                nidx += 1
            if self.out_feature_dict[module_name_list[n]] == None:
                raise RuntimeWarning("Cannot fetch the output size of the module: "+ module_name_list[n])
        self.out_feature_dict[module_name_list[-1]]=[inputsize[0],]

        def validate(dic):
            sum = 0
            for key in dic:
                if key != 'total':
                    sum += dic[key]
            assert sum == dic['total'], "Wrong profiling data!"
        
        validate(flops_per_module)
        validate(params_per_module)


        return module_name_list, flops_per_module, params_per_module, mem_per_module
    
    @staticmethod
    def mul(list):
        res = 1
        for i in list:
            res *= i
        return res
    

    def model_partition(self,max_module_flops=None,max_module_mem=None):
        """
        partition the model in a greedy manner, with each module in the 
        max_flops and max_mem constraints
        """
        
        flops_dict = deepcopy(self.flops_dict)
        flops_dict.pop("total")
        mem_dict = deepcopy(self.mem_dict)
        mem_dict.pop("total")

        # use the largest atom layer as the lower bound
        if max_module_flops is None:
            max_module_flops = max(flops_dict.values())
        else:
            assert max(flops_dict.values()) <= max_module_flops, "Max allowed flops is too small to partition!"
        if max_module_mem is None: 
            max_module_mem = max(mem_dict.values())
        else:
            assert max(mem_dict.values()) <= max_module_mem, "Max allowed memory is too small to partition!"

        
        partition_module_list = []

        current_partition_module_list = []

        current_sum_flops = 0
        current_sum_mem = 0

        for atom in flops_dict.keys():
            if current_sum_flops + int(flops_dict[atom]) <= max_module_flops and \
            current_sum_mem + int(mem_dict[atom]) <= max_module_mem:
                current_sum_flops += int(flops_dict[atom])
                current_sum_mem += int(mem_dict[atom])
                current_partition_module_list += atom.split("+")
            
            else:
                partition_module_list.append(current_partition_module_list)
                current_partition_module_list = atom.split("+")
                current_sum_flops = int(flops_dict[atom])
                current_sum_mem = int(mem_dict[atom])


        
        # not including 'normalize' and 'total'
        return partition_module_list