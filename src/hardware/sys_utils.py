import numpy as np
from numpy.random import RandomState
from fvcore.nn import FlopCountAnalysis, parameter_count
import torch
from math import ceil


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


    
class model_summary():
    """
    This class will profile a given model and get its number of parameters,
    flops of each layer, and sizes of intermediate features. 
    The granulty of the model is defined by the model itself.
    self.module_list: name of atom module lists, could be a combination of multiple layers (block)
    self.flops_dict: {module_name: flops}
    self.num_parameter_dict: {module_name: parameters}
    self.mem_dict: {module_name: parameters}
    self.in_feature_dict: {layer_name: input_feature_size}
    self.out_feature_dict: {module_name: output_feature_size}
    """
    def __init__(self,model,inputsize,default_local_eps=[1,]):
        self.inputsize = inputsize
        self.default_local_eps = default_local_eps
        self.module_list, self.flops_dict, self.num_parameter_dict, self.mem_dict\
              = self.profile_model(model,inputsize) 


    def register_feature_hook(self,model):
        for n,m in model.named_modules():
            m.called_name = n
            if n in model.feature_layer_list:
                m.register_forward_hook(self.in_feature_hook)

        

    def in_feature_hook(self,module,fea_in,fea_out):
        self.in_feature_dict[module.called_name] = fea_in[0].size()
        return None
    

    
    def profile_model(self, model, inputsize):
        """
        get the flops and paramenters of each layer (block) in a model 

        return: flops_per_module, params_per_module and mem_per_module
        *** note that the last item in the returned dictionaries is ('total', flops/params) ***
        """
        def validate(dic):
            sum = 0
            for key in dic:
                if key != 'total':
                    sum += dic[key]
            assert sum == dic['total'], "Wrong profiling data!"

        # initialize the dicts
        self.in_feature_dict = {} # layer: input_feature_size
        self.out_feature_dict = {} # module: output_feature_size
        self.num_classes = model.output_size

        self.register_feature_hook(model)
        x = torch.rand(inputsize,device=next(model.parameters()).device)
        flops = FlopCountAnalysis(model,x)
        flops.unsupported_ops_warnings(False)
        _flops_per_module = flops.by_module()
        _params_per_module = parameter_count(model)
        flops_per_module = {}
        params_per_module = {}
        mem_per_module = {}
        module_name_list = []

        flops_per_module['total'] = _flops_per_module['']
        params_per_module['total'] = _params_per_module['']
        mem_per_module['total'] = 3*4*params_per_module['total']
        for k in self.in_feature_dict.keys():
            mem_per_module['total'] += 4*np.prod(self.in_feature_dict[k])
        
        if hasattr(model,"module_list"): # if this model is modularized
            for key in model.module_list:
                if isinstance(key,str):
                    flops_per_module[key] = _flops_per_module[key]
                    params_per_module[key] = _params_per_module[key]
                    mem_per_module[key] = 3*4*params_per_module[key] # memory for parameter
                    if key in self.in_feature_dict.keys():# this module is a layer
                        mem_per_module[key] += 4*np.prod(self.in_feature_dict[key])

                    else: # this module is a block
                        for layer in self.in_feature_dict.keys(): # calculate every layer in this block
                            if (key+'.') in layer:
                                mem_per_module[key] += 4*np.prod(self.in_feature_dict[layer])
                                
                else: # this module is a combination of multiple sub-modules
                    name = "+".join(key)
                    flops_per_module[name] = 0
                    params_per_module[name] = 0
                    mem_per_module[name] = 0
                    for l in key:# calculate the flops and parameters in every sub-module
                        flops_per_module[name] += _flops_per_module[l]
                        params_per_module[name] += _params_per_module[l]                  
                        if l in self.in_feature_dict.keys(): # this sub-module is a layer
                            mem_per_module[name] += 4*np.prod(self.in_feature_dict[l])
                        
                        else: # this sub-module is a block
                            for layer in self.in_feature_dict.keys(): # calculate every layer in this block
                                if (l+'.') in layer:
                                    mem_per_module[name] += 4*np.prod(self.in_feature_dict[layer])
                        
                    mem_per_module[name] += 3*4*params_per_module[name] # memory for parameter
                    
            validate(flops_per_module)
            validate(params_per_module)
            validate(mem_per_module)

            module_name_list = list(flops_per_module.keys())
            module_name_list.remove("total")

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
            self.out_feature_dict[module_name_list[-1]]=[inputsize[0],self.num_classes]
            
            # out_feature_size_per_module = {}
            # for k in self.out_feature_dict:
            #     result = 1
            #     for i in range(len(self.out_feature_dict[k])):
            #         result *= self.out_feature_dict[k][i]
            #     out_feature_size_per_module[k] = result


        return module_name_list, flops_per_module, params_per_module, mem_per_module
    
    def training_latency(self,module_list=None,batches=None,iters_per_input=1,performance=None,memory=None,eff_bandwidth=None,access_latency=None,network_bandwidth=None):
        """
        Calculate the training latency of the whole model with the model profile.
        module_list: a list of atom module names in self.module_lists, 
        notice that this name may not be used to access self.in_feature_dict.
        The inputsizes can be a list of sizes, one for each minibatch.
        The total training latency should be calculated as the sum of all minibatches.
        If memory_bandwidth is not None, then the memory-to-cache latency will be counted.
        If network_bandwidth is not None, then the server-device communication latency will be counted.
        """

        # To do: network latency emulation ...
        # ...

        total_latency = 0
        
        if module_list == None:
            module_list = ['total']
        
        
        total_params = 0
        total_feature_size = 0
        flops_req = 0
        for k in module_list:
            total_params += int(self.num_parameter_dict[k])
            total_feature_size += (int(self.mem_dict[k])-4*3*int(self.num_parameter_dict[k]))//4
            flops_req += int(self.flops_dict[k])
        if module_list[-1] in self.out_feature_dict:
            total_feature_size += int(np.prod(self.out_feature_dict[module_list[-1]]))
        
        if batches is None:
            batches = [self.inputsize]*self.default_local_eps

        for n,batch in enumerate(batches):
            calibrated_factor = batch[0]/self.inputsize[0]
            batch_memory_req = 4*calibrated_factor*total_feature_size \
                                + 3*4*total_params
            batch_flops_req = calibrated_factor*flops_req

            # forward + backward computation
            batch_computation_time = 2*iters_per_input*batch_flops_req/performance if performance is not None else 0

            batch_memory_access_time = 0
            if memory is not None and eff_bandwidth is not None and access_latency is not None:
                if batch_memory_req>memory: # the required memory exceeds the available memory
                    # we adopt load and offload method to train the module
                    # offload feature in forward and load feature in backward + load and offload parameters in forward and backward
                    memory_access_size = 2*4*calibrated_factor*total_feature_size + 4*4*total_params
                    memory_access_times = ceil(memory_access_size/2/memory) # offload can be done together with load
                    batch_memory_access_time = iters_per_input*(memory_access_size/eff_bandwidth+access_latency*memory_access_times)
                else:# we do not need to offload the parameter and intermediate features
                    # load the input once for every iters_per_input
                    batch_memory_access_time = 4*int(np.prod(batch))/eff_bandwidth+access_latency
                    if n == 0: # only load the parameter at the beginning
                        batch_memory_access_time += 4*total_params/eff_bandwidth

            # since the computation and memory access can be parallelized, 
            # we take the larger one as the final latency
            batch_on_device_time = batch_computation_time + batch_memory_access_time

            total_latency += batch_on_device_time
        
        if network_bandwidth is not None:
            # download and upload time
            communication_time = 4*total_params/network_bandwidth[0]+4*total_params/network_bandwidth[1]
            total_latency += communication_time
        
        return total_latency
        # baseline
        # if module_list == None:
        #     # the out feature size of a batch for a given model with batch size = self.inputsize[0]
        #     _one_batch_feature_size = 0
        #     for k in self.out_feature_size_dic:
        #         _one_batch_feature_size += self.out_feature_size_dic[k]
            
        #     # the total out feature size of batches
        #     total_out_feature_size = 0
        #     total_flops = 0
        #     for i in range(total_iter):
        #         batch_size = batches[i][0]
        #         calibrated_factor = batch_size / self.inputsize[0]
        #         total_out_feature_size += calibrated_factor * _one_batch_feature_size
        #         total_flops += calibrated_factor * self.flops_dict['total']


        #     total_params_access = self.num_parameter_dict['total'] * total_iter

        #     # calculate the execution time: ns (if bandwidth - GB/s and performance - GFLOPS)
        #     local_data_access_time = (total_out_feature_size + total_params_access * 2) * 2 / eff_bandwidth + access_latency * total_iter
        #     compute_time = total_flops / performance
        #     server_client_comm_time = self.num_parameter_dict['total'] / network_bandwidth

        # else:
        # _one_batch_feature_size = 0
        # _one_batch_flops = 0
        # total_params = 0
        # for k in module_list:
        #     _one_batch_feature_size += self.out_feature_size_dic[k]
        #     _one_batch_flops += self.flops_dict[k]
        #     total_params += self.num_parameter_dict[k]
        
        # total_out_feature_size = 0
        # total_flops = 0
        # for i in range(total_iter):
        #     batch_size = batches[i][0]
        #     calibrated_factor = batch_size / self.inputsize[0]
        #     total_out_feature_size += calibrated_factor * _one_batch_feature_size
        #     total_flops += calibrated_factor * _one_batch_flops
        
        # total_params_access = total_params

        # local_data_access_time = total_params_access / eff_bandwidth
        # compute_time = total_flops / performance
        # server_client_comm_time = total_params / network_bandwidth

        # total_training_latency = local_data_access_time + compute_time + server_client_comm_time
        
        # return total_training_latency

            