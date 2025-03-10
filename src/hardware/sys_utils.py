import numpy as np
from numpy.random import RandomState
from fvcore.nn import FlopCountAnalysis, parameter_count, flop_count_table
import torch
from math import ceil
from scipy.stats import truncnorm
import csv
import copy


unique_runtime_app_list = ['idle', '1080p', '4k', 'inference', 'detection', 'web']
unique_perf_degrade_dic = {'idle': 1, '1080p': 0.735, '4k': 0.459, 'inference': 0.524, 'detection': 0.167 , 'web': 0.231}
unique_mem_avail_dic = {'idle': 1, '1080p': 0.5, '4k': 0.25, 'inference': 0.75, 'detection': 0.0625, 'web': 0.125}

NETWORK_LIST = ['DSL', 'cable', '4G', '5G', 'wifi', 'fiber', 'satelite']
# upload/download speed: Byte/s |  latency: s
NETWORK_DIC = {'DSL': {'upload speed': (1e6/8,20e6/8), 'download speed': (3e6/8,145e6/8),'latency': (11e-3,40e-3)},
               'cable': {'upload speed': (1e6/8,50e6/8), 'download speed': (25e6/8,1000e6/8), 'latency': (13e-3,27e-3)},
               '4G': {'upload speed': (1e6/8,30e6/8), 'download speed': (9e6/8,60e6/8), 'latency': (30e-3,50e-3)},
               '5G': {'upload speed': (24e6/8,58e6/8), 'download speed': (501e6/8,635e6/8), 'latency': (4e-3,10e-3)},
               'wifi': {'upload speed': (1e6/8,50e6/8), 'download speed': (25e6/8,300e6/8), 'latency': (10e-3,50e-3)},
               'fiber': {'upload speed': (250e6/8,1000e6/8), 'download speed': (30e6/8,5000e6/8), 'latency': (10e-3,12e-3)},
               'satelite': {'upload speed': (3e6/8,5e6/8), 'download speed': (12e6/8,350e6/8), 'latency': (594e-3,624e-3)}}



def get_devices(args,seed=None):
    """
    Returns the devices of each client, which is a dict: idx -> (device_name,perf,mem)
    The available device list is defined in args.flsys_profile_info, and the 
    sampling probability is determined by args.sys_scaling_factor.
    """
    rs = RandomState(seed)
    unique_client_device_dic = read_client_device_info(args.flsys_profile_info)
    device_name_list, device_perf_list, device_mem_list, device_eff_bw_list\
        = sample_devices(args.num_users,rs,unique_client_device_dic,
                         args.sys_scaling_factor)
    
    user_devices = {i:(device_name_list[i],device_perf_list[i],device_mem_list[i], device_eff_bw_list[i]) for i in range(args.num_users)}
    return user_devices
    


def read_client_device_info(flsys_profile_info):

    """
    arg: flsys_profile_info

    return: unique_client_device_dic - {'client_device_name': GFLOPS, GB, Eff_BW}
    """

    unique_client_device_dic = {}
    with open(flsys_profile_info+".csv",'r',newline='') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            client_device_name = row[0]
            if client_device_name != 'Client':
                if client_device_name not in unique_client_device_dic.keys():
                    unique_client_device_dic[client_device_name] = row[1:] 

    # file = open(flsys_profile_info, 'r')
    # rows = []
    # while True:
    #     line = file.readline()
    #     if not line: break
    #     line = line.replace('\n', ' ').replace('\t', ' ')
    #     splitline = line.split(" ")
    #     splitline = splitline[:-1]
    #     compact_line =[]
    #     for item in splitline:
    #         if item != '':
    #             compact_line.append(item)

    #     rows.append(compact_line)
        # client_device_name = compact_line[0]
        # if client_device_name != 'Client':
        #     if client_device_name not in unique_client_device_dic.keys():
        #         unique_client_device_dic[client_device_name] = [compact_line[1]] #FLOPS
        #         unique_client_device_dic[client_device_name].append(compact_line[2]) #Bytes
        #         unique_client_device_dic[client_device_name].append(compact_line[3]) #Bytes/s
        

    
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
    unique_eff_bw_list = []

    

    for k in device_dic.keys():
        unique_name_list.append(k)
        unique_perf_list.append(float(device_dic[k][0]))
        unique_mem_list.append(float(device_dic[k][1]))
        mul_perf_mem_list.append(float(device_dic[k][0])*float(device_dic[k][1]))
        unique_eff_bw_list.append(float(device_dic[k][2]))
    
    scaled_mul_perf_mem_list = np.array([v ** sys_scaling_factor for v in mul_perf_mem_list])
    prob_list = scaled_mul_perf_mem_list/np.sum(scaled_mul_perf_mem_list)


    client_device_name_list = []
    client_device_perf_list = [] # FLOPS
    client_device_mem_list  = [] # B
    client_device_eff_bw_list = [] # B/s


    device_id_list = rs.choice(unique_id_list, p=prob_list, size=num_users)
    # device_id_list = [random.randint(0,num_unique_device-1) for _ in range(num_users)]
    for id in device_id_list:
        client_device_name_list.append(unique_name_list[id])
        client_device_perf_list.append(unique_perf_list[id])
        client_device_mem_list.append(unique_mem_list[id])
        client_device_eff_bw_list.append(unique_eff_bw_list[id])
    
    return client_device_name_list, client_device_perf_list, client_device_mem_list, client_device_eff_bw_list


def sample_runtime_app(rs):
    # generate the specific runtime applications for each client
    # runtime application for each client is dynamic - different random seed per epoch

    runtime_app = rs.choice(unique_runtime_app_list)
    mem_avai_factor = rs.uniform(0.0,0.2)
    return runtime_app, unique_perf_degrade_dic[runtime_app],mem_avai_factor

def sample_networks(rs):
    def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
        return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)
    # generate network speed/latency for each client
    # type of network and runtime speed/latency of network for each client is dynamic - different rand seed per epoch


    
    network = rs.choice(NETWORK_LIST)
    
    lower_speed = NETWORK_DIC[network]['upload speed'][0]
    upper_speed = NETWORK_DIC[network]['upload speed'][1]
    runtime_speed_gen = get_truncated_normal(mean=(lower_speed+upper_speed)/2, sd=upper_speed/5, low=lower_speed, upp=upper_speed)
    upload_runtime_speed = runtime_speed_gen.rvs(random_state=rs)

    lower_speed = NETWORK_DIC[network]['download speed'][0]
    upper_speed = NETWORK_DIC[network]['download speed'][1]
    runtime_speed_gen = get_truncated_normal(mean=(lower_speed+upper_speed)/2, sd=upper_speed/5, low=lower_speed, upp=upper_speed)
    download_runtime_speed = runtime_speed_gen.rvs(random_state=rs)

    lower_latency = NETWORK_DIC[network]['latency'][0]
    upper_latency = NETWORK_DIC[network]['latency'][1]
    runtime_latency_gen = get_truncated_normal(mean=(lower_latency+upper_latency)/2, sd=upper_latency/5, low=lower_latency, upp=upper_latency)
    runtime_latency = runtime_latency_gen.rvs(random_state=rs) 

    
    return network, [download_runtime_speed,upload_runtime_speed], runtime_latency


    
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

    data_Byte = 4 # FP32
    def __init__(self,model,inputsize,default_local_eps=1,optimizer='sgd',**opt_kwargs):
        self.inputsize = inputsize
        self.default_local_eps = default_local_eps
        if optimizer == 'adam':
            self.param_mem_scale = 5
        elif optimizer == 'sgd':
            if 'momentum' in opt_kwargs and opt_kwargs['momentum'] > 0:
                self.param_mem_scale = 3
            else:
                self.param_mem_scale = 2 
        else:
            raise RuntimeError("Not a supported optimizer: {} for model profiling!".format(optimizer))
        self.module_list, self.flops_dict, self.num_parameter_dict, self.mem_dict\
              = self.profile_model(model,inputsize) 


    def register_feature_hook(self,model,output):
        def in_feature_hook(module,fea_in,fea_out):
            output[module.called_name] = fea_in[0].size()
            return None
        handles = []
        for n,m in model.named_modules():
            m.called_name = n
            if n in model.feature_layer_list:
                h = m.register_forward_hook(in_feature_hook)
                handles.append(h)
        return handles

        

    
    

    
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
            if sum != dic['total']:
                print("[Warning]: The total value is not equal to the sum of all items!")

        # initialize the dicts
        in_feature_dict = {} # layer: input_feature_size
        self.out_feature_dict = {} # module: output_feature_size
        self.num_classes = model.output_size

        handles = self.register_feature_hook(model,in_feature_dict)
        x = torch.rand(inputsize,device=next(model.parameters()).device)
        flops = FlopCountAnalysis(model,x)
        flops.unsupported_ops_warnings(False)
        # print(flop_count_table(flops))
        _flops_per_module = flops.by_module()
        _params_per_module = parameter_count(model)
        
        self.in_feature_dict = copy.deepcopy(in_feature_dict)
        for h in handles:
            h.remove()

        flops_per_module = {}
        params_per_module = {}
        mem_per_module = {}
        module_name_list = []

        flops_per_module['total'] = _flops_per_module['']
        params_per_module['total'] = _params_per_module['']
        mem_per_module['total'] = self.param_mem_scale*self.data_Byte*params_per_module['total']
        for k in self.in_feature_dict.keys():
            mem_per_module['total'] += self.data_Byte*np.prod(self.in_feature_dict[k])
        
        if hasattr(model,"module_list"): # if this model is modularized
            for key in model.module_list:
                if isinstance(key,str):
                    flops_per_module[key] = _flops_per_module[key]
                    params_per_module[key] = _params_per_module[key]
                    mem_per_module[key] = self.param_mem_scale*self.data_Byte*params_per_module[key] # memory for parameter
                    if key in self.in_feature_dict.keys():# this module is a layer
                        mem_per_module[key] += self.data_Byte*np.prod(self.in_feature_dict[key])

                    else: # this module is a block
                        for layer in self.in_feature_dict.keys(): # calculate every layer in this block
                            if (key+'.') in layer:
                                mem_per_module[key] += self.data_Byte*np.prod(self.in_feature_dict[layer])
                                
                else: # this module is a combination of multiple sub-modules
                    name = "+".join(key)
                    flops_per_module[name] = 0
                    params_per_module[name] = 0
                    mem_per_module[name] = 0
                    for l in key:# calculate the flops and parameters in every sub-module
                        flops_per_module[name] += _flops_per_module[l]
                        params_per_module[name] += _params_per_module[l]                  
                        if l in self.in_feature_dict.keys(): # this sub-module is a layer
                            mem_per_module[name] += self.data_Byte*np.prod(self.in_feature_dict[l])
                        
                        else: # this sub-module is a block
                            for layer in self.in_feature_dict.keys(): # calculate every layer in this block
                                if (l+'.') in layer:
                                    mem_per_module[name] += self.data_Byte*np.prod(self.in_feature_dict[layer])
                        
                    mem_per_module[name] += self.param_mem_scale*self.data_Byte*params_per_module[name] # memory for parameter
                    
            # print(params_per_module)
            validate(flops_per_module)
            validate(params_per_module)
            validate(mem_per_module)

            module_name_list = list(flops_per_module.keys())
            module_name_list.remove("total")

            # calculate the output size of each module
            cum_output = x
            for module in module_name_list:
                cum_output = model.module_forward(cum_output,[module,])
                self.out_feature_dict[module] = cum_output.size()
                


        return module_name_list, flops_per_module, params_per_module, mem_per_module
    
    def training_latency(self,performance=None,memory=None,
                         eff_bandwidth=None,access_latency=0.17,
                         network_bandwidth=None,network_latency=0.0,
                         batches=None,adv_iters=0,adv_ratio=1.0,
                         module_list=None,partial_frac=1.0,**kwargs):
        """
        Calculate the training latency of the whole model with the model profile.
        module_list: a list of atom module names in self.module_lists, 
        notice that this name may not be used to access self.in_feature_dict.
        The inputsizes can be a list of sizes, one for each minibatch.
        The total training latency should be calculated as the sum of all minibatches.
        If memory_bandwidth is not None, then the memory-to-cache latency will be counted.
        If network_bandwidth is not None, then the server-device communication latency will be counted.
        """


        total_latency = 0
        total_comp_latency = 0
        total_mem_latency = 0
        
        if module_list == None:
            module_list = ['total']
        
        
        total_params = 0
        total_feature_size = 0
        flops_req = 0
        for k in module_list:
            total_params += int(self.num_parameter_dict[k])
            total_feature_size += (int(self.mem_dict[k])-self.param_mem_scale*self.data_Byte*int(self.num_parameter_dict[k]))//self.data_Byte
            flops_req += int(self.flops_dict[k])
        if module_list[-1] in self.out_feature_dict:
            total_feature_size += int(np.prod(self.out_feature_dict[module_list[-1]]))
        if partial_frac < 1.0:
            total_params = int(total_params*partial_frac**2)
            total_feature_size = int(total_feature_size*partial_frac)
            flops_req = int(flops_req*partial_frac**2)
        
        if batches is None:
            batches = [self.inputsize]*self.default_local_eps

        for n,batch in enumerate(batches):
            calibrated_factor = batch[0]/self.inputsize[0]
            batch_memory_req = self.data_Byte*calibrated_factor*total_feature_size \
                                + self.param_mem_scale*self.data_Byte*total_params
            batch_flops_req = calibrated_factor*flops_req

            # forward + backward computation (performance in GFLOPs)
            batch_computation_time = 2*batch_flops_req/performance if performance is not None else 0

            if memory is not None and eff_bandwidth is not None and access_latency is not None:
                if batch_memory_req>memory: # the required memory exceeds the available memory
                    # we adopt load and offload method to train the module
                    # offload feature in forward and load feature in backward + load and offload parameters in forward and backward + load and offload opt status in backward
                    memory_access_size = self.data_Byte*(2*calibrated_factor*total_feature_size \
                                                         + 4*total_params \
                                                         + 2*(self.param_mem_scale-2)*total_params)
                    forward_mem_req = self.data_Byte*(calibrated_factor*total_feature_size + total_params) # 1x parameter + 1x feature in forward
                    backward_mem_req = batch_memory_req
                    memory_access_times = ceil(forward_mem_req/memory) + ceil(backward_mem_req/memory)
                    batch_memory_access_time = memory_access_size/eff_bandwidth+access_latency*memory_access_times
                else:# we do not need to offload the parameter and intermediate features
                    # load the input (MB) once for every iters_per_input
                    batch_memory_access_time = self.data_Byte*int(np.prod(batch))/eff_bandwidth+access_latency
                    if n == 0: # only load the parameter (MB) at the beginning
                        batch_memory_access_time += self.data_Byte*total_params/eff_bandwidth
            if adv_iters>0:# adversarial training part
                batch_computation_time += 2*batch_flops_req*adv_iters*adv_ratio/performance if performance is not None else 0
                # adversarial training only requires 1xparam+1xfeature
                # we should reserve an extra memory for the input gradients, 
                # but this part can usually be compensated by the released features in backward, so we did not count it here
                adv_batch_memory_req = self.data_Byte*(adv_ratio*calibrated_factor*total_feature_size + total_params)
                if memory is not None and eff_bandwidth is not None and access_latency is not None:
                    if adv_batch_memory_req>memory: # the required memory exceeds the available memory
                        # we adopt load and offload method to train the module
                        # offload feature in forward and load feature in backward + load and offload parameters in forward and backward
                        adv_memory_access_size = self.data_Byte*(2*calibrated_factor*adv_ratio*total_feature_size + 4*total_params)
                        adv_forward_mem_req = self.data_Byte*(calibrated_factor*adv_ratio*total_feature_size + total_params) # 1x parameter + 1x feature in forward
                        adv_backward_mem_req = adv_batch_memory_req
                        adv_memory_access_times = ceil(adv_forward_mem_req/memory) + ceil(adv_backward_mem_req/memory)
                        batch_memory_access_time += adv_iters*(adv_memory_access_size/eff_bandwidth+access_latency*adv_memory_access_times)



            total_comp_latency += batch_computation_time
            total_mem_latency += batch_memory_access_time
        
        
        if network_bandwidth is not None:
            # download and upload time
            communication_time = self.data_Byte*total_params/network_bandwidth[0]+self.data_Byte*total_params/network_bandwidth[1]+2*network_latency
        else:
            communication_time = 0
        
        # we do not consider communication time right now
        total_latency = total_comp_latency + total_mem_latency #+ communication_time
        
        return {"total":total_latency,"computation":total_comp_latency,"memory":total_mem_latency,"communication":communication_time}
        

            