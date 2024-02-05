from models.model_utils import get_net
from fvcore.nn import FlopCountAnalysis,flop_count_table, parameter_count
import torch
from hardware.sys_utils import feature_summary

# model = get_net('vgg11','cifar',num_classes=10)
# print("model-----------------------------------------")
# print(model)
# # print("----------------------------------------------")

# x = torch.rand([10,3,32,32])
# flops = FlopCountAnalysis(model,x)

# print("flops per module----------------------------- ")
# flops_per_module = flops.by_module()
# #flops_per_module = parameter_count(model)
# sorted_flops_per_module = {}
# sum = 0
# dot_count = 0
# for key in flops_per_module.keys():
#     for c in key:
#         if c == '.':
#             dot_count += 1
#     if (dot_count == 0 and 'layer' not in key and 'classifier' not in key and 'features' not in key) \
#         or (dot_count == 1 and 'weight' not in key and 'bias' not in key):
#         sorted_flops_per_module[key] = flops_per_module[key]
#     dot_count = 0

# print(sorted_flops_per_module)
# for key in sorted_flops_per_module:
#     if key != '':
#         sum += sorted_flops_per_module[key]
# print(sum)

# temp = 0
# print(sorted_flops_per_module)
# for key in sorted_flops_per_module.keys():
#     if "layer3." in key:
#         for c in key:
#             if c == '.':
#                 temp += 1
#         if temp == 1:
#             sum += sorted_flops_per_module[key]
#             print(sum)
#     temp = 0        
# print("--------")
# print(sum)
# print(sorted_flops_per_module['layer3'])

# print(flops_per_module['layer3'])
# for key in flops_per_module.keys():
#     print(key)
# print("----------------------------------------------")
# print("params per module-----------------------------")
# print(parameter_count(model))
# print("----------------------------------------------")

def profile_model(model, inputsize):
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
    layer_name_list = []
    for k in flops_per_module.keys():
        if k != 'normalize' and k != 'total':
            layer_name_list.append(k)
            

    # print(flops_per_module)
    # print(params_per_module)

    # print(len(flops_per_module))
    # print(len(params_per_module))
    return layer_name_list, flops_per_module, params_per_module
    


model = get_net('resnet50','cifar',num_classes=10,adv_norm=True,modularization=True)
print(model)
inputsize = [10,3,32,32]
i = torch.rand(inputsize)
layer_name_list, flops_per_module, params_per_module = profile_model(model,inputsize)

feature_summary.register_feature_hook(model,layer_name_list)
model(i)
intmd_feature_dic = feature_summary.in_feature_dict
print(intmd_feature_dic)
print(feature_summary.out_feature_dict)
input()
params_per_intmd = {}

for key in layer_name_list:
    current_size = list(intmd_feature_dic[key])
    param_size = 1

    for i in range(len(current_size)):
        param_size = param_size * current_size[i]
    
    params_per_intmd[key] = param_size

print(flops_per_module)
print(params_per_module)
print(params_per_intmd)

l1 = []
l2 = []
l3 = []

for key in layer_name_list:
    l1.append(int(flops_per_module[key]))
    l2.append(int(params_per_module[key]))
    l3.append(int(params_per_intmd[key]))

max_flops = max(l1)
max_mem = max(l2) * 12 + max(l3) * 4
print(max_flops,max_mem)
# module_id_list = [i for i in range(len(layer_name_list))]
partition_module_id_list =[]
partition_module_list = []

current_partition_module_id_list = []
current_partition_module_list = []

current_sum_flops = 0
current_sum_mem = 0

# assert max(flops) or max(params) > constraints 


pos = 0
while pos < len(layer_name_list):
    l = layer_name_list[pos]
    # get the intermidiate features
    # intmd_feature = get_feature_size(current_partition)
    if current_sum_flops + int(flops_per_module[l]) <= max_flops and \
    current_sum_mem + (int(params_per_module[l]) + int(params_per_intmd[l])) * 3 * 4 <= max_mem:
        current_sum_flops += int(flops_per_module[l])
        current_sum_mem += (int(params_per_module[l]) + int(params_per_intmd[l])) * 3 * 4
        current_partition_module_list.append(l)
        current_partition_module_id_list.append(pos)
    
    else:
        assert current_partition_module_list != [], "max_flops/mem is too small!"
        partition_module_list.append(current_partition_module_list)
        partition_module_id_list.append(current_partition_module_id_list)

        current_partition_module_id_list = []
        current_partition_module_list = []
        current_sum_flops = 0
        current_sum_mem = 0

        pos -= 1

    pos += 1

print(partition_module_list)
print(partition_module_id_list)

