from models.model_utils import get_net
from fvcore.nn import FlopCountAnalysis,flop_count_table, parameter_count
import torch
from hardware.sys_utils import model_summary
from scheduler.module_scheduler import module_scheduler
from client import *

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
    


# model = get_net('vgg16_bn','cifar',num_classes=10,adv_norm=True,modularization=True)
# inputsize = [64,3,32,32]


model = get_net('resnet50','imagenet',num_classes=256,adv_norm=True,modularization=True)
inputsize = [64,3,224,224]
# ms = model_summary(model,inputsize,optimizer='adam')
ms = model_summary(model,inputsize,optimizer='sgd',momentum=0.9)

print("module list---------------------------")
print(ms.module_list)
print("\n")

print("flops dictionary----------------------")
print(ms.flops_dict)
print("\n")

print("number of parameters dictionary-------")
print(ms.num_parameter_dict)
print("\n")

print("memory dictionary---------------------")
print(ms.mem_dict)
print("\n")

print("output dictionary---------------------")
print(ms.out_feature_dict)
print("\n")

