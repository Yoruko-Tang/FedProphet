from models import get_net
from fvcore.nn import FlopCountAnalysis,flop_count_table, parameter_count
import torch


model = get_net('resnet50','cifar',num_classes=10)
# print("model-----------------------------------------")
print(model)
# print("----------------------------------------------")

x = torch.rand([10,3,32,32])
#flops = FlopCountAnalysis(model,x)

print("flops per module----------------------------- ")
#flops_per_module = flops.by_module()
flops_per_module = parameter_count(model)
sorted_flops_per_module = {}
sum = 0
dot_count = 0
for key in flops_per_module.keys():
    for c in key:
        if c == '.':
            dot_count += 1
    if (dot_count == 0 and 'layer' not in key and 'classifier' not in key and 'features' not in key) \
        or dot_count == 1:
        sorted_flops_per_module[key] = flops_per_module[key]
    dot_count = 0

print(sorted_flops_per_module)
for key in sorted_flops_per_module:
    if key != '':
        sum += sorted_flops_per_module[key]
print(sum-9408-128-20490)

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