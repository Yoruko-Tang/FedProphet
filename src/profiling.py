from models.model_utils import get_net
from fvcore.nn import FlopCountAnalysis,flop_count_table, parameter_count
import torch
from hardware.sys_utils import model_summary
from scheduler.module_scheduler import module_scheduler
from client import *
import numpy as np



model = get_net('cnn4','cifar',num_classes=10,adv_norm=True,modularization=False,norm_type='BN')
inputsize = [64,3,32,32]
args = {"epochs":500,
        "reserved_flops":None,
        "reserved_mem":6.4e7,
        "adv_epsilon":0,
        "adv_alpha":0,
        "adv_norm":"inf",
        "adv_bound":[0,1],
        "mu":0,
        "lamb":0,
        "psi":1}

# model = get_net('resnet34','imagenet',num_classes=256,adv_norm=True,modularization=True,norm_type='BN')
# inputsize = [32,3,224,224]
# args = {"epochs":500,
#         "reserved_flops":None,
#         "reserved_mem":192e6,
#         "adv_epsilon":0,
#         "adv_alpha":0,
#         "adv_norm":"inf",
#         "adv_bound":[0,1],
#         "mu":0,
#         "lamb":0,
#         "psi":1}
#ms = model_summary(model,inputsize,optimizer='adam')
ms = model_summary(model,inputsize,optimizer='sgd',momentum=0.9)

#print(ST_Client.get_local_state_dict(model).keys())
# # print(ms.training_latency(1e12,1e9,6.4e10,partial_frac=0.5))
# # print(ms.training_latency(1e12,1e9,6.4e10,partial_frac=1.0))
# print(model)
# neuron_dict={n:np.random.choice(range(model.neuron_num[n]),int(model.neuron_num[n]*0.2),replace=False) for n in model.neuron_num}
# print(neuron_dict)
# x = torch.rand(inputsize)
# model.eval()
# a = model.partial_forward(x,neuron_dict)


# for n,m in model.named_modules():
#     print(n)
#     if hasattr(m,'in_retain_idx'):
#         print("in_features: ",m.in_retain_idx)
#     if hasattr(m,'retain_idx'):
#         print("out_features: ",m.retain_idx)
# b = model(x)
# print(a)
# print(b)

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


msch = module_scheduler(args,ms,None,None)
print(msch.auxiliary_model_dict)