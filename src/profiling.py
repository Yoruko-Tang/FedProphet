from models.model_utils import get_net
from hardware.sys_utils import model_summary
from scheduler.module_scheduler import module_scheduler
from client import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m","--modelfamily",default="cifar",type=str)
parser.add_argument("-a","--arch",default="vgg16_bn",type = str)
a = parser.parse_args()

if a.modelfamily == 'cifar':
        model = get_net(a.arch,'cifar',num_classes=10,adv_norm=True,modularization=True,norm_type='BN')
        inputsize = [64,3,32,32]
        args = {"epochs":500,
                "reserved_flops":None,
                "reserved_mem":60e6,
                "adv_train":True,
                "adv_epsilon":0,
                "adv_alpha":0,
                "adv_norm":"inf",
                "adv_bound":[0,1],
                "mu":0,
                "lamb":0,
                "psi":1,
                "target_clean_adv_ratio":1.5}
elif a.modelfamily == 'imagenet':
        model = get_net(a.arch,'imagenet',num_classes=256,adv_norm=True,modularization=True,norm_type='BN')
        inputsize = [32,3,224,224]
        args = {"epochs":500,
                "reserved_flops":None,
                "reserved_mem":224e6,
                "adv_train":True,
                "adv_epsilon":0,
                "adv_alpha":0,
                "adv_norm":"inf",
                "adv_bound":[0,1],
                "mu":0,
                "lamb":0,
                "psi":1,
                "target_clean_adv_ratio":1.5}

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


msch = module_scheduler(args,ms,None,None)

if a.modelfamily == 'cifar':
        print("Full Training:",ms.training_latency(performance=1e11,
                                        memory=60e6,
                                        eff_bandwidth=1.5e9,
                                        batches=[[64,3,32,32]]*30,
                                        adv_iters=10))
        for i in range(len(msch.partition_module_list)):
                print("Module%d:"%i,ms.training_latency(performance=1e11,
                                        memory=60e6,
                                        eff_bandwidth=1.5e9,
                                        batches=[[64,3,32,32]]*30,
                                        adv_iters=10,
                                        module_list=msch.module_dict[msch.partition_module_list[i]]))
elif a.modelfamily == 'imagenet':
        print("Full Training: ",ms.training_latency(performance=1e12,
                                        memory=224e6,
                                        eff_bandwidth=1.5e9,
                                        batches=[[32,3,224,224]]*1,
                                        adv_iters=10))
        for i in range(len(msch.partition_module_list)):
                print("Module%d:"%i,ms.training_latency(performance=1e12,
                                        memory=224e6,
                                        eff_bandwidth=1.5e9,
                                        batches=[[32,3,224,224]]*1,
                                        adv_iters=10,
                                        module_list=msch.module_dict[msch.partition_module_list[i]]))