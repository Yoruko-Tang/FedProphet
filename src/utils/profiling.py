from models import get_net
from fvcore.nn import FlopCountAnalysis,flop_count_table, parameter_count
import torch


model = get_net('resnet50','cifar',num_classes=10)
print(model)
x = torch.rand([10,3,32,32])
flops = FlopCountAnalysis(model,x)
print(flops.by_module())
print(parameter_count(model))