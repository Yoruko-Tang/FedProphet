
from fvcore.nn import FlopCountAnalysis,flop_count_table
import torch
import argparse
import models
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default=None, 
                    help="To use cuda, set to a specific GPU ID. Default set to use CPU.")

args = parser.parse_args()
device = torch.device('cuda:{}'.format(args.gpu) if args.gpu is not None else 'cpu')

# model = models.get_net('vgg11', 'cifar', False, 10).to(device)
model = models.LeNet5().to(device)
# x = torch.rand([10,3,32,32]).to(device)
# 10 3 32 32 
# model = VGG11(block=0).to(device)
x = torch.rand([10,1,28,28]).to(device)
print(model)
flops = FlopCountAnalysis(model,x)
print(flop_count_table(flops))