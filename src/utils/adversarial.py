import torch
import torch.nn.functional as F
import numpy as np
from autoattack import AutoAttack

class Adv_Sample_Generator():
    def __init__(self, criterion=lambda m,i,y:F.cross_entropy(m(i),y), 
                 attack_method='PGD',
                 epsilon=8/255, alpha=2/255,T=10, 
                 norm='inf', bound=[0.0,1.0]):
        """
        An adversarial sample generator. Args:
        criterion: should be a function of (model,input,target)
        attack_method: PGD, FGSM, FGSM_RS, or AutoAttack
        epsilon: norm bound of the perturbation
        alpha: step size
        T: number of iterations
        norm: type of norm, inf or l2
        bound: the boundary of valid value of the input
        """
        self.criterion = criterion
        self.attack_method = attack_method
        self.epsilon = epsilon
        self.alpha = alpha
        self.T = T
        self.norm = norm
        self.bound = bound     
        

    def proj(self,pert):
        """
        project the perturbation back to the norm ball
        """
        if self.norm == 'inf':
            return torch.clip(pert,-self.epsilon,self.epsilon)
        else: # l2 norm
            return pert.renorm(p=2,dim=0,maxnorm=self.epsilon)
            



    def attack_data(self,model,data,label,ratio=1.0):
        if ratio < 1.0:
            origin_data = data.clone()
            perturb_idx = np.random.choice(range(len(data)),int(ratio*len(data)),replace = False)
            data = data[perturb_idx]
            label = label[perturb_idx]
        
        if self.attack_method == 'AutoAttack':
            adversary = AutoAttack(model, norm=self.norm, eps=self.epsilon, version='standard')
            x_adv = adversary.run_standard_evaluation(data, label, bs=len(data))
            if ratio < 1.0:
                origin_data[perturb_idx] = x_adv
                return origin_data
            else:
                return x_adv

        

        if self.attack_method in ['PGD','FGSM_RS']:# randomly start
            if self.norm == 'inf':
                delta = ((torch.rand(data.shape)*2-1)*self.epsilon).to(data)
            elif self.norm == 'l2':
                direction = torch.randn_like(data)
                direction = direction/(torch.norm(direction,p=2,dim=list(range(1,direction.dim())),keepdim=True)+1e-10)
                radius = torch.rand([data.shape[0]]+[1]*(data.dim()-1)).to(data)*self.epsilon
                delta = radius*direction
            delta = torch.clip(data+delta,self.bound[0],self.bound[1])-data
            
        else:
            delta = torch.zeros_like(data)

        delta.requires_grad = True
        model.eval()
        for _ in range(self.T):
            
            loss = self.criterion(model,data+delta,label)
            grad = torch.autograd.grad(loss,delta)[0].detach()
            if self.norm == 'inf':
                step = self.alpha*torch.sign(grad)
                
            elif self.norm == 'l2':
                step = self.alpha*grad/(torch.norm(grad,p=2,dim=list(range(1,grad.dim())),keepdim=True)+1e-10)
                
            delta.data = self.proj(delta.data+step)
            delta.data = torch.clip(data+delta.data,self.bound[0],self.bound[1])-data
            
        delta = delta.detach()
        if ratio<1.0:
            mask = torch.zeros_like(origin_data)
            mask[perturb_idx] = delta
            return origin_data + mask
        else:
            return data + delta



