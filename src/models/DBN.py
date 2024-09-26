import torch
import torch.nn as nn
import torch.nn.functional as F

class DualBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.register_buffer('adv_running_mean',self.running_mean.clone())
        self.register_buffer('adv_running_var',self.running_var.clone())
        self.register_buffer('adv_num_batches_tracked',self.num_batches_tracked.clone())
        self.adv = False

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
            try:
                self.adv_running_mean.zero_()
                self.adv_running_var.fill_(1)
                self.num_batches_tracked.zero_()
            except:
                pass

    def clone_clean_stat(self):
        self.adv_running_mean = self.running_mean.clone()
        self.adv_running_var = self.running_var.clone()

    def forward(self,input:torch.Tensor):
        """
        From Source code of Pytorch 1.7.1
        """
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                if not self.adv:
                    self.num_batches_tracked = self.num_batches_tracked + 1
                else:
                    if self.adv_num_batches_tracked == 0: # reinitialize the adv stat with the current clean stat
                        self.clone_clean_stat()
                    self.adv_num_batches_tracked = self.adv_num_batches_tracked + 1
                    
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked if not self.adv else self.adv_num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            if not self.adv:
                bn_training = (self.running_mean is None) and (self.running_var is None)
            else:
                bn_training = (self.adv_running_mean is None) and (self.adv_running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if not self.adv:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
        else:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.adv_running_mean if not self.training or self.track_running_stats else None,
                self.adv_running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)


class DualBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=0.00001, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.register_buffer('adv_running_mean',self.running_mean.clone())
        self.register_buffer('adv_running_var',self.running_var.clone())
        self.register_buffer('adv_num_batches_tracked',self.num_batches_tracked.clone())
        self.adv = False
    
    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()
            try:
                self.adv_running_mean.zero_()
                self.adv_running_var.fill_(1)
                self.num_batches_tracked.zero_()
            except:
                pass
    
    def clone_clean_stat(self):
        self.adv_running_mean = self.running_mean.clone()
        self.adv_running_var = self.running_var.clone()

    def forward(self,input:torch.Tensor):
        """
        From Source code of Pytorch 1.7.1
        """
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:
                if not self.adv:
                    self.num_batches_tracked = self.num_batches_tracked + 1
                else:
                    if self.adv_num_batches_tracked == 0: # reinitialize the adv stat with the current clean stat
                        self.clone_clean_stat()
                    self.adv_num_batches_tracked = self.adv_num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked if not self.adv else self.adv_num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            if not self.adv:
                bn_training = (self.running_mean is None) and (self.running_var is None)
            else:
                bn_training = (self.adv_running_mean is None) and (self.adv_running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        if not self.adv:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
        else:
            return F.batch_norm(
                input,
                # If buffers are not to be tracked, ensure that they won't be updated
                self.adv_running_mean if not self.training or self.track_running_stats else None,
                self.adv_running_var if not self.training or self.track_running_stats else None,
                self.weight, self.bias, bn_training, exponential_average_factor, self.eps)
        
def adv(self):
    for n,m in self.named_modules():
        if isinstance(m,DualBatchNorm2d) or isinstance(m,DualBatchNorm1d):
            m.adv = True

def clean(self):
    for n,m in self.named_modules():
        if isinstance(m,DualBatchNorm2d) or isinstance(m,DualBatchNorm1d):
            m.adv = False