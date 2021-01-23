import torch
import torch.nn as nn

class BatchNorm(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, weight=True, bias=True):
        super().__init__(num_features, eps=eps, momentum=momentum)
        self.weight.data.fill_(1.0)
        self.bias.data.fill_(0.0)
        self.weight.requires_grad = weight
        self.bias.requires_grad = bias

class GhostBatchNorm(BatchNorm):
    def __init__(self, num_features, num_splits, **kw):
        super().__init__(num_features, **kw)
        self.num_splits = num_splits
        self.register_buffer('running_mean', torch.zeros(num_features * self.num_splits))
        self.register_buffer('running_var', torch.ones(num_features * self.num_splits))

    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            self.running_mean = torch.mean(self.running_mean.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
            self.running_var = torch.mean(self.running_var.view(self.num_splits, self.num_features), dim=0).repeat(
                self.num_splits)
        return super().train(mode)

    def forward(self, input):
        N, C, H, W = input.shape
        if self.training or not self.track_running_stats:
            return F.batch_norm(
                input.view(-1, C * self.num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(self.num_splits), self.bias.repeat(self.num_splits),
                True, self.momentum, self.eps).view(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean[:self.num_features], self.running_var[:self.num_features],
                self.weight, self.bias, False, self.momentum, self.eps)


def Batch_Norm_Layer(self,channels,BN_type='BN'):
    '''
        BN_type == 'BN' -> GhostBatchNorm()
        BN_type == 'GBN' -> nn.BatchNorm2d()
        Selects the type of Batch Normalization which is to be used
    '''
    if BN_type == 'GBN':
        return GhostBatchNorm(channels, num_splits=2, weight=False)
    elif BN_type == 'BN':
        return nn.BatchNorm2d(channels)

class Depth_Sep_Conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3,padding=1, bias=False):
        super(Depth_Sep_Conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias = bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias = bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
