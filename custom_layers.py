# some codes copied from https://github.com/nashory/pggan-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
import copy
from torch.nn.init import kaiming_normal, calculate_gain
from math import sqrt


class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.eps = 1e-8

    def forward(self, x):
        return x / (torch.mean(x**2, dim=1, keepdim=True) + self.eps) ** 0.5


# for equaliaeed-learning rate.
class EqualizedConv2d(nn.Module):
    def __init__(self, c_in, c_out, k_size, stride, pad):
        super(EqualizedConv2d, self).__init__()
        conv = nn.Conv2d(c_in, c_out, k_size, stride, pad)

        conv.weight.data.normal_()
        conv.bias.data.zero_()

        self.conv = equal_lr(conv)

    def forward(self, x):
        return self.conv(x)


class EqualizedLinear(nn.Module):
    def __init__(self, c_in, c_out):
        super(EqualizedLinear, self).__init__()
        linear = nn.Linear(c_in, c_out)
        
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, x):
        return self.linear(x)


class AdaIn(nn.Module):
    def __init__(self, style_dim, channel):
        super(AdaIn, self).__init__()

        self.channel = channel

        self.instance_norm = nn.InstanceNorm2d(channel)
        self.linear = EqualizedLinear(style_dim, channel * 2)

    def forward(self, x, style):
        mu, sig = self.linear(style).chunk(2, dim=1)

        x = self.instance_norm(x)

        x = x * (mu.view(mu.size(0), -1, 1, 1) + 1) + sig.view(sig.size(0), -1, 1, 1) # affine transform

        return x

class NoiseInjection_(nn.Module):
    def __init__(self, channel):
        super(NoiseInjection_, self).__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, x, noise):
        return x + self.weight * noise

class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super(NoiseInjection, self).__init__()

        injection = NoiseInjection_(channel)
        self.injection = equal_lr(injection)

    def forward(self, x, noise):
        return self.injection(x, noise)

class minibatch_stddev_layer(nn.Module):
    def __init__(self, group_size=4, num_new_features=1):
        super(minibatch_stddev_layer, self).__init__()
        self.group_size = group_size
        self.num_new_features = num_new_features

    def forward(self, x):
        group_size = min(self.group_size, x.size(0))
        origin_shape = x.shape
        
        # split group
        y = x.view(
            group_size,
            -1, 
            self.num_new_features, 
            origin_shape[1] // self.num_new_features, 
            origin_shape[2], 
            origin_shape[3]
        )
        
        # calculate stddev over group
        y = torch.sqrt(torch.mean((y - torch.mean(y, dim=0, keepdim=True)) ** 2, dim=0) + 1e-8)
        # [G, F. C, H, W]
        y = torch.mean(y, dim=[2,3,4], keepdim=True)
        # [G, F, 1, 1, 1]
        y = torch.squeeze(y, dim=2)
        # [G, F, 1, 1] 
        y = y.repeat(group_size, 1, origin_shape[2], origin_shape[3])
        # [B, F, H, W]
        
        return torch.cat([x, y], dim=1)

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module
