# https://github.com/nashory/pggan-pytorch

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
        style = self.linear(style).view(2, -1, self.channel, 1, 1)

        x = self.instance_norm(x)

        x = (x * (style[0] + 1)) + style[1] # affine transform

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

# https://github.com/github-pengge/PyTorch-progressive_growing_of_gans/blob/master/models/base_model.py
class minibatch_std_concat_layer(nn.Module):
    def __init__(self, averaging='all'):
        super(minibatch_std_concat_layer, self).__init__()
        self.averaging = averaging.lower()
        if 'group' in self.averaging:
            self.n = int(self.averaging[5:])
        else:
            assert self.averaging in ['all', 'flat', 'spatial', 'none', 'gpool'], 'Invalid averaging mode'%self.averaging
        self.adjusted_std = lambda x, **kwargs: torch.sqrt(torch.mean((x - torch.mean(x, **kwargs)) ** 2, **kwargs) + 1e-8)

    def forward(self, x):
        shape = list(x.size())
        target_shape = copy.deepcopy(shape)
        vals = self.adjusted_std(x, dim=0, keepdim=True)
        if self.averaging == 'all':
            target_shape[1] = 1
            vals = torch.mean(vals, dim=1, keepdim=True)
        elif self.averaging == 'spatial':
            if len(shape) == 4:
                vals = torch.mean(vals, axis=[2,3], keepdim=True)             # torch.mean(torch.mean(vals, 2, keepdim=True), 3, keepdim=True)
        elif self.averaging == 'none':
            target_shape = [target_shape[0]] + [s for s in target_shape[1:]]
        elif self.averaging == 'gpool':
            if len(shape) == 4:
                vals = torch.mean(x, [0,2,3], keepdim=True)                   # torch.mean(torch.mean(torch.mean(x, 2, keepdim=True), 3, keepdim=True), 0, keepdim=True)
        elif self.averaging == 'flat':
            target_shape[1] = 1
            vals = torch.FloatTensor([self.adjusted_std(x)])
        else:                                                           # self.averaging == 'group'
            target_shape[1] = self.n
            vals = vals.view(self.n, self.shape[1]/self.n, self.shape[2], self.shape[3])
            vals = mean(vals, axis=0, keepdim=True).view(1, self.n, 1, 1)
        vals = vals.expand(*target_shape)
        return torch.cat([x, vals], 1)

    def __repr__(self):
        return self.__class__.__name__ + '(averaging = %s)' % (self.averaging)

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