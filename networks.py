import torch.nn as nn
from custom_layers import EqualizedConv2d, EqualizedLinear, AdaIn, minibatch_std_concat_layer

class Generator(nn.Module):
    def __init__(self, channels, style_dim):
        super(Generator, self).__init__()

        self.model = UpBlock(channels[0], channels[1], style_dim, initial=True)
        
        self.style_dim = style_dim
        self.now_growth = 1
        self.channels = channels

    def forward(self, x, style):
        return self.model(x, style)

    def grow(self):
        in_c, out_c = self.channels[self.now_growth], self.channels[self.now_growth+1] 
        up = UpBlock(in_c, out_c, self.style_dim)
        self.model = RecursiveBlock(prev_block=self.model, block=up, next_block=None)
        self.now_growth += 1


class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()

        self.model = DownBlock(channels[0], channels[1], initial=True)
        self.now_growth = 1
        self.channels = channels

    def forward(self, x):
        return self.model(x)

    def grow(self):
        in_c, out_c = self.channels[self.now_growth], self.channels[self.now_growth+1] 
        down = DownBlock(in_c, out_c)
        self.model = RecursiveBlock(prev_block=None, block=down, next_block=self.model)
        self.now_growth += 1


class RecursiveBlock(nn.Module):
    def __init__(self, prev_block, block, next_block):
        super(RecursiveBlock, self).__init__()

        self.prev_block = prev_block
        self.block = block
        self.next_block = next_block

    def forward(self, x):
        if self.prev_block:
            x = self.prev_block(x)

        x = self.block(x)

        if self.next_block:
            x = self.next_block(x)

        return x

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, initial=False):
        super(UpBlock, self).__init__()

        if initial:
            self.input = nn.Parameter(torch.randn(1, out_channel, 4, 4))

        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            self.conv1 = EqualizedConv2d(in_channel, out_channel, 3, 1, 1)

        self.adain1 = AdaIn(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualizedConv2d(out_channel, out_channel, 3, 1, 1)
        self.adain2 = AdaIn(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

        self.initial = initial

    def forward(self, x, style):
        if self.initial:        
            x = self.input.repeat(x.size(0), 1, 1, 1)

        else:
            x = self.upsample(x)
            x = self.conv1(x)

        x = self.adain1(x, style)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.adain2(x, style)
        x = self.lrelu2(x)


class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, initial=False):
        super(DownBlock, self).__init__()

        if initial:
            self.minibatch_std = minibatch_std_concat_layer()

            self.conv1 = EqualizedConv2d(in_channel + 1, out_channel, 3, 1, 1)
            self.conv2 = EqualizedConv2d(out_channel, out_channel, 4, 1, 0)

            self.linear = EqualizedLinear(out_channel, 1)
        else:
            self.conv1 = EqualizedConv2d(in_channel, out_channel, 3, 1, 1)
            self.conv2 = EqualizedConv2d(out_channel, out_channel, 3, 1, 1)

            self.downsample = nn.AvgPool2d(2, 2)

        self.lrelu1 = nn.LeakyReLU(0.2)
        self.lrelu2 = nn.LeakyReLU(0.2)

        self.initial = initial

    def forward(self, x):
        if self.initial:
            x = self.minibatch_std(x)

        x = self.conv1(x)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.lrelu2(x)

        if self.initial:
            x = x.view(x.size(0), -1)
            x = self.linear(x)
        else:
            x = self.downsample(x)
        
        return x