import torch
import torch.nn as nn
from custom_layers import EqualizedConv2d, EqualizedLinear, AdaIn, minibatch_std_concat_layer

class Generator(nn.Module):
    def __init__(self, channels, style_dim):
        super(Generator, self).__init__()
        
        self.style_dim = style_dim
        self.now_growth = 1
        self.channels = channels

        self.model = UpBlock(channels[0], channels[1], style_dim, prev=None)

    def forward(self, style, alpha):
        x, rgb = self.model(x=None, style=style, alpha=alpha)
        return rgb

    def grow(self):
        in_c, out_c = self.channels[self.now_growth], self.channels[self.now_growth+1] 
        self.model = UpBlock(in_c, out_c, self.style_dim, prev=self.model)
        self.now_growth += 1


class Discriminator(nn.Module):
    def __init__(self, channels):
        super(Discriminator, self).__init__()

        self.now_growth = 1
        self.channels = channels

        self.model = DownBlock(channels[1], channels[0], next=None)

    def forward(self, x, alpha):
        return self.model(x=x, alpha=alpha)

    def grow(self):
        in_c, out_c = self.channels[self.now_growth+1], self.channels[self.now_growth] 
        self.model = DownBlock(in_c, out_c, next=self.model)
        self.now_growth += 1


class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim, prev=None):
        super(UpBlock, self).__init__()

        self.prev = prev

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        if prev:
            self.conv1 = EqualizedConv2d(in_channel, out_channel, 3, 1, 1)
        else:
            self.input = nn.Parameter(torch.randn(1, out_channel, 4, 4))

        self.adain1 = AdaIn(style_dim, out_channel)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualizedConv2d(out_channel, out_channel, 3, 1, 1)
        self.adain2 = AdaIn(style_dim, out_channel)
        self.lrelu2 = nn.LeakyReLU(0.2)

        self.to_rgb = EqualizedConv2d(out_channel, 3, 1, 1, 0)

    def forward(self, x, style, alpha=-1.0):
        if self.prev: # if module has prev, then forward first.
            x, prev_rgb = self.prev(x, style, 1 if 0.0 <= alpha < 1.0 else -1.0)

            x = self.upsample(x)
            x = self.conv1(x)
        else: # else initial constant
            x = self.input.repeat(style.size(0), 1, 1, 1)

        x = self.adain1(x, style)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.adain2(x, style)
        x = self.lrelu2(x)

        if 0.0 <= alpha < 1.0:
            prev_rgb = self.upsample(prev_rgb)
            rgb = alpha * self.to_rgb(x) + (1 - alpha) * prev_rgb
        elif alpha == 1:
            rgb = self.to_rgb(x)
        else:
            rgb = None

        return x, rgb


class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, next=None):
        super(DownBlock, self).__init__()

        self.next = next

        self.downsample = nn.AvgPool2d(2, 2)

        if next:
            self.conv1 = EqualizedConv2d(in_channel, out_channel, 3, 1, 1)
            self.conv2 = EqualizedConv2d(out_channel, out_channel, 3, 1, 1)
        else:
            self.minibatch_std = minibatch_std_concat_layer()

            self.conv1 = EqualizedConv2d(in_channel + 1, out_channel, 3, 1, 1)
            self.conv2 = EqualizedConv2d(out_channel, out_channel, 4, 1, 0)

            self.linear = EqualizedLinear(out_channel, 1)

        self.lrelu1 = nn.LeakyReLU(0.2)
        self.lrelu2 = nn.LeakyReLU(0.2)

        self.from_rgb = EqualizedConv2d(3, in_channel, 1, 1, 0)

    def forward(self, x, alpha=-1.0):
        input = x

        if 0 <= alpha:
            x = self.from_rgb(x)

        if not self.next:
            x = self.minibatch_std(x)

        x = self.conv1(x)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.lrelu2(x)

        if self.next:
            x = self.downsample(x)

            if 0.0 <= alpha < 1.0:
                input = self.downsample(input)
                x = alpha * x + (1 - alpha) * self.next.from_rgb(input)

            x = self.next(x)
        else:
            x = x.view(x.size(0), -1)
            x = self.linear(x)
        
        return x