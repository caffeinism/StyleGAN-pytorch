import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, const_channel):
        super(Generator, self).__init__()

        self.model = ConstBlock(const_channel)
        self.grow()

    def forward(self, x, style):
        return self.model(x, style)

    def grow(self):
        up = UpBlock()
        self.model = RecursiveBlock(prev_block=self.model, block=up, next_block=None)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = NotImplemented
    
    def forward(self, x):
        return self.model(x)

    def grow(self):
        down = DownBlock()
        self.model = RecursiveBlock(prev_block=None, block=down, next_block=self.model)

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
    def __init__(self):
        super(UpBlock, self).__init__()

    def forward(self, x, style):
        pass

class DownBlock(nn.Module):
    def __init__(self):
        super(DownBlock, self).__init__()

    def forward(self, x):
        pass

class ConstBlock(nn.Module):
    def __init__(self, channel, size=4):
        super(ConstBlock, self).__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, x, style=None):
        out = self.input.repeat(x.size(0), 1, 1, 1)

        return out