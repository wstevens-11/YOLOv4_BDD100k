'''
CSPDarknet Implementation
Inspired by the CSPDarknet Architecture mentioned in the YOLOv4 paper and inspired by the CSPNet paper:
(https://arxiv.org/abs/2004.10934v1, https://arxiv.org/abs/1608.06993)
'''


import torch
import torch.nn as nn

class Mish(torch.nn.Module):
    def __init__(self):
        '''
        Initialize Mish object
        - Obtain parent class's information, functions, and fields using super().__init__() of nn.Module
        '''
        super(Mish, self).__init__()

    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, filter_size=3, stride=1):
        '''
        Initialize ConvBlock object
        - Obtain parent class's information, functions, and fields using super().__init__() of nn.Module
        - Define function calls using the object initialization parameters
        '''
        super(ConvBlock, self).__init__()
        padding=((filter_size-1)//2)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=filter_size, padding=padding, stride=stride)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.mish = Mish()

    def forward(self, x):
        '''
        Override of forward pass inherited from parent class nn.Module
        - Use function calls defined in object initialization to carry out desired forward pass
        '''
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.mish(x)
        return x

    
    
    
class SPP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=5,  stride=1, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=9,  stride=1, padding=4)
        self.pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.compress = ConvBlock(in_channels * 4, in_channels, filter_size=1)

    def forward(self, x):
        p1 = self.pool1(x)
        p2 = self.pool2(x)
        p3 = self.pool3(x)
        return self.compress(torch.cat([x, p1, p2, p3], dim=1))
    
    
    
    
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        '''
        Author: William Stevens
        Initialize ResBlock object
        - Obtain parent class's information, functions, and fields using super().__init__() of nn.Module
        - Define variables and function calls using the object initialization parameters
        Args:
            conv1x1 (tuple): 
            conv3x3 (tuple): 
        '''
        super().__init__()
        self.firstConvBlock = ConvBlock(in_ch, out_ch, filter_size=1)
        self.secondConvBlock = ConvBlock(out_ch, in_ch, filter_size=3)

    def forward(self, x):
        '''
        Override of forward pass inherited from parent class nn.Module
        - Use function calls defined in object initialization to carry out desired forward pass
        '''
        input = x
        x = self.firstConvBlock(x)
        x = self.secondConvBlock(x)
        return x + input

class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repeats):
        '''
        Initialize CSPBlock object
        - Obtain parent class's information, functions, and fields using super().__init__() of nn.Module
        - Define variables and function calls using the object initialization parameters
        '''
        super(CSPBlock, self).__init__()
        self.part1_conv1 = ConvBlock(in_channels, in_channels//2, filter_size=1)
        self.part2_conv1 = ConvBlock(in_channels, in_channels//2, filter_size=1)

        self.bottleneck = nn.Sequential(*[ResBlock(in_channels//2, in_channels) for _ in range(repeats)])

        self.part2_conv2 = ConvBlock(in_channels, out_channels, filter_size=1)

    def forward(self, x):
        '''
        Override of forward pass inherited from parent class nn.Module
        - Use function calls defined in object initialization to carry out desired forward pass
        - For each convolution, save the output to be procedurally concatenated with subsequent input layers
        '''
        part1 = self.part1_conv1(x)
        part2 = self.part2_conv1(x)
        part2 = self.bottleneck(part2)
        concat = torch.cat((part1, part2), dim=1)

        return self.part2_conv2(concat)

class CSPDarknet(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        '''
        Initialize CSPDarknet object
        - Obtain parent class's information, functions, and fields using super().__init__() of nn.Module
        - Initialize config variable using the CSPDarknet_config defined above
        '''
        super(CSPDarknet, self).__init__()
        self.spp = SPP(in_channels=out_channels * 32)
        self.stem = ConvBlock(in_channels, out_channels, filter_size=3)
        self.c1 = self._build_stage_layer(out_channels, out_channels*2, 1)
        self.c2 = self._build_stage_layer(out_channels*2, out_channels*4, 2)
        self.c3 = self._build_stage_layer(out_channels*4, out_channels*8, 8)
        self.c4 = self._build_stage_layer(out_channels*8, out_channels*16, 8)
        self.c5 = self._build_stage_layer(out_channels*16, out_channels*32, 4)
    
    def _build_stage_layer(self, in_channels, out_channels, repeats):
        return nn.Sequential(ConvBlock(in_channels, out_channels, filter_size=3, stride=2), CSPBlock(out_channels, out_channels, repeats))

    def forward(self, x):
        x = self.stem(x)
        c1 = self.c1(x)
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        c5 = self.spp(self.c5(c4))

        return c3, c4, c5
