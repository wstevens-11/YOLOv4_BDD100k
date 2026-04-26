'''
PANet Implementation
Inspired by the Path Aggregation Network mentioned in the YOLOv4 paper and designed in the PANet paper:
(https://arxiv.org/abs/2004.10934v1, https://arxiv.org/pdf/1803.01534.pdf)
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.CSPDarknet import ConvBlock, CSPBlock

class PANet(nn.Module):
    '''
    Path Aggregation Network: an augmentation of Feature Pyramid Network
    '''
    def __init__(self):
        '''
        Initialize PANet object: Augmentation of FPN with additional Bottom-Up shortcut
        - Obtain parent class's information, functions, and fields using super().__init__() of nn.Module
        Define function calls to be used in forward pass:
        - Define upsampling operation (Mode: Bilinear)

        - Define P5: 1x1 convolution of C5
        - Define M4: CSPBlock of P5: before up-sampling and element-wise addition with C4 to get M4
        - Define P4: 1x1 convolution of M4
        - Define M3: CSPBlock of P4: before up-sampling and element-wise addition with C3 to get P3
        
        - Define N3 and A4: 1x1 convolution of P3 followed by A4 (CSPBlock of N3) to get N4
        - Define N4 and A5: 1x1 convolution of N4 followed by A5 (CSPBlock of N4) to get N5
        '''
        super().__init__()
        
        self.conv_p5 = ConvBlock(in_channels=1024, out_channels=512, filter_size=1)
        self.conv_m4 = CSPBlock(in_channels=1024, out_channels=512, repeats=3)

        self.conv_p4 = ConvBlock(in_channels=512, out_channels=256, filter_size=1)
        self.conv_m3 = CSPBlock(in_channels=512, out_channels=256, repeats=3)

        self.conv_n3 = ConvBlock(in_channels=256, out_channels=256, filter_size=3, stride=2)
        self.conv_a4 = CSPBlock(in_channels=512, out_channels=512, repeats=3)

        self.conv_n4 = ConvBlock(in_channels=512, out_channels=512, filter_size=3, stride=2)
        self.conv_a5 = CSPBlock(in_channels=1024, out_channels=1024, repeats=3)


    def forward(self, c3, c4, c5):
        '''
        Forward pass:
        Override of forward pass inherited from parent class nn.Module
        Use function calls defined in object initialization to carry out desired forward pass:
        - 1x1 convolution of C5 to get P5
        - Up-sample P5, element-wise add with C4 to get P4
        - Up-sample P4, element-wise add with C3 to get P3
        - 1x1 convolution of P3 to get N3
        - Down-sample N3, element-wise add with P4 to get N4
        - Down-sample N4, element-wise add with P5 to get N5
        - 1x1 convolution of N5

        Return N3, N4, N5 as the object detection maps at three scales 
        '''
        p5 = self.conv_p5(c5)
        m4 = torch.cat((F.interpolate(p5, size=c4.shape[2:], mode='bilinear', align_corners=False), c4), dim=1)
        m4 = self.conv_m4(m4)
        p4 = self.conv_p4(m4)
        m3 = torch.cat((F.interpolate(p4, size=c3.shape[2:], mode='bilinear', align_corners=False), c3), dim=1)
        _n3 = self.conv_m3(m3)
        n3 = self.conv_n3(_n3)
        a4 = torch.cat((n3, p4), dim=1)
        _n4 = self.conv_a4(a4)
        n4 = self.conv_n4(_n4)
        a5 = torch.cat((n4, p5), dim=1)
        _n5 = self.conv_a5(a5)
        return _n3, _n4, _n5