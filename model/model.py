'''
YOLOv4 Object Detection Model
Inspired by the YOLOv4 paper:
(https://arxiv.org/abs/2004.10934v1)
'''

import torch.nn as nn
from typing import List

from model.PANet import ConvBlock, PANet
from model.CSPDarknet import CSPDarknet


class DetectionHead(nn.Module):
    def __init__(self, num_classes, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.num_output = 5 + num_classes

        self.dec_n3 = ConvBlock(in_channels=256,  out_channels=num_anchors * self.num_output, filter_size=1)
        self.dec_n4 = ConvBlock(in_channels=512,  out_channels=num_anchors * self.num_output, filter_size=1)
        self.dec_n5 = ConvBlock(in_channels=1024, out_channels=num_anchors * self.num_output, filter_size=1)

    def forward(self, n3, n4, n5):
        bs, _, h3, w3 = n3.shape
        d3 = self.dec_n3(n3).view(bs, self.num_anchors, h3, w3, self.num_output)

        bs, _, h4, w4 = n4.shape
        d4 = self.dec_n4(n4).view(bs, self.num_anchors, h4, w4, self.num_output)

        bs, _, h5, w5 = n5.shape
        d5 = self.dec_n5(n5).view(bs, self.num_anchors, h5, w5, self.num_output)

        return [d3, d4, d5]


class YOLOv4(nn.Module):
    def __init__(self, num_classes=13, num_anchors=3):
        super().__init__()
        self.backbone = CSPDarknet()
        self.neck     = PANet()
        self.head     = DetectionHead(num_classes=num_classes, num_anchors=num_anchors)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        n3, n4, n5 = self.neck(c3, c4, c5)
        return self.head(n3, n4, n5)