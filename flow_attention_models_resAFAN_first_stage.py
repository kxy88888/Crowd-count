#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
from torchvision import models

import torch.nn.functional as F
from src.network import Conv2d, FC
from src.layer import convDU,convLR
model_path='../resnet101-5d3b4d8f.pth'
model_path2='../Pretrained_GCC.pth'

class resFAN_first_stage(nn.Module):
    '''

    '''
    def __init__(self, bn=False):

        super(resFAN_first_stage, self).__init__()
        #layers pretrained
        res = models.resnet101()
        pre_wts = torch.load(model_path)
        res.load_state_dict(pre_wts)
        #appearance
        self.frontend = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        )
        self.own_reslayer_3 = make_res_layer(Bottleneck, 256, 23, stride=1)
        self.own_reslayer_3.load_state_dict(res.layer3.state_dict())

        #flow
        self.frontend_flow = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1, res.layer2
        )
        self.own_reslayer_3_flow = make_res_layer(Bottleneck, 256, 23, stride=1)
        self.own_reslayer_3_flow.load_state_dict(res.layer3.state_dict())


    def forward(self, in_data):
        splited_feature=torch.split(in_data,[3,1],1)
        im_data=splited_feature[0]
        flow=splited_feature[1]
        flow3=torch.cat((flow,flow,flow),1)
        #visual_feature
        x = self.frontend(im_data)
        f = self.frontend_flow(flow3)

        x = self.own_reslayer_3(x)
        f = self.own_reslayer_3_flow(f)

        return x, f

def make_res_layer(block, planes, blocks, stride=1):

    downsample = None
    inplanes=512
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * block.expansion,
                      kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * block.expansion),
        )

    layers = []
    layers.append(block(inplanes, planes, stride, downsample))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(block(inplanes, planes))

    return nn.Sequential(*layers)


def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
    if dilation:
        d_rate = 2
    else:
        d_rate = 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

