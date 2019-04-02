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

class resFAN(nn.Module):
    '''

    '''
    def __init__(self, bn=False):

        super(resFAN, self).__init__()
        #appearance_stream no pretained layers
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.frontend = []
        self.backend = make_layers(self.backend_feat, in_channels=1024, dilation=True)
        self.convDU = convDU(in_out_channels=64,kernel_size=(1,9))
        self.convLR = convLR(in_out_channels=64,kernel_size=(9,1))
        self.o_layer = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.ReLU())

        #flow_stream no pretained layers
        self.frontend_flow = []
        self.backend_flow = make_layers(self.backend_feat, in_channels=1024, dilation=True)
        self.convDU_flow = convDU(in_out_channels=64,kernel_size=(1,9))
        self.convLR_flow = convLR(in_out_channels=64,kernel_size=(9,1))
        self.o_layer_flow = nn.Sequential(nn.Conv2d(64, 1, kernel_size=1), nn.ReLU())



        #Appearance Attention network
        in_dim=64
        self.appearance_A = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.appearance_B = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.appearance_C = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.appearance_gamma = nn.Parameter(torch.zeros(1))
        self.appearance_softmax = nn.Softmax(dim=-1)


        #Flow Attention network
        self.flow_A = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.flow_B = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.flow_C = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.flow_gamma = nn.Parameter(torch.zeros(1))
        self.flow_softmax = nn.Softmax(dim=-1)


        #initialize
        self._initialize_weights()

        #layers pretrained

        pre_wts2=torch.load(model_path2)
        self.backend.load_state_dict(pre_wts2,strict=False)

        #flow
        self.backend_flow.load_state_dict(pre_wts2,strict=False)





    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.fill_(1)
                m.bias.data.fill_(0)

    def forward(self, in_data):
        splited_feature=torch.split(in_data,[1024,1024],1)
        x=splited_feature[0]
        f=splited_feature[1]

        #visual_feature
        x = self.backend(x)
        f = self.backend_flow(f)

        #x = self.convDU(x)
        #f = self.convDU_flow(f)#

        #x = self.convLR(x)
        #f = self.convLR_flow(f)


        #apply flow attention on appearance
        m_batchsize, C, height, width = x.size()
        proj_query_x = self.appearance_A(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key_x = self.appearance_B(x).view(m_batchsize, -1, width*height)
        energy_x = torch.bmm(proj_query_x, proj_key_x)
        attention_x = self.appearance_softmax(energy_x)
        proj_value_x = self.appearance_C(x).view(m_batchsize, -1, width*height)

        out_x = torch.bmm(proj_value_x, attention_x.permute(0, 2, 1))
        out_x = out_x.view(m_batchsize, C, height, width)

        out_x = self.appearance_gamma*out_x + x

        m_batchsize_f,C_f,height_f,width_f=f.size()
        proj_query_f = self.flow_A(f).view(m_batchsize_f, -1, width_f*height_f).permute(0, 2, 1)
        proj_key_f = self.flow_B(f).view(m_batchsize_f, -1, width_f*height_f)
        energy_f = torch.bmm(proj_query_f, proj_key_f)
        attention_f = self.flow_softmax(energy_f)
        proj_value_f = self.flow_C(f).view(m_batchsize_f, -1, width_f*height_f)

        out_f = torch.bmm(proj_value_f, attention_f.permute(0, 2, 1))
        out_f = out_f.view(m_batchsize_f, C_f, height_f, width_f)

        out_f = self.flow_gamma*out_f + f

        x=self.o_layer(out_x)
        f = self.o_layer_flow(out_f)

        x_den2 = F.upsample(x, scale_factor=8)
        f = F.upsample(f, scale_factor=8)

        return x_den2, f

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



class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from DANet
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out