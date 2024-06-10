#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 11:00:29 2022

@author: francesco
"""

import torch.nn as nn
import torchvision.models as models


class FeatureExtractorVGG(nn.Module):
    def __init__(self, model_name, cut_index=None):
        super().__init__()
        
        if model_name == 'Vgg16':
            net = models.vgg16(weights = models.VGG16_Weights.DEFAULT)
        elif model_name == 'Vgg19':
            net = models.vgg19(weights = models.VGG19_Weights.DEFAULT)
        
        if cut_index is not None:
            for i in range(cut_index,len(net.classifier)):
               net.classifier[i] = nn.Identity()
                
        self.fe = net

    def forward(self, x):
        x = self.fe(x)
                
        return x


class FeatureExtractorResnet152(nn.Module):
    def __init__(self, model_name, cut_index=None):
        super().__init__()

        net = models.resnet152(pretrained = True)
        net.fc = nn.Identity()

        self.fe = net

    def forward(self, x):
        x = self.fe(x)

        return x



class FeatureExtractorDENSENET169(nn.Module):
    def __init__(self, model_name, cut_index=None):
        super().__init__()

        net = models.densenet169(pretrained = True)
        net.classifier = nn.Identity()

        self.fe = net

    def forward(self, x):
        x = self.fe(x)

        return x


class FeatureExtractorEFFICIENTNETB7(nn.Module):
    def __init__(self, model_name, cut_index=None):
        super().__init__()

        net = models.efficientnet_b7(pretrained = True)
        net.classifier[0] = nn.Identity()
        net.classifier[1] = nn.Identity()

        self.fe = net

    def forward(self, x):
        x = self.fe(x)

        return x
