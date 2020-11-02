import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import datasets
from torch.autograd import Variable

from implementations.M2N.models import *
from implementations.M2N.datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch


#################################
#     Configure loss
#################################
class HingeGanLoss(nn.Module):
    # 损失的范围为[0, 1.9375]
    def forward(self, input, target_is_real=True, is_equality=True):
        # 输入input是一个list，list中是tensor
        if target_is_real:
            # input的值应该是介于0和1之间，input-1之后在-1到0之间，所以和0比的话选择最小值（真）
            # 对于真来说，值越小置信度越低（从1接近于0），产生的loss越大
            # minval = torch.min(input - 1, get_zero_tensor(input))

            # 新的注释，input的值为-1到1，1为真实，-1为假，所以区间为
            if is_equality:
                all_input = torch.cat(input, dim=1)
                avg_input = torch.sum(all_input, dim=1) / len(input)
                gt = torch.zeros_like(avg_input)
                minval = torch.max(1 - avg_input, gt)
                loss = torch.mean(minval)
            else:
                num = len(input)
                minval = torch.zeros_like(input[0])
                for i in range(num):
                    # 越浅层的权重越大
                    input_i = input[i]
                    gt = torch.zeros_like(input_i)
                    minval += torch.max(1 - input_i, gt) * 0.5 ** (i + 1)
                    # print(minval.shape)
                loss = torch.mean(minval)


        else:
            # input的值应该是介于-1和0之间，-input-1之后在-1到0之间，所以和0比的话选择最小值(假)
            # 对于假来说，值的置信度越低（从-1接近于0），产生的loss越大
            if is_equality:
                all_input = torch.cat(input, dim=1)
                avg_input = torch.sum(all_input, dim=1) / len(input)
                gt = torch.zeros_like(avg_input)
                minval = torch.max(avg_input + 1, gt)
                loss = torch.mean(minval)
            else:
                num = len(input)
                minval = torch.zeros_like(input[0])
                for i in range(num):
                    # 越浅层的权重越大
                    input_i = input[i]
                    gt = torch.zeros_like(input_i)
                    minval += torch.max(input_i + 1, gt) * 0.5 ** (i + 1)
                    # print(minval.shape)
                loss = torch.mean(minval)

        return loss


# todo: 为什么这么计算？有什么用？
# 如果输入的两个参数都是标量： -0.5 * (1 + logvar - mu ** 2 + 2.7183 ** logvar)
# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def forward(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, vgg_path=None):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19(vgg_path).cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class PerceptuaLoss(nn.Module):
    def __init__(self):
        super(PerceptuaLoss, self).__init__()
        self.criterion = nn.L1Loss()
        # self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        # x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        # num = len(x)
        loss = 0
        for i in range(len(x)):
            loss += 2 ** (len(x) - i) * self.criterion(x[i], y[i])
        return loss


class Dice_loss(nn.Module):
    def forward(self, prediction, target):
        """Calculating the dice loss
        Args:
            prediction = predicted image
            target = Targeted image
        Output:
            dice_loss"""

        smooth = 1.0

        i_flat = prediction.view(-1)
        t_flat = target.view(-1)

        intersection = (i_flat * t_flat).sum()

        return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))
