import glob
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_dataset_mean_std(images_dir='/home/pzsuen/Code/I2I/M2N/datasets/nuclear/train_image'):
    images_path = sorted(glob.glob(images_dir + "/*.*"))
    print(len(images_path))
    R_means = []
    G_means = []
    B_means = []
    R_stds = []
    G_stds = []
    B_stds = []
    for ip in tqdm(images_path):
        img = Image.open(ip)
        im = np.array(img)
        im_R = im[:, :, 0] / 255
        im_G = im[:, :, 1] / 255
        im_B = im[:, :, 2] / 255
        im_R_mean = np.mean(im_R)
        im_G_mean = np.mean(im_G)
        im_B_mean = np.mean(im_B)
        im_R_std = np.std(im_R)
        im_G_std = np.std(im_G)
        im_B_std = np.std(im_B)
        R_means.append(im_R_mean)
        G_means.append(im_G_mean)
        B_means.append(im_B_mean)
        R_stds.append(im_R_std)
        G_stds.append(im_G_std)
        B_stds.append(im_B_std)
    a = [R_means, G_means, B_means]
    b = [R_stds, G_stds, B_stds]
    mean = [0, 0, 0]
    std = [0, 0, 0]
    mean[0] = np.mean(a[0])
    mean[1] = np.mean(a[1])
    mean[2] = np.mean(a[2])
    std[0] = np.mean(b[0])
    std[1] = np.mean(b[1])
    std[2] = np.mean(b[2])
    print('数据集的RGB平均值为\n[{},{},{}]'.format(mean[0], mean[1], mean[2]))
    print('数据集的RGB方差为\n[{},{},{}]'.format(std[0], std[1], std[2]))
    '''
    数据集的RGB平均值为
    [0.6700539749841021,0.5465126513221624,0.7413476545071912]
    数据集的RGB方差为
    [0.16030703740184896,0.2000897845573496,0.17028963625391302]
    '''


if __name__ == "__main__":
    # compute_dataset_mean_std()
    # import torch.nn as nn
    #
    # dim = 64
    # n_downsample = 4
    # style_dim = 5
    # downmodels = nn.ModuleList()
    # channelmodels = nn.ModuleList()
    # Downsampling
    # for i in range(2):
    #     downmodels.add_module(
    #         "se_%d" % i,
    #         nn.Sequential(
    #             nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
    #             nn.ReLU(inplace=True)
    #         ),
    #     ),
    #     downmodels.add_module(
    #         "cc%d" % i,
    #         nn.Sequential(
    #             nn.Conv2d(dim * 2, style_dim, 1, 1, 0),
    #             nn.ReLU(inplace=True)
    #         )
    #     ),
    #     dim *= 2
    #
    # # Downsampling with constant depth
    # for i in range(2, n_downsample):
    #     downmodels.add_module(
    #         "se_%d" % i,
    #         nn.Sequential(
    #             nn.Conv2d(dim, dim, 4, stride=2, padding=1),
    #             nn.ReLU(inplace=True)
    #         ),
    #     )
    #     downmodels.add_module(
    #         "cc%d" % i,
    #         nn.Sequential(
    #             nn.Conv2d(dim, style_dim, 1, 1, 0),
    #             nn.ReLU(inplace=True)
    #         )
    #     ),
    # import torch.nn as nn
    #
    # nf = 64
    # input_nc = 3
    # wid = 128
    # style = torch.FloatTensor(np.random.random(size=(4, 3, wid, wid))).cuda()
    #
    # norm_layer_nf = nn.InstanceNorm2d(nf, affine=False)
    # m = nn.InstanceNorm2d(nf, affine=True)
    #
    # # bottom-up pathway
    # # width = w
    # enc1 = nn.Sequential(nn.Conv2d(input_nc, nf, kernel_size=4, stride=2, padding=1),
    #                      nn.InstanceNorm2d(nf, affine=True)).cuda()
    # out = enc1(style)
    # print(enc1)
    # print(out.shape)

    # input = torch.randn(20, 100, 35, 45)
    # output = m(input)
    # print(output.shape)

    # bg = torch.zeros(size=(1, 1, 2, 2), dtype=torch.int64)
    # a = torch.ones(size=(1, 1, 2, 2), dtype=torch.int64)
    # b = torch.ones(size=(1, 1, 2, 2), dtype=torch.int64) * 2
    # c = torch.ones(size=(1, 1, 2, 2), dtype=torch.int64) * 3
    # d = torch.cat([bg, a], dim=-1)
    # e = torch.cat([b, c], dim=-1)
    # f = torch.cat([d, e], dim=-2)
    # # d = torch.randint(0, 4, (1, 1, 5, 5))
    # print(f.shape)
    # print(f)
    #
    # label_nc = 4
    # bs, cs, h, w = f.shape
    # re = torch.zeros(size=(bs, label_nc, h, w))
    # for i in range(bs):
    #     for m in range(h):
    #         for n in range(w):
    #             re[i, f[i, 0, m, n], m, n] = 1
    #
    # print("#" * 20)
    # print(re.shape)
    # print(re)
    a = torch.ones(size=(12, 1, 1, 1)) * -1
    b = torch.ones(size=(12, 1, 1, 1)) * -1
    c = torch.ones(size=(12, 1, 1, 1)) * -1
    d = torch.ones(size=(12, 1, 1, 1)) * -1
    e = torch.ones(size=(12, 1, 1, 1)) * -1

    # a = torch.zeros(size=(12, 1, 1, 1))
    # b = torch.zeros(size=(12, 1, 1, 1))
    # c = torch.zeros(size=(12, 1, 1, 1))
    # d = torch.zeros(size=(12, 1, 1, 1))
    # e = torch.zeros(size=(12, 1, 1, 1))

    # a = torch.ones(size=(12, 1, 1, 1))
    # b = torch.ones(size=(12, 1, 1, 1))
    # c = torch.ones(size=(12, 1, 1, 1))
    # d = torch.ones(size=(12, 1, 1, 1))
    # e = torch.ones(size=(12, 1, 1, 1))

    input = [a, b, c, d, e]
    # all_input = torch.cat(input, dim=1)

    num = len(input)
    # avg_input = torch.zeros_like(input[0])
    # print(avg_input)
    minval = torch.zeros_like(input[0])
    for i in range(num):
        # 越浅层的权重越大
        input_i = input[i]
        gt = torch.zeros_like(input_i)
        minval += torch.max(1 - input_i, gt) * 0.5 ** (i + 1)
        # print(minval.shape)
    loss = torch.mean(minval)
    print(loss)
