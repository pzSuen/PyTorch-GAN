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

import albumentations as A
from albumentations import (OneOf, PadIfNeeded, RandomCrop,
                            HorizontalFlip, VerticalFlip, Transpose, RandomRotate90,
                            ElasticTransform, GridDistortion, OpticalDistortion, CLAHE, RandomBrightnessContrast,
                            ChannelShuffle, Blur, Normalize)


class ImageDataset(Dataset):
    def __init__(self, opt, root, transformer=None):
        self.transformer = transformer
        self.opt = opt

        self.images_dir = os.path.join(root, 'train_image')
        self.masks_dir = os.path.join(root, 'train_mask')
        self.images = sorted(glob.glob(self.images_dir + "/*.*"))
        self.masks = sorted(glob.glob(self.masks_dir + "/*.*"))

    def __getitem__(self, index, show=False):
        fname = os.path.basename(self.images[index % len(self.images)])
        img = np.array(Image.open(os.path.join(self.images_dir, fname)), dtype=np.float64)
        mask = np.array(Image.open(os.path.join(self.masks_dir, fname)), dtype=np.int64)

        # print("-" * 20)
        # print(type(img), type(mask))
        if show:
            print(np.max(np.array(img)), np.max(np.array(mask)))
            plt.subplot(221)
            plt.imshow(img)
            plt.subplot(222)
            plt.imshow(mask)

        augmented = None
        seed = np.random.randint(2147483647)  # make a seed with numpy generator
        random.seed(seed)  # apply this seed to img tranfsorms
        if self.transformer:
            augmented = self.transformer(image=img, mask=mask)

        img = augmented['image']
        mask = augmented['mask']
        if show:
            plt.subplot(223)
            plt.imshow(np.array(img))
            plt.subplot(224)
            plt.imshow(np.array(mask))
            plt.show()

            print(torch.max(img), torch.max(mask))

        # print("............")

        # img = img.transpose(2, 0, 1)
        img = np.transpose(img, (2, 0, 1))
        mask = np.expand_dims(mask, 0)

        print(img.shape, mask.shape)
        # mask = self.preprocess_input(mask)
        mask = self.my_scatter(mask)
        # print(img.shape, mask.shape)
        print("+" * 20)
        print(type(img), type(mask))
        print(img.shape, mask.shape)

        img, mask = torch.from_numpy(img), torch.from_numpy(mask)
        # img = img.permute(2, 0, 1)
        # mask = mask.unsqueeze(0)

        return {"image": img, "mask": mask}

    def __len__(self):
        return len(self.images)

    def preprocess_input(self, mask):
        # move to GPU and change data types
        # mask = mask
        # create one-hot label map
        # 将不同的类别映射到不同的channel上
        # 读取的label map是单通道的，背景是0，类别一是1，类别二是2……
        _, h, w = mask.shape
        nc = self.opt.label_nc  # label_nc是包含背景的
        # input_label = torch.FloatTensor(nc, h, w).zero_()
        input_label = torch.zeros(size=(nc, h, w), dtype=torch.float)
        # input_label = np.zeros(shape=(nc, h, w))
        # 对于该图像不存在的类别的那一个channel会是全0
        print("#" * 20)
        mask = torch.from_numpy(mask)
        print(type(mask))
        input_semantics = input_label.scatter(dim=1, index=mask, src=torch.ones(1))  # (dim,index,src)

        return input_semantics

    def my_scatter(self, mask):
        cs, h, w = mask.shape
        # assert cs == 1, "The channel of the input mask must be 1."
        label_nc = self.opt.label_nc
        # re = torch.zeros(size=(label_nc, h, w))
        re = np.zeros(shape=(label_nc, h, w))

        for j in range(cs):
            for m in range(h):
                for n in range(w):
                    re[mask[j, m, n], m, n] = 1

        return re


if __name__ == "__main__":
    '''
    华为官方内胆包：335*150*24.5
    matebook x pro: 304 * 217* 14.6(放入官方包非常合适)
    magicbook 14: 322.5* 214.8*15.9
    MOFT 13: 340* 240* 7 (最大305*215*17)
    MOFT 13.3: 360* 255 *7(最大325*230*17)
    MOFT 15-16: 395*275*7 
    Macbook Air 13.3: 304*212*16.1
    '''
    input_size = 512
    # Configure dataloaders
    transformer = A.Compose([
        # 非刚体转换
        PadIfNeeded(input_size, input_size),
        RandomCrop(input_size, input_size),
        # 非破坏性转换
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        HorizontalFlip(p=0.5),
        Transpose(p=0.5),
        #
        # Blur(p=0.3),
        # 非刚体转换
        # OneOf([
        #     ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        #     GridDistortion(p=0.5),
        #     OpticalDistortion(p=0.5, distort_limit=0.25, shift_limit=0.25)
        # ], p=0.8),
        # 非空间性转换
        # CLAHE(p=0.8),
        # RandomBrightnessContrast(p=0.8),
        # RandomGamma(p=0.8),
        Normalize(
            mean=[0.67, 0.55, 0.74],
            std=[0.16, 0.20, 0.17],
        )
    ])
    data = ImageDataset('/home/pzsuen/Code/I2I/M2N/datasets/nuclear/', transformer=transformer)

    print(data.__len__())
    for i in range(data.__len__()):
        data.__getitem__(i)
