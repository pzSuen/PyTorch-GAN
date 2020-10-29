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
    def __init__(self, root, transformer=None):
        self.transformer = transformer

        self.images_dir = os.path.join(root, 'train_image')
        self.masks_dir = os.path.join(root, 'train_mask')
        self.images = sorted(glob.glob(self.images_dir + "/*.*"))
        self.masks = sorted(glob.glob(self.masks_dir + "/*.*"))

    def __getitem__(self, index,show=True):
        fname = os.path.basename(self.images[index % len(self.images)])
        img = np.array(Image.open(os.path.join(self.images_dir, fname)))
        mask = np.array(Image.open(os.path.join(self.masks_dir, fname)))

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

        img, mask = torch.from_numpy(augmented['image']), torch.from_numpy(augmented['mask'])

        if show:
            plt.subplot(223)
            plt.imshow(np.array(img))
            plt.subplot(224)
            plt.imshow(np.array(mask))
            plt.show()

            print(torch.max(img), torch.max(mask))

        return {"image": img, "mask": mask}

    def __len__(self):
        return len(self.images)

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()
            data['instance'] = data['instance'].cuda()
            data['image'] = data['image'].cuda()

        # create one-hot label map
        # 将不同的类别映射到不同的channel上
        label_map = data['label']  # 读取的label map是单通道的，背景是0，类别一是1，类别二是2……
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label else self.opt.label_nc  # label_nc是包含背景的
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        # 对于该图像不存在的类别的那一个channel会是全0
        input_semantics = input_label.scatter_(1, label_map, 1.0)  # (dim,index,src)

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = torch.cat((input_semantics, instance_edge_map), dim=1)

        return input_semantics, data['image']


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
