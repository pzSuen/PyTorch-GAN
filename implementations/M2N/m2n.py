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
from implementations.M2N.losses import *

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="nuclear", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")

parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_downsample", type=int, default=5, help="number downsampling layers in encoder")
parser.add_argument("--n_residual", type=int, default=3, help="number of residual blocks in encoder / decoder")
parser.add_argument("--dim", type=int, default=64, help="number of filters in first encoder layer")
parser.add_argument("--label_nc", type=int, default=5, help="# of input label classes")
parser.add_argument("--validation_split", type=float, default=0.1, help="ratio of validation dataset in all data")
opt = parser.parse_args()
print(opt)

cuda = torch.cuda.is_available()

#################################
#     Configure dataloaders
#################################
input_size = opt.img_height
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

dataset = ImageDataset(opt, '/home/pzsuen/Code/I2I/M2N/datasets/%s/' % opt.dataset_name, transformer=transformer)

shuffle_dataset = True
random_seed = 42

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(opt.validation_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]
# print(len(train_indices), len(val_indices))
# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_dataloader = DataLoader(dataset, batch_size=opt.batch_size, sampler=train_sampler,
                              num_workers=opt.n_cpu, drop_last=True)
val_dataloader = DataLoader(dataset, batch_size=opt.batch_size, sampler=valid_sampler,
                            num_workers=opt.n_cpu, drop_last=True)

print("Train Batch Number: " + str(train_dataloader.__len__()))
print("Validate Batch Number: " + str(val_dataloader.__len__()))

# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

#################################
#     Configure loss
#################################
ganloss = HingeGanLoss()
styleloss = PerceptuaLoss()
contentloss = Dice_loss()

# Loss weights
lambda_style = 2
lambda_cont = 2

#################################
#     Configure module
#################################
# Initialize encoders, generators and discriminators
encoder = Encoder(opt)
decoder = Decoder(opt)
discriminator = EDDiscriminator(opt)

if cuda:
    encoder = encoder.cuda()
    Dec1 = decoder.cuda()
    discriminator = discriminator.cuda()
    ganloss.cuda()
    styleloss.cuda()
    contentloss.cuda()

if opt.epoch != 0:
    # Load pretrained models
    encoder.load_state_dict(torch.load("saved_models/%s/decoder_%d.pth" % (opt.dataset_name, opt.epoch)))
    decoder.load_state_dict(torch.load("saved_models/%s/decoder_%d.pth" % (opt.dataset_name, opt.epoch)))
    discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    encoder.apply(weights_init_normal)
    decoder.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()),
    lr=opt.lr,
    betas=(opt.b1, opt.b2),
)

optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


def sample_images(batches_done, val_dataloader=None):
    """Saves a generated sample from the validation set"""
    # val_dataloader = None
    imgs = next(iter(val_dataloader))
    img_samples = None
    with torch.no_grad():
        for content, mask in zip(imgs["image"], imgs["mask"]):
            # Generate samples
            content_processed, content_code, style_code = encoder(content, style)
            generated = decoder(content_code, style_code[-1])

            img_sample = torch.cat(
                [content.data.cpu().squeeze(0), mask.data.cpu().squeeze(0), generated.data.cpu().squeeze(0)], -1)

            # Concatenate with previous samples vertically
            img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)

        save_image(img_samples, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)


#################################
#          Training
#################################
# Adversarial ground truths
valid = 1
fake = -1

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(train_dataloader):
        # Set model input
        style = batch["image"].cuda()
        mask = batch["mask"].cuda()
        # print("#" * 20)
        # print(style.shape, mask.shape)
        # print(type(style), type(mask))
        # print(style, mask)
        # print(style)
        # print(mask)
        # Encode style
        content_processed, content_code, style_code = encoder(mask, style)
        generated = decoder(content_code, style_code[-1])

        # fake_real = torch.cat((generated, style), dim=0)
        # print(discriminator2)
        # print(discriminator2)
        # print("*" * 10 + "real" + "*" * 10)
        # [feats, segpreds, truefalses] = discriminator2(style)
        # print('...feature...')
        # for i in feats:
        #     print(i.shape)
        # print('...true false...')
        # for j in truefalses:
        #     print(j.shape)
        # print('...seg map...')
        # for k in segpreds:
        #     print(k.shape)

        # print("*" * 10 + "generated" + "*" * 10)
        ref_feats, ref_segpreds, ref_truefalses = discriminator(generated)
        gen_feats, gen_segpreds, gen_truefalses = discriminator(style)
        print("#" * 15 + ' Generated ' + '#' * 15)
        print("ref_feats: ", len(ref_feats))
        for i in range(len(ref_feats)):
            print(ref_feats[i].shape)
        print("ref_segpreds: ", len(ref_segpreds))
        for j in range(len(ref_segpreds)):
            print(ref_segpreds[j].shape)
        print("ref_feats: ", len(ref_feats))
        for k in range(len(ref_feats)):
            print(ref_truefalses[k].shape)

        print("#" * 15 + ' Style ' + '#' * 15)
        print("gen_feats: ", len(gen_feats))
        for i in range(len(gen_feats)):
            print(gen_feats[i].shape)
        print("gen_segpreds:", len(gen_segpreds))
        for j in range(len(gen_segpreds)):
            print(gen_segpreds[j].shape)
        print("gen_feats: ", len(gen_feats))
        for k in range(len(gen_feats)):
            print(gen_truefalses[k].shape)

        # print('...feature...')
        # for i in feats2:
        #     print(i.shape)
        # print('...true false...')
        # for j in truefalses2:
        #     print(j.shape)
        # print('...seg map')
        # for k in segpreds2:
        #     print(k.shape)

        optimizer_G.zero_grad()

        # Losses
        ref_ganloss = ganloss(ref_truefalses, target_is_real=True, is_equality=True)
        gen_ganloss = ganloss(gen_truefalses, target_is_real=False, is_equality=True)

        ref_styleloss = styleloss(style_code[:-1], ref_feats)
        gen_styleloss = styleloss(style_code, gen_feats)

        # ref_contentloss = contentloss(ref_segpreds)
        gen_contentloss = contentloss(gen_segpreds[-1], content_processed)

        # Total loss
        loss_G = ref_ganloss + gen_ganloss + ref_styleloss + gen_styleloss + gen_contentloss

        loss_G.backward()
        optimizer_G.step()

        loss_D = ref_ganloss + ref_styleloss

        loss_D.backward()
        optimizer_D.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(train_dataloader) + i
        batches_left = opt.n_epochs * len(train_dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
            % (epoch, opt.n_epochs, i, len(train_dataloader), loss_D.item(), loss_G.item(), time_left)
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done, val_dataloader)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D.step()

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(encoder.state_dict(), "saved_models/%s/encoder_%d.pth" % (opt.dataset_name, epoch))
        torch.save(decoder.state_dict(), "saved_models/%s/decoder_%d.pth" % (opt.dataset_name, epoch))
        torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))
