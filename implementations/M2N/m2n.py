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
parser.add_argument("--img_height", type=int, default=512, help="size of image height")
parser.add_argument("--img_width", type=int, default=512, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval saving generator samples")
parser.add_argument("--checkpoint_interval", type=int, default=-1, help="interval between saving model checkpoints")
parser.add_argument("--n_downsample", type=int, default=4, help="number downsampling layers in encoder")
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

dataset = ImageDataset('/home/pzsuen/Code/I2I/M2N/datasets/%s/' % opt.dataset_name, transformer=transformer)

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

print(train_dataloader.__len__())
print(val_dataloader.__len__())


# Create sample and checkpoint directories
os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

criterion_recon = torch.nn.L1Loss()

'''
# Initialize encoders, generators and discriminators
Enc1 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Dec1 = Decoder(dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Enc2 = Encoder(dim=opt.dim, n_downsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
Dec2 = Decoder(dim=opt.dim, n_upsample=opt.n_downsample, n_residual=opt.n_residual, style_dim=opt.style_dim)
D1 = MultiDiscriminator()
D2 = MultiDiscriminator()

if cuda:
    Enc1 = Enc1.cuda()
    Dec1 = Dec1.cuda()
    Enc2 = Enc2.cuda()
    Dec2 = Dec2.cuda()
    D1 = D1.cuda()
    D2 = D2.cuda()
    criterion_recon.cuda()

if opt.epoch != 0:
    # Load pretrained models
    Enc1.load_state_dict(torch.load("saved_models/%s/Enc1_%d.pth" % (opt.dataset_name, opt.epoch)))
    Dec1.load_state_dict(torch.load("saved_models/%s/Dec1_%d.pth" % (opt.dataset_name, opt.epoch)))
    Enc2.load_state_dict(torch.load("saved_models/%s/Enc2_%d.pth" % (opt.dataset_name, opt.epoch)))
    Dec2.load_state_dict(torch.load("saved_models/%s/Dec2_%d.pth" % (opt.dataset_name, opt.epoch)))
    D1.load_state_dict(torch.load("saved_models/%s/D1_%d.pth" % (opt.dataset_name, opt.epoch)))
    D2.load_state_dict(torch.load("saved_models/%s/D2_%d.pth" % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    Enc1.apply(weights_init_normal)
    Dec1.apply(weights_init_normal)
    Enc2.apply(weights_init_normal)
    Dec2.apply(weights_init_normal)
    D1.apply(weights_init_normal)
    D2.apply(weights_init_normal)

# Loss weights
lambda_gan = 1
lambda_id = 10
lambda_style = 1
lambda_cont = 1
lambda_cyc = 0

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters()),
    lr=opt.lr,
    betas=(opt.b1, opt.b2),
)
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D1, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)
lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D2, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor


def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    val_dataloader = None
    imgs = next(iter(val_dataloader))
    img_samples = None
    for img1, img2 in zip(imgs["A"], imgs["B"]):
        # Create copies of image
        X1 = img1.unsqueeze(0).repeat(opt.style_dim, 1, 1, 1)
        X1 = Variable(X1.type(Tensor))
        # Get random style codes
        s_code = np.random.uniform(-1, 1, (opt.style_dim, opt.style_dim))
        s_code = Variable(Tensor(s_code))
        # Generate samples
        c_code_1, _ = Enc1(X1)
        X12 = Dec2(c_code_1, s_code)
        # Concatenate samples horisontally
        X12 = torch.cat([x for x in X12.data.cpu()], -1)
        img_sample = torch.cat((img1, X12), -1).unsqueeze(0)
        # Concatenate with previous samples vertically
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
    save_image(img_samples, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)


#################################
#          Training
#################################
# Adversarial ground truths
valid = 1
fake = 0

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(train_dataloader):

        # Set model input
        ref_img = batch["img"].to(cuda)
        mask = batch["mask"].to(cuda)

        # Encode style

        X1 = None
        X2 = None
        style_1 = None
        style_2 = None
        # -------------------------------
        #  Train Encoders and Generators
        # -------------------------------

        optimizer_G.zero_grad()

        # Get shared latent representation
        c_code_1, s_code_1 = Enc1(X1)
        c_code_2, s_code_2 = Enc2(X2)

        # Reconstruct images
        X11 = Dec1(c_code_1, s_code_1)
        X22 = Dec2(c_code_2, s_code_2)

        # Translate images
        X21 = Dec1(c_code_2, style_1)
        X12 = Dec2(c_code_1, style_2)

        # Cycle translation
        c_code_21, s_code_21 = Enc1(X21)
        c_code_12, s_code_12 = Enc2(X12)
        X121 = Dec1(c_code_12, s_code_1) if lambda_cyc > 0 else 0
        X212 = Dec2(c_code_21, s_code_2) if lambda_cyc > 0 else 0

        # Losses
        loss_GAN_1 = lambda_gan * D1.compute_loss(X21, valid)
        loss_GAN_2 = lambda_gan * D2.compute_loss(X12, valid)
        loss_ID_1 = lambda_id * criterion_recon(X11, X1)
        loss_ID_2 = lambda_id * criterion_recon(X22, X2)
        loss_s_1 = lambda_style * criterion_recon(s_code_21, style_1)
        loss_s_2 = lambda_style * criterion_recon(s_code_12, style_2)
        loss_c_1 = lambda_cont * criterion_recon(c_code_12, c_code_1.detach())
        loss_c_2 = lambda_cont * criterion_recon(c_code_21, c_code_2.detach())
        loss_cyc_1 = lambda_cyc * criterion_recon(X121, X1) if lambda_cyc > 0 else 0
        loss_cyc_2 = lambda_cyc * criterion_recon(X212, X2) if lambda_cyc > 0 else 0

        # Total loss
        loss_G = (
                loss_GAN_1
                + loss_GAN_2
                + loss_ID_1
                + loss_ID_2
                + loss_s_1
                + loss_s_2
                + loss_c_1
                + loss_c_2
                + loss_cyc_1
                + loss_cyc_2
        )

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator 1
        # -----------------------

        optimizer_D1.zero_grad()

        loss_D1 = D1.compute_loss(X1, valid) + D1.compute_loss(X21.detach(), fake)

        loss_D1.backward()
        optimizer_D1.step()

        # -----------------------
        #  Train Discriminator 2
        # -----------------------

        optimizer_D2.zero_grad()

        loss_D2 = D2.compute_loss(X2, valid) + D2.compute_loss(X12.detach(), fake)

        loss_D2.backward()
        optimizer_D2.step()

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
            % (epoch, opt.n_epochs, i, len(train_dataloader), (loss_D1 + loss_D2).item(), loss_G.item(), time_left)
        )

        # If at sample interval save image
        if batches_done % opt.sample_interval == 0:
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D1.step()
    lr_scheduler_D2.step()

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(Enc1.state_dict(), "saved_models/%s/Enc1_%d.pth" % (opt.dataset_name, epoch))
        torch.save(Dec1.state_dict(), "saved_models/%s/Dec1_%d.pth" % (opt.dataset_name, epoch))
        torch.save(Enc2.state_dict(), "saved_models/%s/Enc2_%d.pth" % (opt.dataset_name, epoch))
        torch.save(Dec2.state_dict(), "saved_models/%s/Dec2_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D1.state_dict(), "saved_models/%s/D1_%d.pth" % (opt.dataset_name, epoch))
        torch.save(D2.state_dict(), "saved_models/%s/D2_%d.pth" % (opt.dataset_name, epoch))
'''