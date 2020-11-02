import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
from torch.autograd import Variable
import numpy as np
import os
# from models.networks.sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


#################################
#           Encoder
#################################
class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.content_encoder = ContentEncoder(opt)
        self.style_encoder = StyleEncoder(opt)

    def forward(self, content, style):
        content, content_code = self.content_encoder(content)
        style_code = self.style_encoder(style)
        return content, content_code, style_code


#################################
#           EncoderS
#################################
class StyleEncoder(nn.Module):
    '''
        测试：
        encoder = StyleEncoder(opt)
        style = torch.FloatTensor(np.random.random(size=(2, 3, 32, 32))).cuda()
        style_encoder = StyleEncoder(opt).cuda()
        sx = style_encoder(style)
        输出(downsamle 4 次)：
        style: [2, 3, 32, 32], sx: [2, 5, 1, 1]
    '''

    def __init__(self, opt):
        super(StyleEncoder, self).__init__()
        self.opt = opt
        style_dim = self.opt.label_nc
        n_downsample = self.opt.n_downsample
        dim = self.opt.dim
        in_channels = 3

        # Initial conv block
        self.init_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(in_channels, dim, 7), nn.ReLU(inplace=True))

        # # Downsampling
        # for _ in range(2):
        #     layers += [nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1), nn.ReLU(inplace=True)]
        #     dim *= 2
        #
        # # Downsampling with constant depth
        # for _ in range(n_downsample - 2):
        #     layers += [nn.Conv2d(dim, dim, 4, stride=2, padding=1), nn.ReLU(inplace=True)]

        # self.model = nn.Sequential(*layers)

        self.downmodels = nn.ModuleList()
        # self.channelmodels = nn.ModuleList()
        # Downsampling
        for i in range(2):
            self.downmodels.add_module(
                "se_%d" % i,
                nn.Sequential(
                    nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                ),
            ),
            self.downmodels.add_module(
                "cc%d" % i,
                nn.Sequential(
                    nn.Conv2d(dim * 2, style_dim, 1, 1, 0),
                    nn.ReLU(inplace=True)
                )
            ),
            dim *= 2

        # Downsampling with constant depth
        for i in range(2, n_downsample):
            self.downmodels.add_module(
                "se_%d" % i,
                nn.Sequential(
                    nn.Conv2d(dim, dim, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                ),
            )
            self.downmodels.add_module(
                "cc%d" % i,
                nn.Sequential(
                    nn.Conv2d(dim, style_dim, 1, 1, 0),
                    nn.ReLU(inplace=True)
                )
            ),

        # Average pool and output layer
        self.last_poollayer = nn.AdaptiveAvgPool2d(1)
        # self.lab_poollayer = nn.MaxPool2d(kernel_size=2)
        self.last_covlayer = nn.Conv2d(dim, style_dim, 1, 1, 0)

    def forward(self, x):
        # x = self.init_layer(x)
        # outputs = []
        # print(len(self.downmodels))
        # for i in range(len(self.downmodels)):
        #     x = self.downmodels[i](x)
        #     out = self.channelmodels[i](x)
        #     outputs.append(out)
        # x = self.last_layer(x)
        # outputs.append(x)
        # return outputs
        x = self.init_layer(x)
        # outputs = [x]
        outputs = []
        for i, m in enumerate(self.downmodels):
            if i % 2 == 0:
                x = m(x)
            else:
                out = m(x)
                outputs.append(out)
        # outputs.append(x)
        x = self.last_poollayer(x)
        # outputs.append(x)
        x = self.last_covlayer(x)
        outputs.append(x)
        return outputs


#################################
#           EncoderM
#################################
class ContentEncoder(nn.Module):
    '''
    测试：
    encoder = Encoder(opt)
    content = torch.from_numpy(np.random.random_integers(0, 4, size=(2, 1,32,32))).cuda()
    content_encoder = ContentEncoder(opt).cuda()
    x, cx = content_encoder(content)
    输出(downsamle 4 次)：
    content: [2, 1, 32, 32], x: [2, 5, 32, 32], cx: [2, 1024, 2, 2]
    '''

    def __init__(self, opt):
        super(ContentEncoder, self).__init__()
        self.opt = opt
        in_channels = self.opt.label_nc
        n_downsample = self.opt.n_downsample
        dim = self.opt.dim
        n_residual = self.opt.n_residual
        self.use_gpu = True
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor

        # Initial convolution block
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, dim, 7),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
        ]

        # Downsampling
        for _ in range(n_downsample):
            layers += [
                nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                nn.InstanceNorm2d(dim * 2),
                nn.ReLU(inplace=True),
            ]
            dim *= 2

        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="in")]
        # layers.append(nn.AdaptiveAvgPool2d(1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x = self.preprocess_input(x)
        return x, self.model(x)

    def preprocess_input(self, mask):
        # move to GPU and change data types
        # mask = mask
        # create one-hot label map
        # 将不同的类别映射到不同的channel上
        # 读取的label map是单通道的，背景是0，类别一是1，类别二是2……
        bs, _, h, w = mask.size()
        nc = self.opt.label_nc  # label_nc是包含背景的
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        # 对于该图像不存在的类别的那一个channel会是全0
        # print("#" * 20)
        # print(type(mask))
        # torch.scatter()
        input_semantics = input_label.scatter_(dim=1, index=mask, src=1.0)  # (dim,index,src)

        return input_semantics


#################################
#           Discriminator
#################################
class MultiDiscriminator(nn.Module):
    def __init__(self, opt, in_channels=3):
        super(MultiDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # Extracts three discriminator models
        self.label_nc = opt.label_nc
        self.models = nn.ModuleList()
        for i in range(opt.n_downsample - 2):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(in_channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, self.label_nc, 3, padding=1)
                ),
            )

        self.downsample = nn.AvgPool2d(in_channels, stride=2, padding=[1, 1], count_include_pad=False)

    # MSE loss, 均方根误差
    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs


class EDDiscriminator(nn.Module):
    def __init__(self, opt):
        super(EDDiscriminator, self).__init__()
        self.opt = opt
        style_dim = self.opt.label_nc
        n_downsample = self.opt.n_downsample
        dim = self.opt.dim
        in_channels = 3

        # encoder
        # Initial conv block
        self.init_layer = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(in_channels, dim, 7), nn.ReLU(inplace=True))

        self.downmodels = nn.ModuleList()
        # Downsampling
        for i in range(2):
            self.downmodels.add_module(
                "se_%d" % i,
                nn.Sequential(
                    nn.Conv2d(dim, dim * 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                ),
            ),
            self.downmodels.add_module(
                "cc%d" % i,
                nn.Sequential(
                    nn.Conv2d(dim * 2, style_dim, 1, 1, 0),
                    nn.ReLU(inplace=True)
                )
            ),
            dim *= 2

        # Downsampling with constant depth
        for i in range(2, n_downsample):
            self.downmodels.add_module(
                "se_%d" % i,
                nn.Sequential(
                    nn.Conv2d(dim, dim, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                ),
            )
            self.downmodels.add_module(
                "cc%d" % i,
                nn.Sequential(
                    nn.Conv2d(dim, style_dim, 1, 1, 0),
                    nn.ReLU(inplace=True)
                )
            ),

        # Average pool and output layer
        # self.last_poollayer = nn.AdaptiveAvgPool2d(1)  # 无论如何最后一个都是宽高都是1
        self.last_poollayer = nn.MaxPool2d(kernel_size=2)
        self.last_covlayer = nn.Conv2d(dim, dim, 1, 1, 0)

        # decoder
        # n_residual = opt.n_residual
        # out_channels = 3

        self.upmodels = nn.ModuleList()
        # Upsampling
        for i in range(2):
            self.upmodels.add_module(
                "upconv_%d" % i,
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(dim, dim // 2, 5, stride=1, padding=2),
                    LayerNorm(dim // 2),
                    nn.ReLU(inplace=True),
                ),
            ),
            self.upmodels.add_module(
                "upcontent_%d" % i,
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(dim // 2, style_dim, 1, 1, 0),
                    nn.ReLU(inplace=True)
                )
            ),
            self.upmodels.add_module(
                "uptruefalse%d" % i,
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(dim // 2, 1, 1, 1, 0),
                    nn.ReLU(inplace=True)
                )
            )
            dim //= 2

        # Upsampling with constant depth
        for i in range(2, n_downsample):
            self.upmodels.add_module(
                "upconv_%d" % i,
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(dim, dim, 5, stride=1, padding=2),
                    LayerNorm(dim),
                    nn.ReLU(inplace=True),
                ),
            ),
            self.upmodels.add_module(
                "upcontent_%d" % i,
                nn.Sequential(
                    nn.Upsample(scale_factor=2),
                    nn.Conv2d(dim, style_dim, 1, 1, 0),
                    nn.ReLU(inplace=True)
                )
            ),
            self.upmodels.add_module(
                "uptruefalse_%d" % i,
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(dim, 1, 1, 1, 0),
                    nn.ReLU(inplace=True)
                )
            ),
        # # Residual blocks
        # for _ in range(n_residual):
        #     layers += [ResidualBlock(dim, norm="adain")]
        #
        # # Upsampling
        # for _ in range(n_downsample):
        #     layers += [
        #         nn.Upsample(scale_factor=2),
        #         nn.Conv2d(dim, dim // 2, 5, stride=1, padding=2),
        #         LayerNorm(dim // 2),
        #         nn.ReLU(inplace=True),
        #     ]
        #     dim = dim // 2

        # Average pool and output layer
        # self.last_poollayer = nn.AdaptiveAvgPool2d(1)  # 无论如何最后一个都是宽高都是1
        # # self.lab_poollayer = nn.MaxPool2d(kernel_size=2)
        # self.last_covlayer = nn.Conv2d(dim, style_dim, 1, 1, 0)

        # Output layer
        # layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]
        #
        # self.model = nn.Sequential(*layers)

    def forward(self, img):
        x = self.init_layer(img)
        # outputs = [x]
        feats = []  # encoder部分输出的style特征
        conts = []
        fakereals = []
        for i, m in enumerate(self.downmodels):
            if i % 2 == 0:
                x = m(x)
            else:
                out = m(x)
                feats.append(out)
        # outputs.append(x)
        x = self.last_poollayer(x)
        # outputs.append(x)
        x = self.last_covlayer(x)
        # feats.append(out)

        tag = 0
        for i, m in enumerate(self.upmodels):
            if tag == 0:
                x = m(x)
            elif tag == 1:
                out = m(x)
                conts.append(out)
            else:
                out = m(x)
                fakereals.append(out)

            if tag == 2:
                tag = 0
            else:
                tag += 1

        # # outputs.append(x)
        # x = self.lab_poollayer(x)
        # # outputs.append(x)
        # out = self.last_covlayer(x)
        # outputs.append(out)
        return feats, conts, fakereals


# Feature-Pyramid Semantics Embedding Discriminator
class FPSEDiscriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.dim
        input_nc = 3
        label_nc = opt.label_nc

        self.use_gpu = True
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor

        # bottom-up pathway
        # width = w
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_nc, nf, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(nf, affine=False),
            nn.LeakyReLU(0.2, True))
        # w/2
        self.enc2 = nn.Sequential(
            nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(2 * nf, affine=False),
            nn.LeakyReLU(0.2, True))
        # w/4
        self.enc3 = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 4, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(4 * nf, affine=False),
            nn.LeakyReLU(0.2, True))
        # w/8
        self.enc4 = nn.Sequential(
            nn.Conv2d(nf * 4, nf * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(8 * nf, affine=False),
            nn.LeakyReLU(0.2, True))
        # w/16
        self.enc5 = nn.Sequential(
            nn.Conv2d(nf * 8, nf * 8, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(8 * nf, affine=False),
            nn.LeakyReLU(0.2, True))
        # w/32

        # top-down pathway
        self.lat2 = nn.Sequential(
            nn.Conv2d(nf * 2, nf * 4, kernel_size=1),
            nn.InstanceNorm2d(4 * nf, affine=False),
            nn.LeakyReLU(0.2, True))
        self.lat3 = nn.Sequential(
            nn.Conv2d(nf * 4, nf * 4, kernel_size=1),
            nn.InstanceNorm2d(4 * nf, affine=False),
            nn.LeakyReLU(0.2, True))
        self.lat4 = nn.Sequential(
            nn.Conv2d(nf * 8, nf * 4, kernel_size=1),
            nn.InstanceNorm2d(4 * nf, affine=False),
            nn.LeakyReLU(0.2, True))
        self.lat5 = nn.Sequential(
            nn.Conv2d(nf * 8, nf * 4, kernel_size=1),
            nn.InstanceNorm2d(4 * nf, affine=False),
            nn.LeakyReLU(0.2, True))

        # upsampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        # final layers
        self.final2 = nn.Sequential(
            nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(2 * nf, affine=False),
            nn.LeakyReLU(0.2, True))
        self.final3 = nn.Sequential(
            nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(2 * nf, affine=False),
            nn.LeakyReLU(0.2, True))
        self.final4 = nn.Sequential(
            nn.Conv2d(nf * 4, nf * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(2 * nf, affine=False),
            nn.LeakyReLU(0.2, True))

        # true/false prediction and semantic alignment prediction
        self.tf = nn.Conv2d(nf * 2, 1, kernel_size=1)
        self.seg = nn.Conv2d(nf * 2, nf * 2, kernel_size=1)
        self.embedding = nn.Conv2d(label_nc, nf * 2, kernel_size=1)

    def forward(self, img, has_segmap=False, segmap=None):
        # segmap输入为灰度
        # bottom-up pathway
        feat11 = self.enc1(img)  # w/2

        feat12 = self.enc2(feat11)  # w/4
        feat13 = self.enc3(feat12)  # w/8
        feat14 = self.enc4(feat13)  # w/16
        feat15 = self.enc5(feat14)  # w/32
        # top-down pathway and lateral connections
        feat25 = self.lat5(feat15)  # w/32
        feat24 = self.up(feat25) + self.lat4(feat14)  # w/16
        feat23 = self.up(feat24) + self.lat3(feat13)  # w/8
        feat22 = self.up(feat23) + self.lat2(feat12)  # w/4
        # final prediction layers
        feat32 = self.final2(feat22)  # w/4
        feat33 = self.final3(feat23)  # w/8
        feat34 = self.final4(feat24)  # w/16

        # Patch-based True/False prediction
        pred2 = self.tf(feat32)
        pred3 = self.tf(feat33)
        pred4 = self.tf(feat34)

        # intermediate features for discriminator feature matching loss
        # style consistence loss
        feats = [feat12, feat13, feat14, feat15]  # w/4,w/8,w/16,w/32
        truefalses = [pred2, pred3, pred4]  # w/4,w/8,2/16
        # segmentation map embedding
        if has_segmap:
            seg2 = self.seg(feat32)  # w/4
            seg3 = self.seg(feat33)  # w/8
            seg4 = self.seg(feat34)  # w/16

            segmap = self.preprocess_input(segmap)
            segemb = self.embedding(segmap)
            segemb = F.avg_pool2d(segemb, kernel_size=2, stride=2)  # w/2
            segemb2 = F.avg_pool2d(segemb, kernel_size=2, stride=2)  # w/4
            segemb3 = F.avg_pool2d(segemb2, kernel_size=2, stride=2)  # w/8
            segemb4 = F.avg_pool2d(segemb3, kernel_size=2, stride=2)  # w/16

            # semantics embedding discriminator score
            pred2 += torch.mul(segemb2, seg2).sum(dim=1, keepdim=True)
            pred3 += torch.mul(segemb3, seg3).sum(dim=1, keepdim=True)
            pred4 += torch.mul(segemb4, seg4).sum(dim=1, keepdim=True)

            # concat results from multiple resolutions
            segpreds = [pred2, pred3, pred4]  # w/4,w/8,w/16
            return [feats, truefalses, segpreds]
        else:
            return [feats, truefalses]

    def preprocess_input(self, mask):
        # move to GPU and change data types
        mask = mask.cuda()

        # create one-hot label map
        # 将不同的类别映射到不同的channel上
        # 读取的label map是单通道的，背景是0，类别一是1，类别二是2……
        bs, _, h, w = mask.size()
        nc = self.opt.label_nc  # label_nc是包含背景的
        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        # 对于该图像不存在的类别的那一个channel会是全0
        input_semantics = input_label.scatter_(1, mask, 1.0)  # (dim,index,src)

        return input_semantics


#################################
#            Decoder
#################################
class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.opt = opt
        n_residual = opt.n_residual
        out_channels = 3
        n_upsample = opt.n_downsample
        style_dim = opt.label_nc

        layers = []
        input_dim = opt.dim * 2 ** n_upsample
        dim = input_dim
        # Residual blocks
        for _ in range(n_residual):
            layers += [ResidualBlock(dim, norm="adain")]

        # Upsampling
        for _ in range(n_upsample):
            layers += [
                nn.Upsample(scale_factor=2),
                nn.Conv2d(dim, dim // 2, 5, stride=1, padding=2),
                LayerNorm(dim // 2),
                nn.ReLU(inplace=True),
            ]
            dim = dim // 2

        # Output layer
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(dim, out_channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*layers)

        # Initiate mlp (predicts AdaIN parameters)
        num_adain_params = self.get_num_adain_params()
        self.changechannel = nn.Conv2d(input_dim, style_dim, 1, 1)
        self.changesize = nn.AdaptiveAvgPool2d(1)
        self.mlp = MLP(style_dim, num_adain_params)

    def get_num_adain_params(self):
        """Return the number of AdaIN parameters needed by the model"""
        # print("Return the number of AdaIN parameters needed by the model")
        num_adain_params = 0
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                # print(m.num_features)
                num_adain_params += 2 * m.num_features

        # print(num_adain_params)
        return num_adain_params

    def assign_adain_params(self, adain_params):
        """Assign the adain_params to the AdaIN layers in model"""
        for m in self.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                # Extract mean and std predictions
                mean = adain_params[:, : m.num_features]
                std = adain_params[:, m.num_features: 2 * m.num_features]
                # Update bias and weight
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                # Move pointer
                if adain_params.size(1) > 2 * m.num_features:
                    adain_params = adain_params[:, 2 * m.num_features:]

    def forward(self, content_code, style_code):
        # Update AdaIN parameters by MLP prediction based off style code
        new_c = self.changesize(self.changechannel(content_code))
        print(new_c.shape, style_code.shape)
        style_conditional_content = torch.matmul(new_c, style_code)
        print("style_conditional_content:", style_conditional_content.shape)
        self.assign_adain_params(self.mlp(style_conditional_content))
        img = self.model(content_code)
        return img


######################################
#   MLP (predicts AdaIn parameters)
######################################
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, dim=256, n_blk=3, activ="relu"):
        super(MLP, self).__init__()

        layers = [nn.Linear(input_dim, dim), nn.ReLU(inplace=True)]
        for _ in range(n_blk - 2):
            layers += [nn.Linear(dim, dim), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dim, output_dim)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))


##############################
#       Custom Blocks
##############################
class ResidualBlock(nn.Module):
    def __init__(self, features, norm="in"):
        super(ResidualBlock, self).__init__()

        norm_layer = AdaptiveInstanceNorm2d if norm == "adain" else nn.InstanceNorm2d

        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            norm_layer(features),
        )

    def forward(self, x):
        return x + self.block(x)


##############################
#        Custom Layers
##############################
class AdaptiveInstanceNorm2d(nn.Module):
    """Reference: https://github.com/NVlabs/MUNIT/blob/master/networks.py"""

    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        assert (
                self.weight is not None and self.bias is not None
        ), "Please assign weight and bias before calling AdaIN!"
        b, c, h, w = x.size()
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, h, w)

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias, True, self.momentum, self.eps
        )

        return out.view(b, c, h, w)

    def __repr__(self):
        return self.__class__.__name__ + "(" + str(self.num_features) + ")"


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


# VGG architecter, used for the perceptual loss using a pretrained VGG network
class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()

        # 原 vgg19 =  16 卷积层 + 3 全连接层
        # torchvision中的vgg不包含全连接层，共16个卷积层
        # 每个卷积层带一个relu，因此vgg19总的层数为16*2+5 = 37 [0:37]
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


if __name__ == "__main__":
    import argparse

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

    wid = opt.img_height
    encoder = Encoder(opt).cuda()
    # print(encoder)
    content = torch.from_numpy(np.random.random_integers(0, 4, size=(4, 1, wid, wid))).cuda()
    style = torch.FloatTensor(np.random.random(size=(4, 3, wid, wid))).cuda()
    print(content.shape, style.shape)
    c, cc, sc = encoder(content, style)
    # torch.Size([2, 5, 64, 64]) torch.Size([2, 1024, 4, 4]) torch.Size([2, 5, 1, 1])
    print(c.shape, cc.shape)
    print("*" * 10 + "style encoder" + "*" * 10)
    for i in sc:
        print(i.shape)

    print("*" * 10 + "generated" + "*" * 10)
    decoder = Decoder(opt).cuda()
    # print(decoder)
    generated = decoder(cc, sc[-1])
    # torch.Size([2, 3, 64, 64])
    print(generated.shape)

    # print("*" * 10 + "discriminate" + "*" * 10)
    # discriminator = MultiDiscriminator(opt).cuda()
    # results = discriminator(generated)
    # # print(discriminator)
    # # 3 torch.Size([2, 1, 4, 4]) torch.Size([2, 1, 2, 2]) torch.Size([2, 1, 1, 1])
    # # print(len(results), results[0].shape, results[1].shape, results[-1].shape)
    # for r in results:
    #     print(r.shape)

    discriminator2 = EDDiscriminator(opt).cuda()
    # fake_real = torch.cat((generated, style), dim=0)
    # print(discriminator2)
    # print(discriminator2)
    print("*" * 10 + "real" + "*" * 10)
    [feats, segpreds, truefalses] = discriminator2(style)
    print('...feature...')
    for i in feats:
        print(i.shape)
    print('...true false...')
    for j in truefalses:
        print(j.shape)
    print('...seg map...')
    for k in segpreds:
        print(k.shape)

    print("*" * 10 + "generated" + "*" * 10)
    [feats2, segpreds2, truefalses2] = discriminator2(generated)
    print('...feature...')
    for i in feats2:
        print(i.shape)
    print('...true false...')
    for j in truefalses2:
        print(j.shape)
    print('...seg map')
    for k in segpreds2:
        print(k.shape)
