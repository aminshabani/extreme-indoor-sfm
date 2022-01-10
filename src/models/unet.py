""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
logger = logging.getLogger('log')


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if mid_channels != 1:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.GroupNorm(4,mid_channels), nn.ReLU(),)
                # nn.BatchNorm2d(mid_channels), nn.ReLU(),)
                # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                # nn.GroupNorm(4,out_channels), nn.ReLU())
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.GroupNorm(1,mid_channels), nn.ReLU(),)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels,
                                         in_channels // 2,
                                         kernel_size=2,
                                         stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)
        '''
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        '''
        x = x1
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, decoder=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.decoder = decoder

        base_ch = 4
        self.inc = DoubleConv(n_channels, 4)
        self.down1 = DoubleConv(4, 8)  # 32
        self.down2 = Down(8, 8)  # 128
        self.down3 = Down(8, 8)  # 128
        self.down4 = Down(8, 8)  # 256
        self.down5 = Down(8, 8)  # 256
        self.down6 = Down(8, 8)  # 128
        self.down7 = Down(8, 8)  # 128
        # self.down8 = Down(8, 8)  # 64
        self.linear = nn.Linear(128, n_classes)
        # self.do = nn.Dropout()
        if (decoder):
            self.up1 = Up(1024, 512, bilinear)
            self.up2 = Up(512, 256, bilinear)
            self.up3 = Up(256, 128, bilinear)
            self.up4 = Up(128, 64, bilinear)
            self.up5 = Up(64, 32, bilinear)
            self.up6 = Up(32, 16, bilinear)
            self.up7 = Up(16, 8, bilinear)
            self.up8 = Up(8, 8, bilinear)
            self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x8 = self.down7(x7)
        # x9 = self.down8(x8)
        if (self.decoder):
            x = self.up1(x9)
            x = self.up2(x)
            x = self.up3(x)
            x = self.up4(x)
            x = self.up5(x)
            x = self.up6(x)
            x = self.up7(x)
            x = self.up8(x)
            x = self.outc(x)
            output = torch.tanh(x)
        else:
            output = torch.flatten(x8, 1)
            # output = self.do(output)
            # print(output.shape)
            output = self.linear(output)
        # logger.debug("output size is {}".format(output.size()))
        return output
