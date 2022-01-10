""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
logger = logging.getLogger('log')


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(mid_channels),
            nn.GroupNorm(np.minimum(mid_channels, 4), mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(out_channels),
            nn.GroupNorm(np.minimum(out_channels, 4), out_channels),
            nn.ReLU()
            )

    def forward(self, x):
        return self.double_conv(x)


class MSP(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(DoubleConv(in_channels*2, in_channels))

    def forward(self, x):
        for i in range(len(x)):
            condition = list(range(len(x)))
            condition.remove(i)
            if i==0:
                common_features = torch.sum(x[1:], 0).unsqueeze(0)
            else:
                common_features = torch.cat([common_features, torch.sum(x[condition], 0).unsqueeze(0)], 0)
        x = torch.cat([x, common_features], 1)
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MyModel(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(MyModel, self).__init__()
        # self.l1 = nn.Sequential(nn.Linear(16, 1), nn.ReLU(True))
        # self.msp0 = MSP(16)
        self.inc = DoubleConv(16, 32)
        self.msp1 = MSP(32)
        self.down1 = Down(32, 64)  # 16
        self.msp2 = MSP(64)
        self.down2 = Down(64, 128)  # 32
        self.msp3 = MSP(128)
        self.down3 = Down(128, 256)  # 64
        self.msp4 = MSP(256)
        self.down4 = Down(256, 128)  # 64
        self.msp5 = MSP(128)
        self.down5 = Down(128, 1)  # 8
        # self.msp6 = MSP(128)
        # self.down6 = Down(128, 32)  # 128
        # self.msp7 = MSP(64)
        # self.down7 = Down(64, 32)  # 128
        # self.msp8 = MSP(32)
        # self.down8 = Down(32, 16)  # 64
        self.linear_out = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(-1, 16, 256, 256)
        x = x[1:, :, :, :]
        # x = self.msp0(x)
        x = self.inc(x)
        x = self.msp1(x)
        x = self.down1(x)
        x = self.msp2(x)
        x = self.down2(x)
        x = self.msp3(x)
        x = self.down3(x)
        x = self.msp4(x)
        x = self.down4(x)
        x = self.msp5(x)
        x = self.down5(x)
        # x = self.msp6(x)
        # x = self.down6(x)
        # x = self.msp7(x)
        # x = self.down7(x)
        x = torch.mean(x, 0)
        x = torch.flatten(x).unsqueeze(0)
        x = self.linear_out(x)

        return x


def get_convmpn_model(name, inp_dim=1, out_dim=1):
    if (name == 'convmpn'):
        model = MyModel(inp_dim, out_dim)
    else:
        logging.error("model type {} has not found".format(name))
        sys.exit(1)

    model = model.cuda()
    return model
