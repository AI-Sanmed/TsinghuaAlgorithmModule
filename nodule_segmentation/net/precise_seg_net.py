import torch
from torch import nn
from segment_segmentation.net.lobe_net import LoGFilter, DoubleConv, up1, OutConv
import numpy as np


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels * 3, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=in_channels),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels * 4, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x1c = torch.cat([x1, x], dim=1)
        x2 = self.conv2(x1c)
        x2c = torch.cat([x2, x1, x], dim=1)
        x3 = self.conv3(x2c)
        x3c = torch.cat([x3, x2, x1, x], dim=1)
        x4 = self.conv4(x3c)
        return x4


class UpDense1(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up.weight.data = self.bilinear_kernel3D(in_channels // 2, in_channels // 2, 2)
        self.conv = DenseBlock(in_channels, out_channels)

    def forward(self, x1, x2):  # 需要传入之前的卷积层结果作为参数x2，x1为待反卷积的层
        x1 = self.up(x1)

        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

    def bilinear_kernel3D(self, in_channels, out_channels, kernel_size):  # 类中的函数第一个参数写self，后面再加其他参数！
        '''
        return a 3D bilinear filter tensor
        '''
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:kernel_size, :kernel_size, :kernel_size]
        filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor) * (
                1 - abs(og[2] - center) / factor)
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size, kernel_size), dtype='float32')
        weight[range(in_channels), range(out_channels), :, :, :] = filt
        return torch.from_numpy(weight)


class DenseUNet3DGuideEdgeDistLower2ConvCat(nn.Module):
    def __init__(self, in_channels, n_classes, bilinear=True, log=False):
        super().__init__()
        self.log = log
        if log:
            self.loglayer = LoGFilter()

        self.bilinear = bilinear
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.DoubleConv1 = DoubleConv(in_channels, 32)
        self.Down1 = nn.Sequential(nn.MaxPool3d(2, 2), DenseBlock(32, 64))
        self.Down2 = nn.Sequential(nn.MaxPool3d(2, 2), DenseBlock(64, 128))
        self.Down3 = nn.Sequential(nn.MaxPool3d(2, 2), DenseBlock(128, 128))

        self.up1 = UpDense1(256, 64)
        self.up2 = UpDense1(128, 32)
        self.up3 = UpDense1(64, 32)

        self.OutConv = OutConv(64, n_classes)
        
        self.up21 = up1(128, 32, bilinear)
        self.up31 = up1(64, 32, bilinear)
        self.conv = DoubleConv(32, 32)
        self.out2 = OutConv(32, n_classes)
        
        self.up31_d = up1(64, 32, bilinear)
        self.out_d = OutConv(32, n_classes)

    def forward(self, x):
        if self.log:
            x = self.loglayer(x)
        x = self.DoubleConv1(x)
        xd1 = self.Down1(x)
        xd2 = self.Down2(xd1)
        xd3 = self.Down3(xd2)

        xu3 = self.up1(xd3, xd2)
        xu2 = self.up2(xu3, xd1)
        xu1 = self.up3(xu2, x)
        
        xu21 = self.up21(xu3, xd1)
        xu11 = self.up31(xu21, x)
        xu11d = self.up31_d(xu21, x)
        out2 = self.out2(xu11)
        out_d = self.out_d(xu11d)
        
        xu11 = self.conv(xu11)
        xu1 = torch.cat([xu1, xu11], dim=1)
        out = self.OutConv(xu1)

        return out, out2, out_d