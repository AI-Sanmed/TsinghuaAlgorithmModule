import torch
import torch.nn as nn
import numpy as np


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y = self.double_conv(x)
        return y


class Down1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        y = self.maxpool_conv(x)
        return y


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        y = self.maxpool_conv(x)
        return y


class up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class up1(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up.weight.data = self.bilinear_kernel3D(in_channels // 2, in_channels // 2, 2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

    def bilinear_kernel3D(self, in_channels, out_channels, kernel_size):
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


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels, n_classes, bilinear=True):
        super(UNet3D, self).__init__()

        self.bilinear = bilinear
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.DoubleConv1 = DoubleConv(in_channels, 32)
        self.Down1 = Down1(32, 64)
        self.Down2 = Down1(64, 128)
        self.Down3 = Down1(128, 128)

        self.up1 = up1(256, 64, bilinear)
        self.up2 = up1(128, 32, bilinear)
        self.up3 = up1(64, 32, bilinear)

        self.OutConv = OutConv(32, n_classes)

    def forward(self, x):
        x = self.DoubleConv1(x)
        xd1 = self.Down1(x)
        xd2 = self.Down2(xd1)
        xd3 = self.Down3(xd2)

        xu3 = self.up1(xd3, xd2)
        xu2 = self.up2(xu3, xd1)
        xu1 = self.up3(xu2, x)

        out = self.OutConv(xu1)

        return out