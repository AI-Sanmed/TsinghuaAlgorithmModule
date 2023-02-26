import torch
import torch.nn as nn
import numpy as np


class LoGFilter(nn.Module):
    def __init__(self, gaussian_size=3, gaussian_var=1, mask_time=15):
        super().__init__()
        self.gaussian_size = gaussian_size
        self.gaussian_var = gaussian_var
        self.mask_time = mask_time
        self.gaussian = self.generate_3d_gaussian()
        sh = self.gaussian.shape
        self.gaussian = self.gaussian.reshape([1, 1, sh[0], sh[1], sh[2]])
        self.grad = torch.ones((3, 3, 3))
        self.grad[1][1][1] = -26
        self.grad = self.grad / 26
        sh = self.grad.shape
        self.grad = self.grad.reshape([1, 1, sh[0], sh[1], sh[2]])
        self.conv_gauss = nn.Conv3d(1, 1, kernel_size=gaussian_size, stride=1, padding=gaussian_size // 2, bias=False)
        self.conv_gauss.weight.data = self.gaussian
        self.conv_gauss.weight.data.requires_grad = False
        self.conv_grad = nn.Conv3d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_grad.weight.data = self.grad
        self.conv_grad.weight.data.requires_grad = False

    def forward(self, x):
        xg = self.conv_gauss(x)
        x = -self.conv_grad(xg) * self.mask_time + x
        return x

    def generate_3d_gaussian(self):
        a = torch.arange(self.gaussian_size)
        a = a - (self.gaussian_size - 1) / 2
        X, Y, Z = torch.meshgrid(a, a, a)
        E = torch.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * self.gaussian_var ** 2))
        E = E / torch.sum(E)
        return E


class DoubleConv(nn.Module):  # 前向传播之后，图像大小不变
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
        
        
class DilatedConvBlock(nn.Module):
    """
    Dilated conv block from Zhang et al. NODULe
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=1)
        self.conv2 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=5, padding=2)
        self.conv4 = nn.Conv3d(in_channels, in_channels // 4, kernel_size=3, dilation=2, padding=2)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        y = torch.cat([x1, x2, x3, x4], dim=1)
        return y


class Down1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            DoubleConv(in_channels, out_channels)  # 注意channels是信道数量，跟图片大小不是一个维度，没有关系！！
        )

    def forward(self, x):
        y = self.maxpool_conv(x)
        return y
        
        
class DownDilate1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            DoubleConv(in_channels, out_channels),
            DilatedConvBlock(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.maxpool_conv(x)
        return y


class DownLarge1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            DoubleConv(in_channels, out_channels),
            DoubleConv(out_channels, out_channels)
        )

    def forward(self, x):
        y = self.maxpool_conv(x)
        return y


class Down(nn.Module):  # 先把图像缩小至原来的二分之一（向下取整），再调用DoubleConv前向传播,最终输出大小是输入大小的1/2。
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2)),  # No down sampling in the depth dimension!!
            DoubleConv(in_channels, out_channels)  # 注意channels是信道数量，跟图片大小不是一个维度，没有关系！！
        )

    def forward(self, x):
        y = self.maxpool_conv(x)
        return y


class up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        # self.up.weight.data = self.bilinear_kernel3D(in_channels // 2, in_channels // 2, (1, 2, 2))
        self.conv = DoubleConv(in_channels, out_channels)

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


class up1(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up.weight.data = self.bilinear_kernel3D(in_channels // 2, in_channels // 2, 2)
        self.conv = DoubleConv(in_channels, out_channels)

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
        
        
class UpDilate1(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up.weight.data = self.bilinear_kernel3D(in_channels // 2, in_channels // 2, 2)
        self.conv = nn.Sequential(DoubleConv(in_channels, out_channels), nn.ReLU(), DilatedConvBlock(out_channels), nn.ReLU())

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


class UpLarge1(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up.weight.data = self.bilinear_kernel3D(in_channels // 2, in_channels // 2, 2)
        self.conv = nn.Sequential(DoubleConv(in_channels, out_channels),
                                  DoubleConv(out_channels, out_channels))

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
        
        
class DenseBlockSE(nn.Module):
    def __init__(self, in_channels, out_channels, mod=False):
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
        if mod:
            self.se1 = SEBlockMod(in_channels)
            self.se2 = SEBlockMod(in_channels)
            self.se3 = SEBlockMod(in_channels)
            self.se4 = SEBlockMod(out_channels)
        else:
            self.se1 = SEBlock(in_channels)
            self.se2 = SEBlock(in_channels)
            self.se3 = SEBlock(in_channels)
            self.se4 = SEBlock(out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x1se = self.se1(x1)
        x1 = x1 * x1se
        x1c = torch.cat([x1, x], dim=1)
        x2 = self.conv2(x1c)
        x2se = self.se2(x2)
        x2 = x2 * x2se
        x2c = torch.cat([x2, x1, x], dim=1)
        x3 = self.conv3(x2c)
        x3se = self.se3(x3)
        x3 = x3 * x3se
        x3c = torch.cat([x3, x2, x1, x], dim=1)
        x4 = self.conv4(x3c)
        x4se = self.se4(x4)
        x4 = x4 * x4se
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
        
        
class UpDenseSE1(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, mod=False):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.up.weight.data = self.bilinear_kernel3D(in_channels // 2, in_channels // 2, 2)
        self.conv = DenseBlockSE(in_channels, out_channels, mod=mod)

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


class OutConv(nn.Module):  # 最后输出的一层，分割完成后输出图片。
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.attention(x)
        x = x + 1
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, round(in_channels / r))
        self.fc2 = nn.Linear(round(in_channels / r), in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.mean(x, dim=(2, 3, 4))
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        y = self.sigmoid(x1)
        y = y.reshape((y.shape[0], y.shape[1], 1, 1, 1))
        return y


class SEBlockMod(nn.Module):  # uses both mean and max
    def __init__(self, in_channels, r=4):
        super().__init__()
        self.gmp = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.fc1 = nn.Linear(in_channels, round(in_channels / r))
        self.fc2 = nn.Linear(round(in_channels / r), in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.mean(x, dim=(2, 3, 4))
        x11 = self.gmp(x).squeeze(dim=4).squeeze(dim=3).squeeze(dim=2)
        x1 = x1 + x11
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        y = self.sigmoid(x1)
        y = y.reshape((y.shape[0], y.shape[1], 1, 1, 1))
        return y


class AttDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, se=False):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.att = Attention(in_channels, out_channels)
        self.SEblock = SEBlock(out_channels)
        self.se = se

    def forward(self, x):
        x1 = self.double_conv(x)
        x2 = self.att(x)
        y = x2 * x1
        if self.se:
            y1 = self.SEblock(y)
            y = y * y1
        return y


class UNet3DGuideEdgeLower2Cat(nn.Module):
    def __init__(self, in_channels, n_classes, bilinear=True, log=False):
        super().__init__()
        self.log = log
        if log:
            self.loglayer = LoGFilter()

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

        self.OutConv = OutConv(64, n_classes)
        
        self.up21 = up1(128, 32, bilinear)
        self.up31 = up1(64, 32, bilinear)
        self.out2 = OutConv(32, n_classes)

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
        out2 = self.out2(xu11)
        
        xu1 = torch.cat([xu1, xu11], dim=1)
        out = self.OutConv(xu1)

        return out, out2