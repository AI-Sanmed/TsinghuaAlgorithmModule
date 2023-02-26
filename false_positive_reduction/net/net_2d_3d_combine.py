import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, bn_track=True, se=False):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels, track_running_stats=bn_track),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels, track_running_stats=bn_track),
            nn.ReLU(inplace=True)
        )
        self.se = se
        if se:
            self.SEBlock = SEBlock(out_channels)

    def forward(self, x):
        x = self.double_conv(x)
        if self.se:
            y1 = self.SEBlock(x)
            x = x * y1
        return x
        
        
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
        
        
class Attention2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
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


class SEBlock2d(nn.Module):
    def __init__(self, in_channels, r=4):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, round(in_channels / r))
        self.fc2 = nn.Linear(round(in_channels / r), in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = torch.mean(x, dim=(2, 3))
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        y = self.sigmoid(x1)
        y = y.reshape((y.shape[0], y.shape[1], 1, 1))
        return y


class DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, bn_track=True, se=False):
        super(DoubleConv2d, self).__init__()
        self.se = se
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, track_running_stats=bn_track),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, track_running_stats=bn_track),
            nn.ReLU(inplace=True)
        )
        if self.se:
            self.se_block = SEBlock2d(out_channels)

    def forward(self, x):
        x = self.double_conv(x)
        if self.se:
            y = self.se_block(x)
            x = y * x
        return x


class AttDoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, bn_track=True, se=False):
        super(AttDoubleConv2d, self).__init__()
        self.se = se
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, track_running_stats=bn_track),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels, track_running_stats=bn_track),
            nn.ReLU(inplace=True)
        )
        self.att = Attention2d(in_channels, out_channels)
        if self.se:
            self.se_block = SEBlock2d(out_channels)

    def forward(self, x):
        x1 = self.att(x)
        x = self.double_conv(x)
        x = x * x1
        if self.se:
            y = self.se_block(x)
            x = y * x
        return x


class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SKConv, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=5, padding=2)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, max(out_channels / 4, 16)),
            nn.BatchNorm1d(max(out_channels / 4, 16)),
            nn.ReLU(inplace=True))
        self.fca = nn.Linear(max(out_channels / 4, 16), out_channels)
        self.fcb = nn.Linear(max(out_channels / 4, 16), out_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        xu = x1 + x2
        xu = torch.mean(xu, dim=(2, 3, 4))
        xu = self.fc(xu)
        xua = torch.exp(self.fca(xu))
        xub = torch.exp(self.fcb(xu))
        xus = xua + xub
        xua /= xus
        xub /= xus
        xua = torch.reshape(xua, (xua.shape[0], xua.shape[1], 1, 1, 1))
        xub = torch.reshape(xub, (xub.shape[0], xub.shape[1], 1, 1, 1))
        x1 *= xua
        x2 *= xub
        out = x1 + x2
        return out


class DoubleSK(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleSK, self).__init__()
        self.sk1 = SKConv(in_channels, out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sk2 = SKConv(out_channels, out_channels)

    def forward(self, x):
        x = self.sk1(x)
        x = self.relu(x)
        x = self.sk2(x)
        x = self.relu(x)
        return x


class Net32_2d(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, se=False):
        super(Net32_2d, self).__init__()
        self.conv1 = DoubleConv2d(in_channels, 32, se=se)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = DoubleConv2d(32, 64, se=se)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = DoubleConv2d(64, 128, se=se)
        self.fc1 = nn.Linear(8192, 128)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = x.reshape([-1, 8192])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x, x1
        

class Net24_2d(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, se=False):
        super(Net24_2d, self).__init__()
        self.conv1 = DoubleConv2d(in_channels, 32, se=se)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = DoubleConv2d(32, 64, se=se)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = DoubleConv2d(64, 128, se=se)
        self.fc1 = nn.Linear(4608, 96)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(96, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = x.reshape([-1, 4608])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x, x1
        
        
class Net40_2d(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, se=False):
        super(Net40_2d, self).__init__()
        self.conv1 = DoubleConv2d(in_channels, 32, se=se)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = DoubleConv2d(32, 64, se=se)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = DoubleConv2d(64, 128, se=se)
        self.fc1 = nn.Linear(12800, 160)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(160, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = x.reshape([-1, 12800])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x, x1
        
        
class Net24_2dLarge(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, se=False):
        super(Net24_2dLarge, self).__init__()
        self.conv1 = DoubleConv2d(in_channels, 32, se=se)
        self.conv11 = DoubleConv2d(32, 32, se=se)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = DoubleConv2d(32, 64, se=se)
        self.conv21 = DoubleConv2d(64, 64, se=se)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = DoubleConv2d(64, 128, se=se)
        self.fc1 = nn.Linear(4608, 96)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(96, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv21(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = x.reshape([-1, 4608])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x, x1
        
        
class Net40_2dLarge(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, se=False):
        super(Net40_2dLarge, self).__init__()
        self.conv1 = DoubleConv2d(in_channels, 32, se=se)
        self.conv11 = DoubleConv2d(32, 32, se=se)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = DoubleConv2d(32, 64, se=se)
        self.conv21 = DoubleConv2d(64, 64, se=se)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = DoubleConv2d(64, 128, se=se)
        self.fc1 = nn.Linear(12800, 160)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(160, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv21(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = x.reshape([-1, 12800])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x, x1


class Net32SE(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, bn_track=True):
        super().__init__()
        self.conv1 = AttDoubleConv(in_channels, 16, se=True)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = AttDoubleConv(16, 32, se=True)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.fc1 = nn.Linear(16384, 1600)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1600, 65)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(65, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.reshape(x, [-1, 16384])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x, x1
        
        
class Net40SE(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, bn_track=True):
        super().__init__()
        self.conv1 = AttDoubleConv(in_channels, 16, se=True)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = AttDoubleConv(16, 32, se=True)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.fc1 = nn.Linear(32000, 2400)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2400, 85)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(85, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.reshape(x, [-1, 32000])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x, x1
        
        
class Net24SE(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, bn_track=True):
        super().__init__()
        self.conv1 = AttDoubleConv(in_channels, 16, se=True)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = AttDoubleConv(16, 32, se=True)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.fc1 = nn.Linear(6912, 1200)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1200, 45)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(45, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.reshape(x, [-1, 6912])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x, x1


class Net2d3dMerge(nn.Module):
    def __init__(self, in_channels, n_classes, se=False, dropout_rate=0.0):
        super(Net2d3dMerge, self).__init__()
        self.net_2d = Net32_2d(in_channels, n_classes, se=se, dropout_rate=dropout_rate)
        self.net_3d = Net32SE(in_channels, n_classes, dropout_rate=dropout_rate)

        self.fc1 = nn.Linear(24576, 1000)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1000, n_classes)

    def forward(self, x):
        x0 = x[:, :, 15, :, :]
        x_2d, x1_2d = self.net_2d(x0)
        x_3d, x1_3d = self.net_3d(x)
        x1 = torch.cat([x1_2d, x1_3d], dim=1)
        x1 = self.fc1(x1)
        x1 = self.drop1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        return x1, x_2d, x_3d
        
        
class Net2d3dMerge40(nn.Module):
    def __init__(self, in_channels, n_classes, se=False, dropout_rate=0.0):
        super(Net2d3dMerge40, self).__init__()
        self.net_2d = Net40_2d(in_channels, n_classes, se=se, dropout_rate=dropout_rate)
        self.net_3d = Net40SE(in_channels, n_classes, dropout_rate=dropout_rate)

        self.fc1 = nn.Linear(44800, 1500)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1500, n_classes)

    def forward(self, x):
        x0 = x[:, :, 15, :, :]
        x_2d, x1_2d = self.net_2d(x0)
        x_3d, x1_3d = self.net_3d(x)
        x1 = torch.cat([x1_2d, x1_3d], dim=1)
        x1 = self.fc1(x1)
        x1 = self.drop1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        return x1, x_2d, x_3d
        
        
class Net2d3dMerge24(nn.Module):
    def __init__(self, in_channels, n_classes, se=False, dropout_rate=0.0):
        super(Net2d3dMerge24, self).__init__()
        self.net_2d = Net24_2d(in_channels, n_classes, se=se, dropout_rate=dropout_rate)
        self.net_3d = Net24SE(in_channels, n_classes, dropout_rate=dropout_rate)

        self.fc1 = nn.Linear(11520, 800)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(800, n_classes)

    def forward(self, x):
        x0 = x[:, :, 15, :, :]
        x_2d, x1_2d = self.net_2d(x0)
        x_3d, x1_3d = self.net_3d(x)
        x1 = torch.cat([x1_2d, x1_3d], dim=1)
        x1 = self.fc1(x1)
        x1 = self.drop1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        return x1, x_2d, x_3d
        
        
class Net2d3dMergeV2(nn.Module):
    """
    Uses 3 2d slices from 3 angles instead of 1 slice.
    """
    def __init__(self, in_channels, n_classes, se=False):
        super(Net2d3dMergeV2, self).__init__()
        self.net_2d_1 = Net32_2d(in_channels, n_classes, se=se)
        self.net_2d_2 = Net32_2d(in_channels, n_classes, se=se)
        self.net_2d_3 = Net32_2d(in_channels, n_classes, se=se)
        self.net_3d = Net32SE(in_channels, n_classes)

        self.fc1 = nn.Linear(40960, 1000)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1000, n_classes)

    def forward(self, x):
        x00 = x[:, :, 15, :, :]
        x01 = x[:, :, :, 15, :]
        x02 = x[:, :, :, :, 15]
        x_2d0, x1_2d0 = self.net_2d_1(x00)
        x_2d1, x1_2d1 = self.net_2d_2(x01)
        x_2d2, x1_2d2 = self.net_2d_3(x02)
        x_3d, x1_3d = self.net_3d(x)
        x1 = torch.cat([x1_2d0, x1_2d1, x1_2d2, x1_3d], dim=1)
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        return x1, x_2d0, x_2d1, x_2d2, x_3d


class Net2d3dMergeV3(nn.Module):
    """
    Uses 3 2d slices from 3 angles instead of 1 slice.
    Firstly merge the 3 2d feature maps, then merge 2d and 3d.
    """
    def __init__(self, in_channels, n_classes, se=False):
        super(Net2d3dMergeV3, self).__init__()
        self.net_2d_1 = Net32_2d(in_channels, n_classes, se=se)
        self.net_2d_2 = Net32_2d(in_channels, n_classes, se=se)
        self.net_2d_3 = Net32_2d(in_channels, n_classes, se=se)
        self.net_3d = Net32SE(in_channels, n_classes)

        self.fcb1 = nn.Linear(24576, 1000)
        self.fcb2 = nn.Linear(16384, 1000)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2000, n_classes)

    def forward(self, x):
        x00 = x[:, :, 15, :, :]
        x01 = x[:, :, :, 15, :]
        x02 = x[:, :, :, :, 15]
        x_2d0, x1_2d0 = self.net_2d_1(x00)
        x_2d1, x1_2d1 = self.net_2d_2(x01)
        x_2d2, x1_2d2 = self.net_2d_3(x02)
        x_3d, x1_3d = self.net_3d(x)
        xb1 = torch.cat([x1_2d0, x1_2d1, x1_2d2], dim=1)
        xb1 = self.fcb1(xb1)
        xb1 = self.relu(xb1)
        xb2 = self.fcb2(x1_3d)
        xb2 = self.relu(xb2)
        x1 = torch.cat([xb1, xb2], dim=1)
        x1 = self.fc2(x1)
        return x1, x_2d0, x_2d1, x_2d2, x_3d


class Net2d3dMergeMIP(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, se=False):
        super(Net2d3dMergeMIP, self).__init__()
        self.net_2d = Net32_2dLarge(in_channels, n_classes, se=se)
        self.net_3d = Net32SELarge(in_channels, n_classes)

        self.fc1 = nn.Linear(24576, 1000)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1000, n_classes)

    def forward(self, x):
        x0 = x[:, :, 14:17, :, :]
        x0 = torch.max(x0, dim=2)[0]
        x_2d, x1_2d = self.net_2d(x0)
        x_3d, x1_3d = self.net_3d(x)
        x1 = torch.cat([x1_2d, x1_3d], dim=1)
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        return x1, x_2d, x_3d
        
        
class Net2d3dMergeV3MIP(nn.Module):
    def __init__(self, in_channels, n_classes, se=False):
        super(Net2d3dMergeV3MIP, self).__init__()
        self.net_2d_1 = Net32_2d(in_channels, n_classes, se=se)
        self.net_2d_2 = Net32_2d(in_channels, n_classes, se=se)
        self.net_2d_3 = Net32_2d(in_channels, n_classes, se=se)
        self.net_3d = Net32SE(in_channels, n_classes)

        self.fcb1 = nn.Linear(24576, 1000)
        self.fcb2 = nn.Linear(16384, 1000)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2000, n_classes)

    def forward(self, x):
        x00 = x[:, :, 13:18, :, :]
        x01 = x[:, :, :, 13:18, :]
        x02 = x[:, :, :, :, 13:18]
        x00 = torch.max(x00, dim=2)[0]
        x01 = torch.max(x01, dim=3)[0]
        x02 = torch.max(x02, dim=4)[0]
        x_2d0, x1_2d0 = self.net_2d_1(x00)
        x_2d1, x1_2d1 = self.net_2d_2(x01)
        x_2d2, x1_2d2 = self.net_2d_3(x02)
        x_3d, x1_3d = self.net_3d(x)
        xb1 = torch.cat([x1_2d0, x1_2d1, x1_2d2], dim=1)
        xb1 = self.fcb1(xb1)
        xb1 = self.relu(xb1)
        xb2 = self.fcb2(x1_3d)
        xb2 = self.relu(xb2)
        x1 = torch.cat([xb1, xb2], dim=1)
        x1 = self.fc2(x1)
        return x1, x_2d0, x_2d1, x_2d2, x_3d
        
        
class Net2d3dMergeLHI(nn.Module):
    def __init__(self, in_channels, n_classes, se=False):
        super(Net2d3dMergeLHI, self).__init__()
        self.net_2d = Net32_2d(in_channels, n_classes, se=se)
        self.net_3d = Net32SE(in_channels, n_classes)

        self.fc1 = nn.Linear(24576, 1000)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1000, n_classes)

    def forward(self, x):
        x_lhi = self._get_lhi(x)
        x_2d, x1_2d = self.net_2d(x_lhi)
        x_3d, x1_3d = self.net_3d(x)
        x1 = torch.cat([x1_2d, x1_3d], dim=1)
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        return x1, x_2d, x_3d
        
    def _get_lhi(self, x):
        im_lhi_last = torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[4]).to(device=x.device)
        for j in range(11, 19):
            im = x[:, :, j, :, :]
            diff = torch.abs(im - x[:, :, j-1, :, :])
            thres = 25 / 300
            diff[diff < thres] = 0
            diff[diff >= thres] = 1
            im_lhi = torch.zeros(im.shape).to(device=x.device)
            im_lhi[diff == 0] = im_lhi_last[diff == 0] - 1
            im_lhi[diff == 1] = 11
            im_lhi[im_lhi < 0] = 0
            im_lhi_last = im_lhi
        return im_lhi_last / 11
        
        
class Net2d3dMergeV4(nn.Module):
    def __init__(self, in_channels, n_classes, se=False, dropout_rate=0.0):
        super(Net2d3dMergeV4, self).__init__()
        self.net_2d = Net32_2d(3, n_classes, se=se, dropout_rate=dropout_rate)
        self.net_3d = Net32SE(in_channels, n_classes, dropout_rate=dropout_rate)

        self.fc1 = nn.Linear(24576, 1000)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1000, n_classes)

    def forward(self, x):
        x0_1 = x[:, :, 13, :, :]
        x0_2 = x[:, :, 15, :, :]
        x0_3 = x[:, :, 17, :, :]
        x0 = torch.cat([x0_1, x0_2, x0_3], dim=1)
        x_2d, x1_2d = self.net_2d(x0)
        x_3d, x1_3d = self.net_3d(x)
        x1 = torch.cat([x1_2d, x1_3d], dim=1)
        x1 = self.fc1(x1)
        x1 = self.drop1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        return x1, x_2d, x_3d
            

class Net32SK(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, bn_track=True):
        super().__init__()
        self.conv1 = DoubleSK(in_channels, 16)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = DoubleSK(16, 32)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.fc1 = nn.Linear(16384, 1600)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1600, 65)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(65, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.reshape(x, [-1, 16384])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x, x1
        
        
class Net32SELarge(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, bn_track=True):
        super().__init__()
        self.conv1 = AttDoubleConv(in_channels, 16, se=True)
        self.conv11 = AttDoubleConv(16, 16, se=True)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = AttDoubleConv(16, 32, se=True)
        self.conv22 = AttDoubleConv(32, 32, se=True)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.fc1 = nn.Linear(16384, 1600)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1600, 65)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(65, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv22(x)
        x = self.pool2(x)
        x = torch.reshape(x, [-1, 16384])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x, x1
        
        
class Net40SELarge(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, bn_track=True):
        super().__init__()
        self.conv1 = AttDoubleConv(in_channels, 16, se=True)
        self.conv11 = AttDoubleConv(16, 16, se=True)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = AttDoubleConv(16, 32, se=True)
        self.conv22 = AttDoubleConv(32, 32, se=True)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.fc1 = nn.Linear(32000, 2400)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(2400, 85)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(85, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv22(x)
        x = self.pool2(x)
        x = torch.reshape(x, [-1, 32000])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x, x1
        
        
class Net24SELarge(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, bn_track=True):
        super().__init__()
        self.conv1 = AttDoubleConv(in_channels, 16, se=True)
        self.conv11 = AttDoubleConv(16, 16, se=True)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = AttDoubleConv(16, 32, se=True)
        self.conv22 = AttDoubleConv(32, 32, se=True)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.fc1 = nn.Linear(6912, 1200)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1200, 45)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(45, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv22(x)
        x = self.pool2(x)
        x = torch.reshape(x, [-1, 6912])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x, x1
        
        
class Net32SELarger(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, bn_track=True):
        super().__init__()
        self.conv1 = AttDoubleConv(in_channels, 16, se=True)
        self.conv11 = AttDoubleConv(16, 16, se=True)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = AttDoubleConv(16, 32, se=True)
        self.conv22 = AttDoubleConv(32, 32, se=True)
        self.conv23 = AttDoubleConv(32, 32, se=True)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.conv3 = AttDoubleConv(32, 32, se=True)
        self.conv32 = AttDoubleConv(32, 32, se=True)
        self.conv33 = AttDoubleConv(32, 32, se=True)
        self.fc1 = nn.Linear(16384, 1600)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1600, 65)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(65, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv22(x)
        x = self.conv23(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = torch.reshape(x, [-1, 16384])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x, x1
        
        
class Net32_2dLarge(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, se=False):
        super(Net32_2dLarge, self).__init__()
        self.conv1 = DoubleConv2d(in_channels, 32, se=se)
        self.conv11 = DoubleConv2d(32, 32, se=se)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = DoubleConv2d(32, 64, se=se)
        self.conv21 = DoubleConv2d(64, 64, se=se)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = DoubleConv2d(64, 128, se=se)
        # self.conv31 = DoubleConv2d(128, 128, se=se)
        self.fc1 = nn.Linear(8192, 128)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv21(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # x = self.conv31(x)
        x = x.reshape([-1, 8192])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x, x1


class Net32_2dAttLarge(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, se=False):
        super(Net32_2dAttLarge, self).__init__()
        self.conv1 = AttDoubleConv2d(in_channels, 32, se=se)
        self.conv11 = AttDoubleConv2d(32, 32, se=se)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = AttDoubleConv2d(32, 64, se=se)
        self.conv21 = AttDoubleConv2d(64, 64, se=se)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = AttDoubleConv2d(64, 128, se=se)
        # self.conv31 = DoubleConv2d(128, 128, se=se)
        self.fc1 = nn.Linear(8192, 128)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv21(x)
        x = self.pool2(x)
        x = self.conv3(x)
        # x = self.conv31(x)
        x = x.reshape([-1, 8192])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x, x1
        
        
class Net32_2dLarger(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, se=False):
        super(Net32_2dLarger, self).__init__()
        self.conv1 = DoubleConv2d(in_channels, 32, se=se)
        self.conv11 = DoubleConv2d(32, 32, se=se)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = DoubleConv2d(32, 64, se=se)
        self.conv21 = DoubleConv2d(64, 64, se=se)
        self.conv22 = DoubleConv2d(64, 64, se=se)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = DoubleConv2d(64, 128, se=se)
        self.conv31 = DoubleConv2d(128, 128, se=se)
        self.conv32 = DoubleConv2d(128, 128, se=se)
        self.fc1 = nn.Linear(8192, 128)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv21(x)
        x = self.conv22(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.conv31(x)
        x = self.conv32(x)
        x = x.reshape([-1, 8192])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x, x1


class Net2d3dMergeSK(nn.Module):
    def __init__(self, in_channels, n_classes, se=False, dropout_rate=0.0):
        super(Net2d3dMergeSK, self).__init__()
        self.net_2d = Net32_2d(in_channels, n_classes, se=se, dropout_rate=dropout_rate)
        self.net_3d = Net32SK(in_channels, n_classes, dropout_rate=dropout_rate)

        self.fc1 = nn.Linear(24576, 1000)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1000, n_classes)

    def forward(self, x):
        x0 = x[:, :, 15, :, :]
        x_2d, x1_2d = self.net_2d(x0)
        x_3d, x1_3d = self.net_3d(x)
        x1 = torch.cat([x1_2d, x1_3d], dim=1)
        x1 = self.fc1(x1)
        x1 = self.drop1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        return x1, x_2d, x_3d
        
        
class Net2d3dMergeLarge(nn.Module):
    def __init__(self, in_channels, n_classes, se=False, dropout_rate=0.0):
        super(Net2d3dMergeLarge, self).__init__()
        self.net_2d = Net32_2dLarge(in_channels, n_classes, se=se, dropout_rate=dropout_rate)
        self.net_3d = Net32SELarge(in_channels, n_classes, dropout_rate=dropout_rate)

        self.fc1 = nn.Linear(24576, 1000)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1000, n_classes)

    def forward(self, x):
        x0 = x[:, :, 15, :, :]
        x_2d, x1_2d = self.net_2d(x0)
        x_3d, x1_3d = self.net_3d(x)
        x1 = torch.cat([x1_2d, x1_3d], dim=1)
        x1 = self.fc1(x1)
        x1 = self.drop1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        return x1, x_2d, x_3d


class Net2d3dMergeAttLarge(nn.Module):
    def __init__(self, in_channels, n_classes, se=False, dropout_rate=0.0):
        super(Net2d3dMergeAttLarge, self).__init__()
        self.net_2d = Net32_2dAttLarge(in_channels, n_classes, se=se, dropout_rate=dropout_rate)
        self.net_3d = Net32SELarge(in_channels, n_classes, dropout_rate=dropout_rate)

        self.fc1 = nn.Linear(24576, 1000)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1000, n_classes)

    def forward(self, x):
        x0 = x[:, :, 15, :, :]
        x_2d, x1_2d = self.net_2d(x0)
        x_3d, x1_3d = self.net_3d(x)
        x1 = torch.cat([x1_2d, x1_3d], dim=1)
        x1 = self.fc1(x1)
        x1 = self.drop1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        return x1, x_2d, x_3d
        
        
class Net2d3dMergeLarge40(nn.Module):
    def __init__(self, in_channels, n_classes, se=False, dropout_rate=0.0):
        super(Net2d3dMergeLarge40, self).__init__()
        self.net_2d = Net40_2dLarge(in_channels, n_classes, se=se, dropout_rate=dropout_rate)
        self.net_3d = Net40SELarge(in_channels, n_classes, dropout_rate=dropout_rate)

        self.fc1 = nn.Linear(44800, 1400)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1400, n_classes)

    def forward(self, x):
        x0 = x[:, :, 15, :, :]
        x_2d, x1_2d = self.net_2d(x0)
        x_3d, x1_3d = self.net_3d(x)
        x1 = torch.cat([x1_2d, x1_3d], dim=1)
        x1 = self.fc1(x1)
        x1 = self.drop1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        return x1, x_2d, x_3d
        
        
class Net2d3dMergeLarge24(nn.Module):
    def __init__(self, in_channels, n_classes, se=False, dropout_rate=0.0):
        super(Net2d3dMergeLarge24, self).__init__()
        self.net_2d = Net24_2dLarge(in_channels, n_classes, se=se, dropout_rate=dropout_rate)
        self.net_3d = Net24SELarge(in_channels, n_classes, dropout_rate=dropout_rate)

        self.fc1 = nn.Linear(11520, 800)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(800, n_classes)

    def forward(self, x):
        x0 = x[:, :, 15, :, :]
        x_2d, x1_2d = self.net_2d(x0)
        x_3d, x1_3d = self.net_3d(x)
        x1 = torch.cat([x1_2d, x1_3d], dim=1)
        x1 = self.fc1(x1)
        x1 = self.drop1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        return x1, x_2d, x_3d
        
        
class Net2d3dMergeLarger(nn.Module):
    def __init__(self, in_channels, n_classes, se=False, dropout_rate=0.0):
        super(Net2d3dMergeLarger, self).__init__()
        self.net_2d = Net32_2dLarger(in_channels, n_classes, se=se, dropout_rate=dropout_rate)
        self.net_3d = Net32SELarger(in_channels, n_classes, dropout_rate=dropout_rate)

        self.fc1 = nn.Linear(24576, 1000)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1000, n_classes)

    def forward(self, x):
        x0 = x[:, :, 15, :, :]
        x_2d, x1_2d = self.net_2d(x0)
        x_3d, x1_3d = self.net_3d(x)
        x1 = torch.cat([x1_2d, x1_3d], dim=1)
        x1 = self.fc1(x1)
        x1 = self.drop1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        return x1, x_2d, x_3d


class Net32SELargeDilate(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, bn_track=True):
        super().__init__()
        self.conv1 = AttDoubleConv(in_channels, 16, se=True)
        self.conv11 = AttDoubleConv(16, 16, se=True)
        self.dilate1 = DilatedConvBlock(16)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = AttDoubleConv(16, 32, se=True)
        self.conv22 = AttDoubleConv(32, 32, se=True)
        self.dilate2 = DilatedConvBlock(32)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.fc1 = nn.Linear(16384, 1600)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1600, 65)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(65, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv11(x)
        x = self.dilate1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.conv22(x)
        x = self.dilate2(x)
        x = self.pool2(x)
        x = torch.reshape(x, [-1, 16384])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x, x1


class Net2d3dMergeLargeDilate(nn.Module):
    def __init__(self, in_channels, n_classes, se=False, dropout_rate=0.0):
        super().__init__()
        self.net_2d = Net32_2dLarge(in_channels, n_classes, se=se, dropout_rate=dropout_rate)
        self.net_3d = Net32SELargeDilate(in_channels, n_classes, dropout_rate=dropout_rate)

        self.fc1 = nn.Linear(24576, 1000)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(1000, n_classes)

    def forward(self, x):
        x0 = x[:, :, 15, :, :]
        x_2d, x1_2d = self.net_2d(x0)
        x_3d, x1_3d = self.net_3d(x)
        x1 = torch.cat([x1_2d, x1_3d], dim=1)
        x1 = self.fc1(x1)
        x1 = self.drop1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        return x1, x_2d, x_3d


if __name__ == '__main__':
    a = torch.randn((32, 1, 32, 32, 32))
    b = Net32SK(1, 2)
    c = b(a)
