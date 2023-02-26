import torch
import torch.nn as nn
import sys
from false_positive_reduction.net.net_2d_3d_combine import AttDoubleConv, DoubleConv2d


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels, conv_times, se):
        super().__init__()
        self.conv_lst = []
        self.conv_lst.append(AttDoubleConv(in_channels, out_channels, se=se))
        for _ in range(1, conv_times):
            self.conv_lst.append(AttDoubleConv(out_channels, out_channels, se=se))
        self.conv = nn.Sequential(*self.conv_lst)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvBlock2d(nn.Module):
    def __init__(self, in_channels, out_channels, conv_times, se):
        super().__init__()
        self.conv_lst = []
        self.conv_lst.append(DoubleConv2d(in_channels, out_channels, se=se))
        for _ in range(1, conv_times):
            self.conv_lst.append(DoubleConv2d(out_channels, out_channels, se=se))
        self.conv = nn.Sequential(*self.conv_lst)

    def forward(self, x):
        x = self.conv(x)
        return x

class UNet3d(nn.Module):
    def __init__(self, in_channels, n_classes_cls, n_classes_seg, base_num_channel=16, 
    fc1_channels=1600, fc2_channels=65, se=True, dropout_rate=0.0, conv_times=2):
        super().__init__()
        b = base_num_channel
        self.b = b
        self.conv1 = ConvBlock3d(in_channels, b, conv_times=conv_times, se=se)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = ConvBlock3d(b, 2 * b, conv_times=conv_times, se=se)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.conv3 = ConvBlock3d(2 * b, 2 * b, conv_times=conv_times, se=se)
        self.up2 = nn.ConvTranspose3d(2 * b, 2 * b, kernel_size=2, stride=2)
        self.conv4 = ConvBlock3d(4 * b, b, conv_times=conv_times, se=se)
        self.up1 = nn.ConvTranspose3d(b, b, kernel_size=2, stride=2)
        self.conv5 = ConvBlock3d(2 * b, b, conv_times=conv_times, se=se)
        self.outconv = nn.Conv3d(b, n_classes_seg, kernel_size=1)

        self.fc1 = nn.Linear(2 * b * 512, fc1_channels)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc1_channels, fc2_channels)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(fc2_channels, n_classes_cls)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x2 = self.conv2(x2)
        x3 = self.pool2(x2)
        x31 = x3
        x3 = self.conv3(x3)
        x4 = self.up2(x3)
        x4 = torch.cat([x2, x4], dim=1)
        x4 = self.conv4(x4)
        x5 = self.up1(x4)
        x5 = torch.cat([x1, x5], dim=1)
        x5 = self.conv5(x5)
        seg_out = self.outconv(x5)

        x31 = torch.reshape(x31, [-1, 2 * self.b * 512])
        x_sideout = x31
        x31 = self.fc1(x31)
        x31 = self.drop1(x31)
        x31 = self.relu(x31)
        x31 = self.fc2(x31)
        x31 = self.drop2(x31)
        x31 = self.relu(x31)
        cls_out = self.fc3(x31)

        return cls_out, seg_out, x_sideout


class UNet2d(nn.Module):
    def __init__(self, in_channels, n_classes_cls, n_classes_seg, base_num_channel=32, 
    fc_channel=128, se=True, dropout_rate=0.0, conv_times=2):
        super().__init__()
        b = base_num_channel
        self.b = b
        self.conv1 = ConvBlock2d(in_channels, b, conv_times=conv_times, se=se)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = ConvBlock2d(b, 2 * b, conv_times=conv_times, se=se)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = ConvBlock2d(2 * b, 4 * b, conv_times=conv_times, se=se)
        self.up1 = nn.ConvTranspose2d(4 * b, 4 * b, kernel_size=2, stride=2)
        self.conv4 = ConvBlock2d(6 * b, b, conv_times=conv_times, se=se)
        self.up2 = nn.ConvTranspose2d(b, b, kernel_size=2, stride=2)
        self.conv5 = ConvBlock2d(2 * b, b, conv_times=conv_times, se=se)
        self.outconv = nn.Conv2d(b, n_classes_seg, kernel_size=1)

        self.fc1 = nn.Linear(256 * b, fc_channel)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc_channel, n_classes_cls)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x2 = self.conv2(x2)
        x3 = self.pool2(x2)
        x3 = self.conv3(x3)
        x31 = x3
        x4 = self.up1(x3)
        x4 = torch.cat([x2, x4], dim=1)
        x4 = self.conv4(x4)
        x5 = self.up2(x4)
        x5 = torch.cat([x1, x5], dim=1)
        x5 = self.conv5(x5)
        seg_out = self.outconv(x5)

        x31 = torch.reshape(x31, [-1, self.b * 256])
        x_sideout = x31
        x31 = self.fc1(x31)
        x31 = self.drop1(x31)
        x31 = self.relu(x31)
        x31 = self.fc2(x31)
        cls_out = x31

        return cls_out, seg_out, x_sideout


class NetCombineClsSeg(nn.Module):
    def __init__(self, in_channels, n_classes_cls, n_classes_seg, b3=16, b2=32, fc13=1600, fc23=65, 
    fc12=128, fc=1000, se=True, dropout_rate=0.0, leak=False, conv_times_3d=2, conv_times_2d=2):
        super().__init__()
        self.unet3d = UNet3d(in_channels, n_classes_cls, n_classes_seg, base_num_channel=b3, fc1_channels=fc13, fc2_channels=fc23, se=se, dropout_rate=dropout_rate, conv_times=conv_times_3d)
        self.unet2d = UNet2d(in_channels, n_classes_cls, n_classes_seg, base_num_channel=b2, fc_channel=fc12 ,se=se, dropout_rate=dropout_rate, conv_times=conv_times_2d)

        self.fc1 = nn.Linear(1024 * b3 + 256 * b2, fc)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc, n_classes_cls)

        self.leak = leak

    def forward(self, x):
        x0 = x[:, :, 15, :, :]
        cls_2d, seg_2d, x1_2d = self.unet2d(x0)
        cls_3d, seg_3d, x1_3d = self.unet3d(x)
        x1 = torch.cat([x1_2d, x1_3d], dim=1)
        x1 = self.fc1(x1)
        x_leak = x1
        x1 = self.drop1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        if self.leak:
            return x1, cls_2d, cls_3d, seg_2d, seg_3d, x_leak
        return x1, cls_2d, cls_3d, seg_2d, seg_3d
    

class NetCombineClsSegMod(nn.Module):
    def __init__(self, in_channels, n_classes_cls, n_classes_seg, b3=16, b2=32, fc13=1600, fc23=65, 
    fc12=128, fc=1000, se=True, dropout_rate=0.0, leak=False, conv_times_3d=2, conv_times_2d=2, mod=0):
        super().__init__()
        self.unet3d = UNet3dMod(in_channels, n_classes_cls, n_classes_seg, base_num_channel=b3, fc1_channels=fc13, fc2_channels=fc23, se=se, dropout_rate=dropout_rate, conv_times=conv_times_3d, mod=mod)
        self.unet2d = UNet2d(in_channels, n_classes_cls, n_classes_seg, base_num_channel=b2, fc_channel=fc12 ,se=se, dropout_rate=dropout_rate, conv_times=conv_times_2d)

        self.fc1 = nn.Linear(4096 + 256 * b2, fc)
        if mod == 1:
            self.fc1 = nn.Linear(b3 * 512 + b2 * 256, fc)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc, n_classes_cls)

        self.leak = leak

    def forward(self, x):
        x0 = x[:, :, 15, :, :]
        cls_2d, seg_2d, x1_2d = self.unet2d(x0)
        cls_3d, seg_3d, x1_3d = self.unet3d(x)
        x1 = torch.cat([x1_2d, x1_3d], dim=1)
        x1 = self.fc1(x1)
        x_leak = x1
        x1 = self.drop1(x1)
        x1 = self.relu(x1)
        x1 = self.fc2(x1)
        if self.leak:
            return x1, cls_2d, cls_3d, seg_2d, seg_3d, x_leak
        return x1, cls_2d, cls_3d, seg_2d, seg_3d


class UNet3dMod(nn.Module):
    def __init__(self, in_channels, n_classes_cls, n_classes_seg, base_num_channel=16, 
    fc1_channels=1600, fc2_channels=65, se=True, dropout_rate=0.0, conv_times=2, mod=0):
        super().__init__()
        b = base_num_channel
        self.m = mod
        self.b = b
        self.conv1 = ConvBlock3d(in_channels, b, conv_times=conv_times, se=se)
        self.pool1 = nn.MaxPool3d(2, 2)
        self.conv2 = ConvBlock3d(b, 2 * b, conv_times=conv_times, se=se)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.conv3 = ConvBlock3d(2 * b, 2 * b, conv_times=conv_times, se=se)
        self.up2 = nn.ConvTranspose3d(2 * b, 2 * b, kernel_size=2, stride=2)
        self.conv4 = ConvBlock3d(4 * b, b, conv_times=conv_times, se=se)
        self.up1 = nn.ConvTranspose3d(b, b, kernel_size=2, stride=2)
        self.conv5 = ConvBlock3d(2 * b, b, conv_times=conv_times, se=se)
        self.outconv = nn.Conv3d(b, n_classes_seg, kernel_size=1)

        self.bd1 = nn.Conv3d(b, 1, kernel_size=1)

        self.fc1 = nn.Linear(4096, fc1_channels)
        if mod == 1:
            self.bd1 = nn.Sequential(nn.Conv3d(b, b, kernel_size=3, padding=1), nn.MaxPool3d(2, 2))
            self.fc1 = nn.Linear(b * 512, fc1_channels)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc1_channels, fc2_channels)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(fc2_channels, n_classes_cls)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.pool1(x1)
        x2 = self.conv2(x2)
        x3 = self.pool2(x2)
        x3 = self.conv3(x3)
        x4 = self.up2(x3)
        x4 = torch.cat([x2, x4], dim=1)
        x4 = self.conv4(x4)
        x41 = x4
        x5 = self.up1(x4)
        x5 = torch.cat([x1, x5], dim=1)
        x5 = self.conv5(x5)
        seg_out = self.outconv(x5)

        x31 = self.bd1(x41)
        if self.m == 1:
            x31 = torch.reshape(x31, [-1, self.b * 512])
        else:
            x31 = torch.reshape(x31, [-1, 4096])
        x_sideout = x31
        x31 = self.fc1(x31)
        x31 = self.drop1(x31)
        x31 = self.relu(x31)
        x31 = self.fc2(x31)
        x31 = self.drop2(x31)
        x31 = self.relu(x31)
        cls_out = self.fc3(x31)

        return cls_out, seg_out, x_sideout