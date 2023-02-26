import torch
from torch import nn
import numpy as np


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

        
class Net32SE(nn.Module):
    def __init__(self, in_channels, n_classes, dropout_rate=0.0, base_num_channels=16, fc1_channels=1600, fc2_channels=65, double_conv=False, bn_track=True):
        super().__init__()
        b = base_num_channels
        self.b = b
        if double_conv:
            self.conv1 = nn.Sequential(AttDoubleConv(in_channels, b, se=True), AttDoubleConv(b, b, se=True))
            self.pool1 = nn.MaxPool3d(2, 2)
            self.conv2 = nn.Sequential(AttDoubleConv(b, 2 * b, se=True), AttDoubleConv(2 * b, 2 * b, se=True))
            self.pool2 = nn.MaxPool3d(2, 2)
        else:
            self.conv1 = AttDoubleConv(in_channels, b, se=True)
            self.pool1 = nn.MaxPool3d(2, 2)
            self.conv2 = AttDoubleConv(b, 2 * b, se=True)
            self.pool2 = nn.MaxPool3d(2, 2)
        self.fc1 = nn.Linear(1024 * b, fc1_channels)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc1_channels, fc2_channels)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(fc2_channels, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.reshape(x, [-1, self.b * 1024])
        x1 = x
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x, x1


class Net32SEAux(nn.Module):
    def __init__(self, in_channels, n_classes, num_aux, dropout_rate=0.0, base_num_channels=16, fc1_channels=1600, fc2_channels=65, double_conv=False, bn_track=True):
        super().__init__()
        b = base_num_channels
        self.b = b
        if double_conv:
            self.conv1 = nn.Sequential(AttDoubleConv(in_channels, b, se=True), AttDoubleConv(b, b, se=True))
            self.pool1 = nn.MaxPool3d(2, 2)
            self.conv2 = nn.Sequential(AttDoubleConv(b, 2 * b, se=True), AttDoubleConv(2 * b, 2 * b, se=True))
            self.pool2 = nn.MaxPool3d(2, 2)
        else:
            self.conv1 = AttDoubleConv(in_channels, b, se=True)
            self.pool1 = nn.MaxPool3d(2, 2)
            self.conv2 = AttDoubleConv(b, 2 * b, se=True)
            self.pool2 = nn.MaxPool3d(2, 2)
        self.fc1 = nn.Linear(1024 * b, fc1_channels)
        self.drop1 = nn.Dropout(p=dropout_rate)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(fc1_channels, fc2_channels)
        self.drop2 = nn.Dropout(p=dropout_rate)
        self.fc3 = nn.Linear(fc2_channels, n_classes)
        self.fc21 = nn.Linear(fc1_channels, fc2_channels)
        self.drop21 = nn.Dropout(p=dropout_rate)
        self.fc31 = nn.Linear(fc2_channels, num_aux)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = torch.reshape(x, [-1, self.b * 1024])
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.relu(x)
        x1 = x
        x = self.fc2(x)
        x = self.drop2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x1 = self.fc21(x1)
        x1 = self.drop2(x1)
        x1 = self.relu(x1)
        x1 = self.fc31(x1)
        return x, x1


class Net32SETextureAux(nn.Module):
    def __init__(self, in_channels, num_aux, reg_dropout=0.0, cls_dropout=0.0, base_num_channels=16, fc1_channels=1600, fc2_channels=65, double_conv=False):
        super().__init__()
        self.reg = Net32SEAux(in_channels, 1, num_aux, dropout_rate=reg_dropout, base_num_channels=base_num_channels, fc1_channels=fc1_channels, fc2_channels=fc2_channels, double_conv=double_conv)
        self.cls = Net32SE(in_channels, 2, dropout_rate=cls_dropout, base_num_channels=base_num_channels, fc1_channels=fc1_channels, fc2_channels=fc2_channels, double_conv=double_conv)

    def forward(self, x):
        reg_res = self.reg(x)
        cls_res = self.cls(x)[0]
        return reg_res, cls_res