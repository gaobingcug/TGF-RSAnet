import numpy as np
import torch
import torch.nn as nn
from deform_conv import DeformConv2d


class Begin(nn.Module):
    def __init__(self, input, channels):
        super(Begin, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels//16, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels//16, channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
        )

    def forward(self, x):
        # 第l层
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        return x4


class Registration(nn.Module):
    def __init__(self, channels):
        super(Registration, self).__init__()

        self.conv_down_1 = nn.Conv2d(2*channels, 2*channels, 3, 2, 1)
        self.conv1 = nn.Conv2d(2*channels, channels, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(2*channels, channels, 3, 1, 1)
        self.conv_up_1 = nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=1)
        self.conv_down_rs = nn.Conv2d(channels, channels, 3, 2, 1)
        self.conv_offset = nn.Conv2d(channels, channels, 1, 1, 0)

        self.conv3 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.deconv1 = DeformConv2d(channels, channels, 3, 1, 1)
        self.deconv2 = DeformConv2d(channels, channels, 3, 1, 1)
        self.conv_up_2 = nn.ConvTranspose2d(in_channels=channels, out_channels=channels, kernel_size=4, stride=2, padding=1)

    def forward(self, dem, rs):

        fea_1 = torch.cat([dem, rs], 1)
        fea_2 = self.conv_down_1(fea_1)
        fea_1 = self.conv1(fea_1)
        fea_2 = self.conv1_2(fea_2)
        fea_2_1 = self.conv_up_1(fea_2)
        fea = self.conv_offset(fea_1 + fea_2_1)

        # print(right_1)
        fea_1_1 = self.deconv1(rs, fea)
        rs_2 = self.conv_down_rs(rs)
        fea_2 = self.deconv2(rs_2, fea_2)
        fea_2_1 = self.conv_up_2(fea_2)

        registration_right = fea_1_1 + fea_2_1
        registration_right = self.conv3(registration_right)
        return registration_right


class ResB(nn.Module):
    def __init__(self, channel):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, channel, 3, 1, 1),
        )

    def __call__(self, x):
        out = self.body(x)
        return out + x


class AB(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AB, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel // reduction, channel, 3, padding=1, bias=True),
        )
        self.scab = SCABLayer(channel)

    def forward(self, x):
        res = self.head(x)
        res = self.scab(res)
        res = res+x
        return res


class SCABLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(SCABLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(1, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(avg_out))
        x = spatial_out * x

        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        return x


class RPAB(nn.Module):
    def __init__(self, channels):
        super(RPAB, self).__init__()
        self.rb = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels // 16, channels, 3, 1, 1),
        )

        self.encode = nn.Sequential(
            nn.Conv2d(channels, channels//16, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels//16, channels, 1, 1, 0),
        )
        self.sig = nn.Sigmoid()

    def forward(self, dem, rs, is_training):

        buffer_dem = self.encode(dem)
        buffer_rs = self.rb(rs)
        rs_to_dem = self.sig(buffer_dem)
        buffer_1 = buffer_rs * rs_to_dem
        buffer = buffer_1 + dem

        if is_training == 1:
            buffer_dem = self.rb(rs)
            buffer_rs = self.encode(rs)
            dem_to_rs = self.sig(buffer_rs)
            buffer_2 = buffer_dem * dem_to_rs
            buffer_rs = buffer_2 + rs

            return buffer, buffer_rs

        if is_training == 0:
            return buffer


