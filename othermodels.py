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


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y1 = self.avg_pool(x)
        # y2 = self.max_pool(x)
        y = self.conv_du(y1)
        y = self.sigmoid(y)
        return x*y


# Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, channel, reduction=16):
        super(RCAB, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        )
        self.atte = CALayer(channel)

    def forward(self, x):
        res = self.head(x)
        res = self.atte(res)
        res += x
        return res


# class AB(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(AB, self).__init__()
#         self.head = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#         )
#         self.pixel = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(channel // reduction, 1, 1, padding=0, bias=True),
#         )
#         self.atte = nn.Sigmoid()
#         self.ca = CALayer(channel)
#
#     def forward(self, x):
#         res = self.head(x)
#
#         res_pixel = self.pixel(res)
#         res_pixel = self.atte(res_pixel)
#         res = x*res_pixel
#         res_channel = self.ca(res)
#         res = res_channel+x
#         return res

#
class AB(nn.Module):
    def __init__(self, channel, reduction=16):
        super(AB, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 3, padding=1, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel // reduction, channel, 3, padding=1, bias=True),
        )
        # self.pixel = nn.Sequential(
        #     nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(channel // reduction, 1, 1, padding=0, bias=True),
        # )
        # self.atte = nn.Sigmoid()
        # self.ca = CALayer(channel)
        self.cbam = CBAMLayer(channel)

    def forward(self, x):
        res = self.head(x)
        res = self.cbam(res)
        res = res+x
        return res


class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()

        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )

        # spatial attention
        self.conv = nn.Conv2d(1, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(avg_out))
        # spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x

        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        return x


# class AB(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(AB, self).__init__()
#         self.head = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
#         )
#         self.pixel = nn.Sequential(
#             nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
#             nn.LeakyReLU(0.1, inplace=True),
#             nn.Conv2d(channel // reduction, 1, 1, padding=0, bias=True),
#         )
#         self.atte = nn.Sigmoid()
#         self.ca = CALayer(channel)
#
#     def forward(self, x):
#         res = self.head(x)
#         res_pixel = self.pixel(res)
#         res_pixel = self.atte(res_pixel)
#         res_channel = self.ca(res)
#         res = x*res_pixel*res_channel
#         res += x
#         return res


# Residual Pixel Attention Block (RCAB)
class RPAB(nn.Module):
    def __init__(self, channels):
        super(RPAB, self).__init__()
        self.rb = nn.Sequential(
            nn.Conv2d(channels, channels // 16, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels // 16, channels, 3, 1, 1),
        )

        self.encode = nn.Sequential(
            # nn.Conv2d(channels, channels, 1, 1, 0),
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


