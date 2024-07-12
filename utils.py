# This code file mainly includes the functions used by the network, including loss function, evaluation metric, normalization, and dataset construction.

import numpy as np
import torch
import torch.nn as nn
import gdal
import math
import torchvision


class StyleLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(StyleLoss, self).__init__()
        self.eps = eps
        self.mse_loss = nn.MSELoss()

    def calc_mean_std(self, x, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = x.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = x.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = x.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def forward(self, x, y):
        mean_x, std_x = self.calc_mean_std(x)
        mean_y, std_y = self.calc_mean_std(y)
        loss = self.mse_loss(mean_x, mean_y) + self.mse_loss(std_x, std_y)
        return loss


class VGG19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        if X.size(1) == 1:
            X = X.repeat(1, 3, 1, 1)
        h_relu1 = self.slice1(X)  # torch.Size([1, 64, 256, 256])
        h_relu2 = self.slice2(h_relu1)  # torch.Size([1, 128, 128, 128])
        h_relu3 = self.slice3(h_relu2)  # torch.Size([1, 256, 64, 64])
        h_relu4 = self.slice4(h_relu3)  # torch.Size([1, 512, 32, 32])
        h_relu5 = self.slice5(h_relu4)  # torch.Size([1, 512, 16, 16])
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()

        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)

        vgg = torchvision.models.vgg19(pretrained=True).features

        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = StyleLoss().to(device)

    def forward(self, source, target):
        loss = 0
        if source.size(1) == 1:
            source = source.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight * self.criterion(source, target)

        return loss


class CharbonnierLoss(nn.Module):
    "Charbonnier Loss (L1)"

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        # loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps * self.eps)))
        return loss


def readTif(dataset):
    data = gdal.Open(dataset)
    if dataset == None:
        print(data + "文件无法打开")
    return data


def writeTiff(im_data, im_geotrans, im_proj, path):
    datatype = gdal.GDT_Float32
    im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset


def MAE(img1, img2):
    # MAE_test = np.nanmean(np.abs(y_true-y_pred))
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    MAE = np.sum(np.mean(np.abs(img1_np - img2_np)))
    return MAE


def RMSE(img1, img2):
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    MSE = np.sum(np.mean((img1_np - img2_np) ** 2))
    RMSE = MSE ** 0.5
    return RMSE


def PSNR(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    max = np.max(img2)
    mse = np.sum(np.mean((img1 / 1.0 - img2 / 1.0) ** 2))
    if mse == 0:
        return float('inf')
    else:
        PSNR = 20 * (math.log10(max / mse ** 0.5))
        return PSNR


def MEE(img1, img2):
    # MAE_test = np.nanmean(np.abs(y_true-y_pred))
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    MEE = np.sum(np.max(np.abs(img1_np - img2_np)))
    return MEE

