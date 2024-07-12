# This code file is used for the testing of the TGF-RSAnet, run this file if you want to test the network
import argparse
import csv
import sys
import tifffile as tiff
from matplotlib import pyplot as plt
from statsmodels import robust

sys.path.append("..")
import math
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from utils import *
from torch.nn import functional as F
import torchvision.datasets as datasets
import matplotlib as mpl
from skimage import io
import random
import utils
import os
from osgeo import gdal

os.environ['PROJ_LIB'] = r'D:\ProgramData\Anaconda3\envs\pytorch\Library\share\proj'
import time
######################################################
from DeviceSetting import device
from model import RSnet as srnet
from DEM_features import Slope_net, Aspect_net

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=144, help='the high resolution image size')
parser.add_argument('--upSampling', type=int, default=3, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--dataroot', type=str, default=r'', help='path to dataset')
parser.add_argument('--rsroot', type=str, default=r'', help='path to rs dataset')
parser.add_argument('--lr_root', type=str, default=r'', help='path to lr_dataset')
parser.add_argument('--netweight', type=str, default='', help="path to generator weights (to continue training)")
opt = parser.parse_args()

# print(opt)

MAE_list = []
RMSE_list = []
RMSE_Slope = []
MEE_list = []

dataset = datasets.ImageFolder(root=opt.dataroot)
assert dataset

rs_dataset = datasets.ImageFolder(root=opt.rsroot)
assert rs_dataset

lr_dataset = datasets.ImageFolder(root=opt.lr_root)
assert lr_dataset

model = srnet().to(device)

if opt.netweight != '':
    model.load_state_dict(torch.load(opt.netweight, map_location='cuda:0'))
# print(model)

content_criterion = nn.MSELoss().to(device)

original_dem = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # dem
original_rs = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)
low_dem = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)
low_res = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # low resolution dem

imgs_count = len(dataset.imgs)  # training dems count
batchSize_count = imgs_count // opt.batchSize  # number of batchSize in each epoch

model.eval()
random_ids = list(range(0, imgs_count))
print('testing')
with torch.no_grad():
    for epoch in range(0, opt.nEpochs):
        for i in range(0, batchSize_count):  # batchSize_count):
            # get a batchsize of rivers and dems
            for j in range(opt.batchSize):
                rs_temp, _ = rs_dataset.imgs[random_ids[i * opt.batchSize + j]]  # get one high-resolution rs image
                rs_temp = io.imread(rs_temp)
                rs_min = np.min(rs_temp)
                rs_max = np.max(rs_temp)
                rs_temp = 2 * (rs_temp - rs_min) / (rs_max - rs_min + 10) - 1
                rs_temp = torch.tensor(rs_temp)
                original_rs[j] = rs_temp

                # low-resolution image
                low_img_temp, _ = lr_dataset.imgs[random_ids[i * opt.batchSize + j]]  # get one high-resolution image
                low_img_temp = io.imread(low_img_temp)
                low_img_temp = np.array(low_img_temp, dtype=np.float32)
                base_min = np.min(low_img_temp)
                base_max = np.max(low_img_temp)
                low_dem[j] = torch.tensor(low_img_temp)
                low_img_temp = 2 * (low_img_temp - base_min) / (base_max - base_min + 10) - 1
                low_res[j] = torch.tensor(low_img_temp)

                # high-resolution image
                img_temp, _ = dataset.imgs[random_ids[i * opt.batchSize + j]]  # get one high-resolution image

                img_temp = io.imread(img_temp)
                img_temp = np.array(img_temp, dtype=np.float32)
                H, W = img_temp.shape
                original_dem[j] = torch.tensor(img_temp)

            # fake inputs
            high_res_fake = model(low_res.to(device), original_rs.to(device), is_training=0)
            high_res_fake = ((high_res_fake + 1) * (base_max - base_min + 10)) / 2 + base_min
            #
            high_slope = Slope_net(original_dem)
            fake_slope = Slope_net(high_res_fake)
            bicubic_slope = Slope_net(low_dem)

            # eval
            RMSE_list.append(RMSE(high_res_fake, original_dem))
            MAE_list.append(MAE(high_res_fake, original_dem))
            RMSE_Slope.append(RMSE(high_slope, fake_slope))
            MEE_list.append(MEE(high_res_fake, original_dem))

    print(' mean RMSE: ', float(np.array(RMSE_list).mean()))
    print(' mean MAE: ', float(np.array(MAE_list).mean()))
    print(' mean RMSE_Slope: ', float(np.array(RMSE_Slope).mean()))
    print(' mean MEE: ', float(np.array(MEE_list).mean()))


