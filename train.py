# This code file is used for the training of the TGF-RSAnet

import argparse
import os
import sys
sys.path.append("..")
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from utils import CharbonnierLoss
import torch.backends.cudnn
import torchvision.datasets as datasets
import kornia
from skimage import io
import random
######################################################
from DeviceSetting import device
from model import RSnet as srnet
from DEM_features import Slope_net, Aspect_net

seed = 20
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=1, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=144, help='the high resolution image size')
parser.add_argument('--upSampling', type=int, default=3, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate for generator')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--dataroot', type=str, default=r'', help='path to dataset')
parser.add_argument('--rsroot', type=str, default=r'', help='path to rs dataset')
parser.add_argument('--lrroot', type=str, default=r'', help='path to rs dataset')
parser.add_argument('--netweight', type=str,default='',  help="path to generator weights (to continue training)")
parser.add_argument('--out', type=str, default='', help='folder to output model checkpoints')


opt = parser.parse_args()

try:
    os.makedirs(opt.out)
except OSError:
    pass


dataset = datasets.ImageFolder(root=opt.dataroot)
assert dataset

rs_dataset = datasets.ImageFolder(root=opt.rsroot)
assert rs_dataset
#
lr_dataset = datasets.ImageFolder(root=opt.lrroot)
assert lr_dataset

model = srnet().to(device)

if opt.netweight != '':
    model.load_state_dict(torch.load(opt.netweight))
# print(model)

optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.nEpochs, eta_min=2e-5)
content_criterion = CharbonnierLoss().to(device)

high_res_real = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # high resolution dem
low_res = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # low resolution dem

original_rs = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # rs
original_dem = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)  # dem
fake_dem = torch.FloatTensor(opt.batchSize, 1, opt.imageSize, opt.imageSize).to(device)


imgs_count = len(dataset.imgs)  # training dems count
batchSize_count = imgs_count // opt.batchSize  # number of batchSize in each epoch

loss_list = []
edge_loss = []
content_loss = []
epoch_num = []

model.train()
random_ids = list(range(0, imgs_count))
random.shuffle(random_ids)  # shuffle these dems
print('river training')
for epoch in range(0, opt.nEpochs):
    for i in range(batchSize_count):
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
            base_min = np.min(low_img_temp)
            base_max = np.max(low_img_temp)
            low_img_temp = 2 * (low_img_temp - base_min) / (base_max - base_min + 10) - 1
            low_res[j] = torch.tensor(low_img_temp)
            low_slope = Slope_net(low_res)
            slope_max, slope_min = torch.max(low_slope), torch.min(low_slope)

            # high-resolution image
            img_temp, _ = dataset.imgs[random_ids[i * opt.batchSize + j]]  # get one high-resolution image
            img_temp = io.imread(img_temp)
            H, W = img_temp.shape
            original_dem[j] = torch.tensor(img_temp)
            img_temp = 2 * (img_temp - base_min) / (base_max - base_min + 10) - 1
            high_res_real[j] = torch.tensor(img_temp)  # -1~1

        # Generate real and fake inputs
        high_res_real = Variable(high_res_real.to(device))
        high_res_fake, grad_fusion, grad_rs = model(low_res.to(device), original_rs.to(device), is_training=1)
        sr_dem = ((high_res_fake + 1) * (base_max - base_min + 10)) / 2 + base_min
        ######### Train generator #########
        optimizer.zero_grad()
        high_slope = Slope_net(high_res_real)
        fake_slope = Slope_net(high_res_fake)
        fake_slope = 2 * (fake_slope - slope_min) / (slope_max - slope_min + 10) - 1
        high_slope = 2 * (high_slope - slope_min) / (slope_max - slope_min + 10) - 1
        high_aspect = Aspect_net(high_res_real)
        fake_aspect = Aspect_net(high_res_fake)

        high_sobel = kornia.filters.sobel(high_res_real)
        rs_sobel = kornia.filters.sobel(original_rs)

        generator_content_loss = content_criterion(high_res_fake.to(device), high_res_real.to(device))
        generator_edge_loss = content_criterion(high_slope.to(device), fake_slope.to(device)) + 0.0001*content_criterion(high_aspect.to(device), fake_aspect.to(device))
        generator_identity_loss = content_criterion(grad_rs.to(device), rs_sobel.to(device)) + content_criterion(grad_fusion.to(device), high_sobel.to(device))
        generator_total_loss = generator_content_loss + generator_edge_loss + generator_identity_loss
        generator_total_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)
        optimizer.step()
        scheduler.step()
        epoch_num.append(epoch + 1)
        loss_list.append(generator_total_loss.data.cpu())
        edge_loss.append(generator_edge_loss.data.cpu())
        content_loss.append(generator_content_loss.data.cpu())

    print('Epoch----%5d, loss---%f, edge_loss---%f, content_loss---%f' % (epoch + 1, float(np.array(loss_list).mean()), float(np.array(edge_loss).mean()), float(np.array(content_loss).mean())))

    # if epoch == 0 or (epoch+1) % 10 == 0:
    torch.save(model.state_dict(), '%s/rsanet_x%03d_%03d.pth' % (opt.out, opt.upSampling, epoch))





