import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np
import math
from DeviceSetting import device


class Slope(nn.Module):
    def __init__(self):
        super(Slope, self).__init__()
        weight1 = np.zeros(shape=(3, 3), dtype=np.float32)
        weight2 = np.zeros(shape=(3, 3), dtype=np.float32)

        weight1[0][0] = 1
        weight1[0][1] = math.sqrt(2)
        weight1[0][2] = 1
        weight1[1][0] = 0
        weight1[1][1] = 0
        weight1[1][2] = 0
        weight1[2][0] = -1
        weight1[2][1] = -math.sqrt(2)
        weight1[2][2] = -1

        weight2[0][0] = -1
        weight2[0][1] = 0
        weight2[0][2] = 1
        weight2[1][0] = -math.sqrt(2)
        weight2[1][1] = 0
        weight2[1][2] = math.sqrt(2)
        weight2[2][0] = -1
        weight2[2][1] = 0
        weight2[2][2] = 1

        weight1 = np.reshape(weight1, (1, 1, 3, 3))
        weight2 = np.reshape(weight2, (1, 1, 3, 3))
        weight1 = weight1 / ((4+2*math.sqrt(2)) * 10)
        weight2 = weight2 / ((4+2*math.sqrt(2)) * 10)
        self.weight1 = nn.Parameter(torch.tensor(weight1))
        self.weight2 = nn.Parameter(torch.tensor(weight2))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        dx = F.conv2d(x, self.weight1, self.bias, stride=1, padding=1)
        dy = F.conv2d(x, self.weight2, self.bias, stride=1, padding=1)
        ij_slope = torch.sqrt(torch.pow(dx, 2) + torch.pow(dy, 2))
        ij_slope = torch.arctan(ij_slope) * 180 / math.pi
        return ij_slope


class Aspect(nn.Module):
    def __init__(self):
        super(Aspect, self).__init__()
        weight1 = np.zeros(shape=(3, 3), dtype=np.float32)
        weight2 = np.zeros(shape=(3, 3), dtype=np.float32)

        weight1[0][0] = 1
        weight1[0][1] = math.sqrt(2)
        weight1[0][2] = 1
        weight1[1][0] = 0
        weight1[1][1] = 0
        weight1[1][2] = 0
        weight1[2][0] = -1
        weight1[2][1] = -math.sqrt(2)
        weight1[2][2] = -1

        weight2[0][0] = -1
        weight2[0][1] = 0
        weight2[0][2] = 1
        weight2[1][0] = -math.sqrt(2)
        weight2[1][1] = 0
        weight2[1][2] = math.sqrt(2)
        weight2[2][0] = -1
        weight2[2][1] = 0
        weight2[2][2] = 1

        weight1 = np.reshape(weight1, (1, 1, 3, 3))
        weight2 = np.reshape(weight2, (1, 1, 3, 3))
        weight1 = weight1 / (8)
        weight2 = weight2 / (8)
        self.weight1 = nn.Parameter(torch.tensor(weight1))
        self.weight2 = nn.Parameter(torch.tensor(weight2))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        dx = F.conv2d(x, self.weight1, self.bias, stride=1, padding=1) + 1e-8
        dy = F.conv2d(x, self.weight2, self.bias, stride=1, padding=1) + 1e-8

        aspect = 57.29578 * torch.atan2(dy, -dx)
        aspect = torch.where(aspect > 90, 360 - aspect + 90, 90 - aspect)
        return aspect


Slope_net = Slope().to(device)
Aspect_net = Aspect().to(device)
