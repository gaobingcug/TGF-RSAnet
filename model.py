import kornia
from mmcv.ops.modulated_deform_conv import ModulatedDeformConv2dPack as DCN
from othermodels import *
from DEM_features import Slope_net


class RSnet(nn.Module):
    def __init__(self):
        super(RSnet, self).__init__()

        # feature extraction
        self.begin_dem = Begin(1, 64)
        self.begin_rs = Begin(1, 64)

        # registration
        self.registration = Registration(64)

        # content branch
        for i in range(4):
            self.add_module('AB' + str(i + 1), AB(128))
        self.conv_content = nn.Conv2d(64*2, 1, 1, 1, 0, bias=True)

        # edge branch
        # self.begin_edge_dem = nn.Conv2d(1, 64, 3, 1, 1, bias=True)
        # self.begin_edge_rs = nn.Conv2d(1, 64, 3, 1, 1, bias=True)
        self.fusion_edge_1 = RPAB(64)
        self.conv_edge = nn.Conv2d(64, 1, 1, 1, 0, bias=True)

        # fusion branch
        self.conv_fusion = nn.Conv2d(2, 64, 3, 1, 1, bias=True)
        for i in range(10):
            self.add_module('fusion' + str(i + 1), ResB(64))

        self.end = nn.Conv2d(64, 1, 3, 1, 1, bias=True)

    def forward(self, dem, rs, is_training):

        global fusion

        left = self.begin_dem(dem)
        right = self.begin_rs(rs)

        registration_rs = self.registration(left, right)

        fea = torch.cat((left, registration_rs), 1)
        for i in range(4):
            fusion = self.__getattr__('AB' + str(i + 1))(fea)
        buffer_fusion = self.conv_content(fusion)
        # for i in range(4):
        #     right_fusion = self.__getattr__('RCAB' + str(i + 1))(right)

        edge_lr = kornia.filters.sobel(left)
        edge_rs = kornia.filters.sobel(registration_rs)

        if is_training == 1:
            grad_fusion_64, buffer_rs = self.fusion_edge_1(edge_lr, edge_rs, is_training=is_training)
            grad_fusion = self.conv_edge(grad_fusion_64)
            grad_rs = self.conv_edge(buffer_rs)

            buffer = self.conv_fusion(torch.cat([buffer_fusion, grad_fusion], 1))
            res = buffer.clone()
            for i in range(10):
                res = self.__getattr__('fusion' + str(i + 1))(res)
            buffer = self.end(buffer + res)
            out = buffer+dem
            return out, grad_fusion, grad_rs

        if is_training == 0:
            grad_fusion = self.fusion_edge_1(edge_lr, edge_rs, is_training=is_training)
            grad_fusion = self.conv_edge(grad_fusion)

            buffer = self.conv_fusion(torch.cat([buffer_fusion, grad_fusion], 1))
            res = buffer.clone()
            for i in range(10):
                res = self.__getattr__('fusion' + str(i + 1))(res)
            buffer = self.end(buffer + res)
            out = buffer+dem
            return out
