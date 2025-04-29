import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm3d
from torch.nn.modules.conv import Conv3d

class UNet3D_encoder(nn.Module):
    def __init__(self, in_channel, batchnorm_flag=True):
        self.in_channel = in_channel
        super(UNet3D_encoder, self).__init__()

        self.ec1 = self.encoder(self.in_channel, 32, batchnorm=batchnorm_flag)
        self.ec2 = self.encoder(64, 64, batchnorm=batchnorm_flag)
        self.ec3 = self.encoder(128, 128, batchnorm=batchnorm_flag)
        self.ec4 = self.encoder(256, 256, batchnorm=batchnorm_flag)


    def encoder(self, in_channels, out_channels, bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias),
                BatchNorm3d(out_channels, affine=False),
                ReLU(),
                Conv3d(out_channels, 2*out_channels, kernel_size=1, stride=1, bias=bias),
                BatchNorm3d(2*out_channels, affine=False),
                ReLU())
        else:
            layer = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=bias),
                ReLU(),
                Conv3d(out_channels, 2*out_channels, kernel_size=1, stride=1, bias=bias),
                ReLU())
        return layer

    def forward(self, x):

        down1 = self.ec1(x)
        down2 = self.ec2(down1)
        down3 = self.ec3(down2)

        u3 = self.ec4(down3)

        return u3, [down3, down2, down1]

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                