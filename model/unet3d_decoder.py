import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm3d
from torch.nn.modules.conv import Conv3d

class UNet3D_decoder(nn.Module):
    def __init__(self, in_channel, batchnorm_flag=True):
        self.in_channel = in_channel
        super(UNet3D_decoder, self).__init__()

        self.up3 = nn.Upsample(scale_factor=2., mode='trilinear', align_corners=True)
        self.dc3 = self.decoder(256 + 512, 256, batchnorm=batchnorm_flag)
        self.up2 = nn.Upsample(scale_factor=2., mode='trilinear', align_corners=True)
        self.dc2 = self.decoder(128 + 256, 128, batchnorm=batchnorm_flag)
        self.up1 = nn.Upsample(scale_factor=2., mode='trilinear', align_corners=True)
        self.dc1 = self.last_decoder(64 + 128, 64, self.in_channel, batchnorm=batchnorm_flag)

        # self.dc0 = nn.Conv3d(64, n_classes, 1)
        self.softmax = nn.Softmax(dim=1)

    def decoder(self, in_channels, out_channels, bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
                BatchNorm3d(out_channels, affine=False),
                ReLU(),
                Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=bias),
                BatchNorm3d(out_channels, affine=False),
                ReLU())
        else:
            layer = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias),
                ReLU(),
                Conv3d(out_channels, out_channels, kernel_size=1, stride=1, bias=bias),
                ReLU())
        return layer

    def last_decoder(self, in_channels, out_channels, num_classes, batchnorm):
        if batchnorm:
            layer = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                BatchNorm3d(out_channels, affine=False),
                ReLU(),
                Conv3d(out_channels, self.in_channel , kernel_size=1, stride=1, padding=0)
            )
        else:
            layer = nn.Sequential(
                Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                ReLU(),
                Conv3d(out_channels, self.in_channel, kernel_size=1, stride=1, padding=0)
            )
        return layer

    def forward(self, u3, down_list):

        d3 = torch.cat((self.up3(u3), down_list[0]), 1)
        u2 = self.dc3(d3)
        d2 = torch.cat((self.up2(u2), down_list[1]), 1)
        u1 = self.dc2(d2)
        d1 = torch.cat((self.up1(u1), down_list[2]), 1)
        out = self.dc1(d1)

        out_map = F.interpolate(out, size=(64, 256, 256), mode='trilinear', align_corners=True)
        # out = out_map.view(out_map.numel() // self.numClass, self.numClass)
        # out = self.softmax(out_map)
        # out = self.softmax(out)

        return out_map

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, std=0.001)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                