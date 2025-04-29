import torch.nn as nn
from model.swin_encoder import SwinTransformerEncoder
from torchinfo import summary

class SwinUNETR(nn.Module):
    def __init__(self, upsample="vae", dim=1024):
        super(SwinUNETR, self).__init__()
        self.dim = dim
        self.ThreeDimHSFomerEncoder = SwinTransformerEncoder(
            img_size=256,
            in_chans=16,
            patch_size=4,
            window_size=8,
            embed_dim=128,#(4*4*4*1) patch_size*patch_size*patch_size*in_channel
            depths=(2, 2, 6, 2),
            num_heads=(4, 8, 16, 32),
        )
        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(dim, 4)
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(dim, 512)
 
        self.conv = nn.Sequential(
            nn.Conv3d(dim, dim // 2, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 2),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 2, dim // 4, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 4),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 4, dim // 8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 8),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 8, dim // 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
            nn.Conv3d(dim // 16, dim // 16, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(dim // 16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False),
        )
        self.output = nn.Conv3d(dim // 16, 2, kernel_size=1, stride=1)

    def forward(self, x):
        x_out, _ = self.ThreeDimHSFomerEncoder(x.contiguous())
        b, _, _ = x_out.shape
        x_out = x_out.view(b, self.dim, 8, 8, 2)
        _, c, h, w, d = x_out.shape
        x4_reshape = x_out.flatten(start_dim=2, end_dim=4)
        x4_reshape = x4_reshape.transpose(1, 2)
        x_rot = self.rotation_pre(x4_reshape[:, 0])
        x_rot = self.rotation_head(x_rot)
        x_contrastive = self.contrastive_pre(x4_reshape[:, 1])
        x_contrastive = self.contrastive_head(x_contrastive)
        x_rec = x_out.flatten(start_dim=2, end_dim=4)
        x_rec = x_rec.view(-1, c, h, w, d)
        x_rec = self.conv(x_rec)
        x_rec = x_rec.permute((0, 1, 4, 2, 3))
        x_rec = self.output(x_rec)
        return x_rec

if __name__ == "__main__":

    model = SwinUNETR()
    summary(model, input_size=(2, 1, 64, 256, 256))
 