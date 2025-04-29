import torch
import torch.nn as nn

# from monai.networks.nets.swin_unetr import SwinTransformer as SwinViT
from model.swin_encoder import SwinTransformerEncoder
from model.biVoxFormer_encoder import BiVoxFormerEecoder
# from monai.utils import ensure_tuple_rep

class SSLHead(nn.Module):
    def __init__(self, args, upsample="vae", dim=1024):
        super(SSLHead, self).__init__()
        self.dim = dim
        # patch_size = ensure_tuple_rep(2, args.spatial_dims)
        # window_size = ensure_tuple_rep(7, args.spatial_dims)
        self.ThreeDimHSFomerEncoder = SwinTransformerEncoder(
            img_size=args.img_size,
            in_chans=args.model_in_chans,
            patch_size=args.patch_size,
            window_size=args.window_size,
            embed_dim=args.embed_dim,#(4*4*4*1) patch_size*patch_size*patch_size*in_channel
            depths=(2, 2, 6, 2),
            num_heads=(4, 8, 16, 32),
        )
        self.BiVoxFormerEncoder = BiVoxFormerEecoder(
            depth=[2, 2, 8, 2],
            num_classes=2, 
            embed_dim=[48, 96, 192, 384], mlp_ratios=[3, 3, 3, 3],
            #------------------------------
            n_win=4,
            kv_downsample_mode='identity',
            kv_per_wins=[-1, -1, -1, -1],
            topks=[1, -1, -1, -2],
            side_dwconv=5,
            before_attn_dwconv=3,
            layer_scale_init_value=-1,
            qk_dims=[32, 64, 128, 256],
            head_dim=32,
            param_routing=False, diff_routing=False, soft_routing=False,
            pre_norm=True,
            pe=None)
        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(dim, 4)
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(dim, 512)
        if upsample == "large_kernel_deconv":
            self.conv = nn.ConvTranspose3d(dim, args.in_channels, kernel_size=(32, 32, 32), stride=(32, 32, 32))
        elif upsample == "deconv":
            self.conv = nn.Sequential(
                nn.ConvTranspose3d(dim, dim // 2, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 2, dim // 4, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 4, dim // 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 8, dim // 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
                nn.ConvTranspose3d(dim // 16, args.in_channels, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            )
        elif upsample == "vae":
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
                nn.Conv3d(dim // 16, args.in_channels, kernel_size=1, stride=1),
            )

    def forward(self, x):
        x_out, _ = self.ThreeDimHSFomerEncoder(x.contiguous())
        x_out = x_out.view(1, self.dim, 8, 8, 2)
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
        return x_rot, x_contrastive, x_rec