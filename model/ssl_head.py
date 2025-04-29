import argparse
import torch
import torch.nn as nn

from model.ThreeD_hsformer_encoder import SwinTransformerEncoder
from model.ThreeD_hsformer_decoder import SwinTransformerDecoder

from model.unet3d_encoder import UNet3D_encoder
from model.unet3d_decoder import UNet3D_decoder

from model.cpnet_encoder import CPnet3DEncoder
from model.cpnet_decoder import CPnet3DDecoder

from model.vusmamba import VSSEncoder
from model.vusmamba import UNetResDecoder

class SSLHead(nn.Module):
    def __init__(self, args, dim=1024): #CPNet or unet dim:1024 3D-HSFormer dim: 768 3D-SwinUMamba dim: 768
        super(SSLHead, self).__init__()
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
            embed_dim=[32, 64, 128, 256], mlp_ratios=[3, 3, 3, 3],
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

        self.unet_encoder = UNet3D_encoder(in_channel=1)
        self.cpnet_encoder = CPnet3DEncoder(nbase=[1, 32, 64, 128, 256], out_classes=1, sz=3, residual_on=False, style_on=False, concatenation=True)

        self.swinumamba_encoder = VSSEncoder(in_chans=1, patch_size=4, depths=[2,2,6,2], dims=96, drop_path_rate=0.2)

        self.rotation_pre = nn.Identity()
        self.rotation_head = nn.Linear(dim, 4)
        self.contrastive_pre = nn.Identity()
        self.contrastive_head = nn.Linear(dim, 512)

        self.ThreeDimHSFomerDecoder = SwinTransformerDecoder(
            img_size=args.img_size,
            in_chans=args.model_in_chans,
            in_channels=args.in_channels,
            patch_size=args.patch_size,
            window_size=args.window_size,
            embed_dim=args.embed_dim,#(4*4*4*1) patch_size*patch_size*patch_size*in_channel
            depths=(2, 2, 6, 2),
            num_heads=(4, 8, 16, 32),
        )

        self.unet_decoder = UNet3D_decoder(in_channel=1)
        self.cpnet_decoder = CPnet3DDecoder(nbase=[1, 32, 64, 128, 256], out_classes=1, sz=3, residual_on=False, style_on=False, concatenation=True)
        self.swinumamba_decoder = UNetResDecoder(num_classes=1, deep_supervision=False, features_per_stage=[96, 192, 384, 768], drop_path_rate=0.2, d_state=16,)

    def forward(self, x):
        # x_out, x_outs = self.ThreeDimHSFomerEncoder(x.contiguous())
        # x_out, x_outs = self.unet_encoder(x.contiguous())
        x_out = self.cpnet_encoder(x.contiguous())
        # x_out = self.swinumamba_encoder(x.contiguous())
        # x_temp = x_out[4]
        # B, C, D, H, W = x_temp.size()
        # x_temp = x_temp.reshape(B, D*H*W, C)
        # x_rot = self.rotation_pre(x_temp[:, 0])
        x_rot = self.rotation_pre(x_out[3][:, 0])
        ## UNet or CPNet
        B, Z, H, W = x_rot.size()
        x_rot = x_rot.view(B, Z*H*W)
        ##
        x_rot = self.rotation_head(x_rot)
        x_contrastive = self.contrastive_pre(x_out[3][:, 1])
        ## UNet or CPNet
        B, Z, H, W = x_contrastive.size()
        x_contrastive = x_contrastive.view(B, Z*H*W)
        ##
        x_contrastive = self.contrastive_head(x_contrastive)
        # x_rec = self.ThreeDimHSFomerDecoder(x_out, x_outs)
        # x_rec = self.swinumamba_decoder(x_out)
        # x_rec = self.unet_decoder(x_out, x_outs)
        x_rec = self.cpnet_decoder(x_out)

        return x_rot, x_contrastive, x_rec

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PyTorch Training")
    parser.add_argument("--img_size", default=256, type=int, help="size of input image")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--model_in_chans", default=16, type=int, help="Input channel to the model")
    parser.add_argument("--patch_size", default=4, type=int, help="the patch size in 3D-HSFormer")
    parser.add_argument("--window_size", default=8, type=int, help="the window size in 3D-HSFormer")
    parser.add_argument("--embed_dim", default=128, type=int, help="the embed dim in 3D-HSFormer")

    args = parser.parse_args()
    model = SSLHead(args=args)
    # model = swin_base_patch4_window8_256(num_classes=2)

    import numpy as np

    data = np.random.randint(0,256, (1, 1, 64, 256, 256)).astype(np.float32)
    data = torch.from_numpy(data)

    y_rot, y_contrastive, y_rec = model(data)
    print(y_rot.shape)
    print(y_contrastive.shape)
    print(y_rec.shape)    