from functools import partial
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from einops import rearrange, repeat
from mamba_ssm import selective_scan_fn
from torchinfo import summary
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==4 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, Z*H*W, C
        """
        x = x.permute(0, 2, 3, 4, 1) # B, C, D, H, W ==> B, D, H, W, C
        x = self.expand(x)
        B, D, H, W, C = x.shape

        x = x.view(B, D, H, W, C)
        x = rearrange(x, 'b z h w (p1 p2 p3 c)-> b (z p1) (h p2) (w p3) c', p1=2, p2=2, p3=2, c=C//8)
        x = x.view(B,-1,C//8)
        x= self.norm(x)
        x = x.reshape(B, D*2, H*2, W*2, C//8)

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim//4)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        x = x.permute(0, 2, 3, 4, 1)
        x = self.expand(x)
        B, D, H, W, C = x.shape

        # x = x.view(B, Z, H, W, C)
        x = rearrange(x, 'b z h w (p1 p2 p3 c)-> b (z p1) (h p2) (w p3) c', p1=self.dim_scale, p2=self.dim_scale, p3=self.dim_scale, c=C//(self.dim_scale**3))
        x = x.view(B,-1,self.output_dim//4)
        x= self.norm(x)
        x = x.reshape(B, D*self.dim_scale, H*self.dim_scale, W*self.dim_scale, self.output_dim//4)

        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed(nn.Module):
    """
    3D Image to Patch Embedding
    """
    def __init__(self, image_size=256, patch_size=4, in_channel=1, embed_dim=96, norm_layer=None):
        super().__init__()
        image_size = (image_size // 4, image_size, image_size) #(64,256,256)
        patch_size = (patch_size, patch_size, patch_size)
        self.patch_size = patch_size
        self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1], image_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        self.in_channel = in_channel
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(in_channel, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm_layer = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, inputs):
        _, _, z, x, y = inputs.shape

        # padding
        pad_input = (z % self.patch_size[0] != 0) or (x % self.patch_size[1] != 0) or (y % self.patch_size[2] != 0)
        if pad_input:
            inputs = F.pad(inputs, (0, self.patch_size[2] - y % self.patch_size[2],
                                    0, self.patch_size[1] - x % self.patch_size[1],
                                    0, self.patch_size[0] - z % self.patch_size[0]))
        
        inputs = self.proj(inputs).permute(0, 2, 3, 4, 1)
        # _, _, z, x, y = inputs.shape
        # flatten: [B, C, Z, X, Y] -> [B, C, ZXY]
        # transpose: [B, C, ZXY] -> [B, ZXY, C]
        # inputs = inputs.flatten(2).transpose(1, 2)
        inputs = self.norm_layer(inputs)

        return inputs

class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        Z, X, Y = self.input_resolution
        B, Z, X, Y, C = x.shape
        x = x.view(B, Z, X, Y, C)
 
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 1::2, 0::2, :]
        x2 = x[:, 0::2, 0::2, 1::2, :] 
        x3 = x[:, 1::2, 1::2, 1::2, :]
        x_concat = torch.cat([x0, x1, x2, x3], -1)
        x_concat = x_concat.view(B, Z//2, X//2, Y//2, 4 * C)  

        x_norm = self.norm(x_concat)
        x_reduction = self.reduction(x_norm)

        return x_reduction

class SS3D(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2, dt_rank="auto", dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, 
                dt_init_floor=1e-4, dropout=0., conv_bias=True, bias=False, device=None, dtype=None, **kwargs):
        factory_kwargs = {"device": device, "dtype":dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv3d = nn.Conv3d(in_channels=self.d_inner, out_channels=self.d_inner, groups=self.d_inner, bias=conv_bias, kernel_size=d_conv, 
                                padding=(d_conv - 1) // 2, **factory_kwargs)
        self.act = nn.SiLU()
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        
    @staticmethod    
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
         # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A) # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, D, H, W = x.shape
        L = D * H * W
        K = 4

        x_dhwwhd = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=4).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_dhwwhd, torch.flip(x_dhwwhd, dims=[-1])], dim=1) # [B, K, D, L]

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L) # [B, K*D, L]
        dts = dts.contiguous().float().view(B, -1, L) # [B, K*D, L]
        Bs = Bs.float().view(B, K, -1, L) # [B, K, d_state, L]
        Cs = Cs.float().view(B, K, -1, L) # [B, K, d_state, L]
        Ds = self.Ds.float().view(-1) # (K * D)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state) # (K * D, d_state)
        dt_proj_bias = self.dt_projs_bias.float().view(-1) # (K * D)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_proj_bias,
            delta_softplus=True,
            return_last_state=False
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        dwh_y = torch.transpose(out_y[:, 1].view(B, -1, D, W, H), dim0=2, dim1=4).contiguous().view(B, -1, L)
        invdwh_y = torch.transpose(inv_y[:, 1].view(B, -1, D, W, H), dim0=2, dim1=4).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], dwh_y, invdwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, D, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z= xz.chunk(2, dim=-1) # [B, D, H, W, C]

        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self.act(self.conv3d(x)) # [B, C, D, H, W]
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, D, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out
        
class VSSBlock(nn.Module):
    def __init__(self, hidden_dim: int = 0, drop_path: float = 0, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                    attn_drop_rate: float = 0, d_state: int = 16, **kwargs) -> None:
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS3D(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.self_attention(self.ln_1(input)))
        return x 

class VSSLayer(nn.Module):
    def __init__(self, dim, depth, attn_drop=0., drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, d_state=16, **kwargs):
        super().__init__()
        self.dim = dim 
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer = norm_layer,
                attn_drop_rate=attn_drop,
                d_state=d_state,
            )
            for i in range(depth)])

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class VSSEncoder(nn.Module):
    def __init__(self, patch_size=4, in_chans=1, depths=[2, 2, 9, 2], dims=[96, 192, 384, 768], d_state=16,  drop_rate=0., attn_drop_rate=0., 
                drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True, use_checkpoint=False, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channel=in_chans, embed_dim=self.embed_dim, 
                                        norm_layer=norm_layer if patch_norm else None)
        self.patch_grid = self.patch_embed.grid_size
        self.ape = False
        drop_rate = 0.0
        if self.ape:
            self.patches_resolution = self.patch_embed.patches_resolution
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, *self.patches_resolution, self.embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VSSLayer(
                dim=dims[i_layer],
                depth=depths[i_layer],
                d_state=math.ceil(dims[0] / 6) if d_state is None else d_state,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsamples.append(PatchMerging((self.patch_grid[0] // (2 ** i_layer),
                                                  self.patch_grid[1] // (2 ** i_layer),
                                                  self.patch_grid[2] // (2 ** i_layer)), dim=dims[i_layer], norm_layer=norm_layer))
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)  

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x_ret = []
        x_ret.append(x)

        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for s, layer in enumerate(self.layers):
            x = layer(x)
            x_ret.append(x.permute(0, 4, 1, 2, 3))
            if s < len(self.downsamples):
                x = self.downsamples[s](x)
        return x_ret

class UNetResDecoder(nn.Module):
    def __init__(self, num_classes, deep_supervision, features_per_stage, drop_path_rate=0.2, d_state=16) -> None:
        super().__init__()
        enconder_output_channels = features_per_stage
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        n_stages_encoder = len(enconder_output_channels)
        dpr = [x.item() for x in torch.linspace(drop_path_rate, 0, (n_stages_encoder-1)*2)]
        depths = [2, 2, 2, 2]

        stages = []
        expand_layers = []
        seg_layers = []
        concat_back_dim = []
        for s in range(1, n_stages_encoder):
            
            input_features_below = enconder_output_channels[-s]
            input_features_skip = enconder_output_channels[-(s + 1)]
            expand_layers.append(PatchExpand(
                input_resolution=None,
                dim=input_features_below,
                dim_scale=4,
                norm_layer=nn.LayerNorm,
            ))
            stages.append(VSSLayer(
                dim=input_features_skip,
                depth=2,
                attn_drop=0.,
                drop_path=dpr[sum(depths[:s-1]):sum(depths[:s])],
                d_state=math.ceil(2*input_features_skip / 6) if d_state is None else d_state,
                norm_layer=nn.LayerNorm,
                downsample=None,
                use_checkpoint=False,
            ))               
            seg_layers.append(nn.Conv3d(input_features_skip, num_classes, 1, 1, 0, bias=True))
            concat_back_dim.append(nn.Linear(int(1.5*input_features_skip), input_features_skip))

        expand_layers.append(FinalPatchExpand_X4(
            input_resolution=None,
            dim=enconder_output_channels[0],
            dim_scale=4,
            norm_layer=nn.LayerNorm,
        ))
        stages.append(nn.Identity())
        seg_layers.append(nn.Conv3d(input_features_skip//4, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.expand_layers = nn.ModuleList(expand_layers)
        self.seg_layers = nn.ModuleList(seg_layers)
        self.concat_back_dim = nn.ModuleList(concat_back_dim)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.expand_layers[s](lres_input)
            if s < (len(self.stages) - 1):
                x = torch.cat((x, skips[-(s+2)].permute(0, 2, 3, 4, 1)), -1)
                x = self.concat_back_dim[s](x)
            x = self.stages[s](x).permute(0, 4, 1, 2, 3)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]
        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

class Dliatedblock(nn.Module):
    def __init__(self, in_channel, bias=False):
        super(Dliatedblock, self).__init__()
        filters = [4, 8, 16]
        self.conv1 = nn.Conv3d(in_channel, out_channels=filters[0], kernel_size=3, stride=1,
                                padding=1, dilation=1, bias=bias)
        self.bn1 = nn.BatchNorm3d(filters[0])
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv3d(filters[0], filters[1], kernel_size=3, stride=1, padding=2, dilation=2, bias=bias)
        self.bn2 = nn.BatchNorm3d(filters[1])

        self.conv3 = nn.Conv3d(filters[1], filters[2], kernel_size=3, stride=1, padding=5, dilation=5, bias=bias)
        self.bn3 = nn.BatchNorm3d(filters[2])

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        return out

class SwinUMamba(nn.Module):
    def __init__(self, vss_args, decoder_args) -> None:
        super().__init__()
        # self.dliatedblock = Dliatedblock(in_channel=1, bias=False)
        self.vssm_encoder = VSSEncoder(**vss_args)
        self.decoder = UNetResDecoder(**decoder_args)

    def forward(self, x):
        # x = self.dliatedblock(x)
        skips = self.vssm_encoder(x)
        out = self.decoder(skips)
        return out

    @torch.no_grad()
    def freeze_encoder(self):
        for name, param in self.vssm_encoder.named_parameters():
            if "patch_embed" not in name:
                param.requires_grad = False

    @torch.no_grad()
    def unfreeze_encoder(self):
        for param in self.vssm_encoder.parameters():
            param.requires_grad = True

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

def get_swin_umamba(num_classes):
    vss_args = dict(
        in_chans=1,
        patch_size=4,
        depths=[2,2,6,2],
        dims=128,
        drop_path_rate=0.2
    )
    decoder_args = dict(
        num_classes=num_classes,
        deep_supervision=False, 
        features_per_stage=[128, 256, 512, 1024],      
        drop_path_rate=0.2,
        d_state=16,
    )
    model = SwinUMamba(vss_args, decoder_args)
    # summary(model, input_size=(2, 1, 64, 256, 256))
    model.apply(InitWeights_He(1e-2))

    return model

if __name__ == "__main__":
    device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
    model = get_swin_umamba(num_classes=2).to(device)

    import numpy as np

    data = np.random.randint(0,256, (1, 1, 64, 256, 256)).astype(np.float32)
    data = torch.from_numpy(data).to(device)

    y = model(data)
    print(y.shape)