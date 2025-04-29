import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from typing import Optional
from einops import rearrange

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
        
        inputs = self.proj(inputs)
        _, _, z, x, y = inputs.shape
        # flatten: [B, C, Z, X, Y] -> [B, C, ZXY]
        # transpose: [B, C, ZXY] -> [B, ZXY, C]
        inputs = inputs.flatten(2).transpose(1, 2)
        inputs = self.norm_layer(inputs)

        return inputs

class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==4 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, Z*H*W, C
        """
        Z, H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == Z * H * W, "input feature has wrong size"

        x = x.view(B, Z, H, W, C)
        x = rearrange(x, 'b z h w (p1 p2 p3 c)-> b (z p1) (h p2) (w p3) c', p1=2, p2=2, p3=2, c=C//8)
        x = x.view(B,-1,C//8)
        x= self.norm(x)

        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim//4)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        Z, H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == Z * H * W, "input feature has wrong size"

        x = x.view(B, Z, H, W, C)
        x = rearrange(x, 'b z h w (p1 p2 p3 c)-> b (z p1) (h p2) (w p3) c', p1=self.dim_scale, p2=self.dim_scale, p3=self.dim_scale, c=C//(self.dim_scale**3))
        x = x.view(B,-1,self.output_dim//4)
        x= self.norm(x)

        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def window_partition(x, window_size: int):
    """
    x: (B,Z,X,Y,C)
    window_size (int):window_size(M)
    Returns:
        windows: (num_windows*B, window_size, window_size, windows_size, C)
    """
    B, Z, X, Y, C = x.shape
    x = x.view(B, Z // window_size, window_size, X // window_size, window_size, Y // window_size, window_size, C)
    # permute: [B, Z//M, M, X//M, M, Y//M, M, C] -> [B, Z//M, X//M, Y//M, M, M, M, C]
    # view: [B, Z//M, X//M, Y//M, M, M, M, C] -> [B*num_windows, M, M, M, C]
    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size, window_size, window_size, C)

    return windows

def window_reverse(windows, window_size: int, Z: int, X: int, Y: int):
    """
    windows: (num_windows*B, windows_size, windows_size, windows_size, C)
    window_size (int):window_size(M)
    Returns:
        x: (B,Z,X,Y,C)
    """
    B = int(windows.shape[0] / (Z * X * Y / window_size / window_size / window_size))
    #view: [B*num_windows, M, M, M, C] -> [B, Z//M, X//M, Y//M, M, M, M, C]
    x = windows.view(B, Z // window_size, X // window_size, Y // window_size, window_size, window_size, window_size, -1)
    # permute: [B, Z//M, X//M, Y//M, M, M, M, C] -> [B, Z//M, M, X//M, M, Y//M, M, C] 
    # view: [B, Z//M, M, X//M, M, Y//M, M, C] -> [B, Z, X, Y, C]
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(B, Z, X, Y, -1)

    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mz, Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1), num_heads))  # [2*Mz-1 * 2*Mx-1 * 2*My-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_z = torch.arange(self.window_size[0])
        coords_x = torch.arange(self.window_size[1])
        coords_y = torch.arange(self.window_size[2])
        coords = torch.stack(torch.meshgrid([coords_z, coords_x, coords_y]))  # [2, Mz, Mx, My]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mz*Mx*My]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mz*Mx*My, Mz*Mx*My]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mz*Mx*My, Mz*Mx*My , 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mz*Mx*My, Mz*Mx*My]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mz*Mx*My, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, num_tokens, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, num_tokens, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, num_tokens, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, num_tokens, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, num_tokens]
        # @: multiply -> [batch_size*num_windows, num_heads, num_tokens, num_tokens]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1] * self.window_size[2], self.window_size[0] * self.window_size[1] * self.window_size[2], -1)  # [Mz*Mx*My,Mz*Mx*My,nH]
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mz*Mx*My, Mz*Mx*My]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mz*Mx*My, Mz*Mx*My]
            nW = mask.shape[0]  # num_windows
            # [batch_size, num_windows, num_heads, num_tokens, num_tokens]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            Z, X, Y = self.input_resolution
            img_mask  = torch.zeros((1, Z, X, Y, 1))  # 1 Z X Y 1

            z_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))

            x_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            y_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for z in z_slices:
                for x in x_slices:
                    for y in y_slices:
                        img_mask[:, z, x, y, :] = cnt
                        cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # [nW, M, M, M, 1]
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size * self.window_size)# [nW, M*M*M] 
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)#[nW, 1, M*M*M] - [nW, M*M*M, 1]
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        Z, X, Y = self.input_resolution
        B, L, C = x.shape
        assert L == Z * X * Y, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, Z, X, Y, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size * self.window_size, C)  # nW*B, window_size*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size,self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, Z, X, Y)  # B Z' X' Y' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(1, 2, 3))
        else:
            x = shifted_x
        x = x.view(B, Z * X * Y, C)
 
        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class BasicLayer_up(nn.Module):

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # bulid merging layer
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer)
            for i in range(depth)])
        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, input):

        for blk in self.blocks:
            if not torch.jit.is_scripting() and self.use_checkpoint:
                input = checkpoint.checkpoint(blk, input)
            else:
                input = blk(input)
        if self.upsample is not None:
            input = self.upsample(input)

        return input 

class SwinTransformerDecoder(nn.Module):
    r""" 
    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, in_channels=1,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        
        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            image_size=img_size, patch_size=patch_size, in_channel=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        self.patch_grid = self.patch_embed.grid_size

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.absolute_pos_embed, std=.02)
        else:
            self.absolute_pos_embed = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        # concat_linear_num = [576,264,144,72]
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(int(1.5 * embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            # concat_linear = nn.Linear(concat_linear_num[i_layer-1], concat_linear_num[i_layer]) if i_layer > 0 else nn.Identity() 
            if i_layer ==0 :
                layer_up = PatchExpand(input_resolution=(self.patch_grid[0] // (2 ** (self.num_layers-1-i_layer)),
                self.patch_grid[1] // (2 ** (self.num_layers-1-i_layer)), self.patch_grid[2] // (2 ** (self.num_layers-1-i_layer))), 
                dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)), dim_scale=4, norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                input_resolution=(self.patch_grid[0] // (2 ** (self.num_layers-1-i_layer)),
                                                  self.patch_grid[1] // (2 ** (self.num_layers-1-i_layer)),
                                                  self.patch_grid[2] // (2 ** (self.num_layers-1-i_layer))),
                                depth=depths[(self.num_layers-1-i_layer)],
                                num_heads=num_heads[(self.num_layers-1-i_layer)],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                norm_layer=norm_layer,
                                upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm_up = norm_layer(self.embed_dim)

        self.up = FinalPatchExpand_X4(input_resolution=(img_size//(4*patch_size),img_size//patch_size, img_size//patch_size),dim_scale=4,dim=embed_dim)
        self.output = nn.Conv3d(in_channels=embed_dim//4,out_channels=self.in_channels,kernel_size=1,bias=False)

    def forward_up_features(self, x, x_downsample):
        
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x,x_downsample[3-inx]],-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C
  
        return x

    def up_x4(self, x):

        Z, H, W = self.patch_grid
        B, L, C = x.shape
        assert L == Z*H*W, "input features has wrong size"

        # if self.final_upsample=="expand_first":
        x = self.up(x)
        x = x.view(B,4*Z,4*H,4*W,-1)
        x = x.permute(0,4,1,2,3) #B,C,H,W
            
        return x
    def forward(self, x, x_downsample):

        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)
        x = self.output(x)

        return x