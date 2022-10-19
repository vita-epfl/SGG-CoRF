import torch
import torch.nn as nn

from .positional_encoding import (PositionalEncodingFourier, PositionalEncodingLearned1d,
                                  PositionalEncodingLearned2d, PositionalEncodingFourierv2)

try:
    from timm.models.layers import DropPath, trunc_normal_
except ImportError:
    pass


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_cls=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.with_cls = with_cls

    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, nh, N, C // nh

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            # Pad with input token
            if self.with_cls:
                mask = nn.functional.pad(mask, (1, 0), "constant", 0)
            # Set masked scores to -inf, softmax then sets them to 0
            attn = attn.masked_fill(mask.reshape(B, 1, 1, N) == 1, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # B, nh, N, N

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, with_cls=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            with_cls=with_cls)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, mask=None, return_attention=False):
        y, attn = self.attn(self.norm1(x), mask=mask)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerRefine(nn.Module):

    def __init__(self, in_channels, embed_dim=512, out_channels=0, depth=6, num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, pos_encoding_type="fourierv2", **kwargs):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim

        self.input_proj = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        if pos_encoding_type == "fourier":
            self.pos_embedder = PositionalEncodingFourier(dim=embed_dim)
        elif pos_encoding_type == "fourierv2":
            self.pos_embedder = PositionalEncodingFourierv2(dim=embed_dim)
        elif pos_encoding_type == "learned1d":
            # TODO: give option to choose H, W of encoding
            # Default H, W: 32, 32
            self.pos_embedder = PositionalEncodingLearned1d(dim=embed_dim)
        elif pos_encoding_type == "learned2d":
            # TODO: give option to choose H, W of encoding
            # Default H, W: 32, 32
            # TODO: give option to choose  hidden_dim of encoding
            # Default hidden_dim: 256
            self.pos_embedder = PositionalEncodingLearned2d(dim=embed_dim)
        else:
            raise ValueError(f"Invalid type of positional encoding: {pos_encoding_type}")

        # Output projection output_proj
        self.output_proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1) if out_channels > 0 else nn.Identity()

        # Weight init
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.input_proj(x)
        B, C, Hp, Wp = x.shape
        x = x.flatten(2).permute(0, 2, 1)

        # Adding positional encoding
        pos_encoding = self.pos_embedder(B, Hp, Wp).reshape(B, -1, (Hp * Wp)).permute(0, 2, 1)
        x = x + pos_encoding

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, C, Hp, Wp)
        x = self.output_proj(x)
        return x

    def get_last_selfattention(self, x):
        x = self.input_proj(x)
        B, C, Hp, Wp = x.shape
        x = x.flatten(2).permute(0, 2, 1)

        # Adding positional encoding
        pos_encoding = self.pos_embedder(B, Hp, Wp, device=x.device).reshape(B, -1, (Hp * Wp)).permute(0, 2, 1)
        x = x + pos_encoding
        attns = []

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                attns.append(blk(x, return_attention=True))
                x = blk(x)
            else:
                # return attention of the last block
                attns.append(blk(x, return_attention=True))
                return attns
