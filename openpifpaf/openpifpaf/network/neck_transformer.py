import math

import torch
import torch.nn as nn

try:
    from timm.models.layers import DropPath, trunc_normal_
except ImportError:
    pass

class PositionalEncodingFourierv2(nn.Module):

    def __init__(self, hidden_dim=64, dim=768, temperature=10000, scale_factor=32, projection=True):
        super().__init__()
        enc_dim = (hidden_dim + 1) * 2
        self.token_projection = nn.Conv2d(enc_dim, dim, kernel_size=1) if projection else nn.Identity()
        self.scale = math.pi * scale_factor
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, Hp, Wp, mask=None, device=None):
        if mask is None:
            if device is None:
                device = self.token_projection.weight.device
            mask = torch.zeros(B, Hp, Wp).bool().to(device)
        else:
            mask = mask.reshape(B, Hp, Wp)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        # Alternative to get centered coordinates per patch (for better positional information)
        y_embed = ((y_embed - 0.5) / (y_embed[:, -1:, :]) - 0.5) * 2 * self.scale
        x_embed = ((x_embed - 0.5) / (x_embed[:, :, -1:]) - 0.5) * 2 * self.scale
        yx_embed_unscaled = torch.stack([y_embed, x_embed], dim=-1) / self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # FIXME: Check if stacking them (instead of interleaving them) makes any difference
        # I don't think so given that they go through a projection afterwards either way
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x, yx_embed_unscaled), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)  # B, dim, Hp, Wp
        return pos


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


class JointTransformer(nn.Module):

    def __init__(self, in_channels, embed_dim=256, out_channels=0, depth=6, num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, pos_encoding_type="fourierv2", **kwargs):
        super().__init__()

        self.num_features = self.embed_dim = embed_dim

        self.input_projs = nn.ModuleList(
            [nn.Conv2d(in_channels, embed_dim, kernel_size=1),
             nn.Conv2d(in_channels, embed_dim, kernel_size=1)])


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        if pos_encoding_type == "fourierv2":
            self.pos_embedder = PositionalEncodingFourierv2(dim=embed_dim)
        else:
            raise ValueError(f"Invalid type of positional encoding: {pos_encoding_type}")

        self.task_embedding = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, embed_dim)), nn.Parameter(torch.zeros(1, 1, embed_dim))]
        )

        # Output projection output_proj
        self.output_projs = nn.ModuleList(
            [nn.Conv2d(embed_dim, out_channels, kernel_size=1) if out_channels > 0 else nn.Identity(),
             nn.Conv2d(embed_dim, out_channels, kernel_size=1) if out_channels > 0 else nn.Identity()])


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
        x0 = self.input_projs[0](x)
        x1 = self.input_projs[1](x)  # B, C, Hp, Wp
        B, C, Hp, Wp = x0.shape
        x0 = x0.flatten(2).permute(0, 2, 1)  # B, Hp * Wp, C
        x1 = x1.flatten(2).permute(0, 2, 1)

        # Adding positional encoding
        pos_encoding = self.pos_embedder(B, Hp, Wp).reshape(B, -1, (Hp * Wp)).permute(0, 2, 1)  # B, Hp * Wp, C
        x0 = x0 + pos_encoding
        x1 = x1 + pos_encoding

        # Add task embedding
        x0 = x0 + self.task_embedding[0]
        x1 = x1 + self.task_embedding[1]
        x = torch.cat((x0, x1), dim=1)  # B, Hp * Wp * 2, C

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        x0, x1 = torch.chunk(x, 2, dim=1)
        x0 = x0.permute(0, 2, 1).reshape(B, C, Hp, Wp)
        x1 = x1.permute(0, 2, 1).reshape(B, C, Hp, Wp)

        x0 = self.output_projs[0](x0)
        x1 = self.output_projs[1](x1)
        return x0, x1


class SoloTransformer(nn.Module):

    def __init__(self, in_channels, embed_dim=256, out_channels=0, depth=6, num_heads=8, mlp_ratio=4.,
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

        if pos_encoding_type == "fourierv2":
            self.pos_embedder = PositionalEncodingFourierv2(dim=embed_dim)
        else:
            raise ValueError(f"Invalid type of positional encoding: {pos_encoding_type}")

        # Output projection output_proj
        self.output_projs = nn.ModuleList(
            [nn.Conv2d(embed_dim, out_channels, kernel_size=1) if out_channels > 0 else nn.Identity(),
             nn.Conv2d(embed_dim, out_channels, kernel_size=1) if out_channels > 0 else nn.Identity()])

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

        x0 = self.output_projs[0](x)
        x1 = self.output_projs[1](x)
        return x0, x1
