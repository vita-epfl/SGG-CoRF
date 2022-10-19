import math

import torch
import torch.nn as nn
import torch.nn.functional as F

#from .transformer_utils import trunc_normal_

try:
    from timm.models.layers import trunc_normal_
except ImportError:
    pass


class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
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
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        # FIXME: Alternative to get centered coordinates per patch (for better positional information)
        # Assuming cumsum is always >= 1
        # eps = 0
        # y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * 2 - 1
        # x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * 2 - 1

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)  # B, dim, Hp, Wp
        return pos


class PositionalEncodingLearned1d(nn.Module):
    """
    Learned 1d positional encoding

    """
    def __init__(self, H=32, W=32, dim=768):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, dim, H, W))
        self.H, self.W, self.dim = H, W, dim
        trunc_normal_(self.pos_embed, std=.02)

    def interpolate_pos_encoding(self, B, Hp, Wp, mask=None):
        pos = self.pos_embed

        if self.H == Hp and self.W == Wp and mask is None:
            pos = pos.expand(B, -1, -1, -1)
            return pos

        if mask is None:
            pos = F.interpolate(
                pos,
                size=(Hp, Wp),
                mode='bilinear',
                align_corners=False,
            )
            pos = pos.expand(B, -1, -1, -1)  # B, dim, Hp, Wp
            return pos
        else:
            mask = mask.reshape(B, Hp, Wp)
            no_mask_Hp = (~mask).sum(dim=1)[:, 0]
            no_mask_Wp = (~mask).sum(dim=1)[:, 0]

            # TODO: is it possible to do it without a list?
            resized_positions = []
            # Need to resize each embedding individually
            for nm_Hp, nm_Wp in zip(no_mask_Hp.tolist(), no_mask_Wp.tolist()):
                resized_pos = F.interpolate(
                    pos,
                    size=(nm_Hp, nm_Wp),
                    mode='bilinear',
                    align_corners=False,
                )
                resized_pos = F.pad(
                    resized_pos, (0, Wp - nm_Wp, 0, Hp - nm_Hp), "constant", 0,
                )
                resized_positions.append(resized_pos)

            pos = torch.cat(resized_positions, dim=0)
            return pos

    def forward(self, B, Hp, Wp, mask=None, **kwargs):
        return self.interpolate_pos_encoding(B, Hp, Wp, mask=mask)


class PositionalEncodingLearned2d(nn.Module):
    """
    Learned 2d positional encoding
    """

    def __init__(self, H=32, W=32, hidden_dim=256, dim=768):
        super().__init__()
        self.pos_embed_y = nn.Parameter(torch.zeros(1, hidden_dim, H))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, hidden_dim, W))
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.H, self.W, self.dim = H, W, dim
        trunc_normal_(self.pos_embed_y, std=.02)
        trunc_normal_(self.pos_embed_x, std=.02)

    def interpolate_pos_encoding(self, B, Hp, Wp, mask=None):
        pos_y = self.pos_embed_y
        pos_x = self.pos_embed_x

        if self.H == Hp and self.W == Wp and mask is None:
            pos_y = pos_y.expand(B, -1, -1)  # B, dim, Hp, Wp
            pos_x = pos_x.expand(B, -1, -1)
            return pos_y, pos_x

        if mask is None:
            pos_y = F.interpolate(
                pos_y,
                size=Hp,
                mode='linear',
                align_corners=False,
            )
            pos_x = F.interpolate(
                pos_x,
                size=Wp,
                mode='linear',
                align_corners=False,
            )
            pos_y = pos_y.expand(B, -1, -1)  # B, dim, Hp, Wp
            pos_x = pos_x.expand(B, -1, -1)
            return pos_y, pos_x
        else:
            mask = mask.reshape(B, Hp, Wp)
            no_mask_Hp = (~mask).sum(dim=1)[:, 0]
            no_mask_Wp = (~mask).sum(dim=1)[:, 0]

            # TODO: is it possible to do it without a list?
            resized_positions_x = []
            resized_positions_y = []
            # Need to resize each embedding individually
            for nm_Hp, nm_Wp in zip(no_mask_Hp.tolist(), no_mask_Wp.tolist()):
                resized_pos_y = F.interpolate(
                    pos_y,
                    size=nm_Hp,
                    mode='linear',
                    align_corners=False,
                )
                resized_pos_x = F.interpolate(
                    pos_x,
                    size=nm_Wp,
                    mode='linear',
                    align_corners=False,
                )

                resized_pos_y = F.pad(resized_pos_y, (0, Hp - nm_Hp), "constant", 0)
                resized_pos_x = F.pad(resized_pos_x, (0, Wp - nm_Wp), "constant", 0)

                resized_positions_y.append(resized_pos_y)
                resized_positions_x.append(resized_pos_x)

            pos_y = torch.cat(resized_positions_y, dim=0)
            pos_x = torch.cat(resized_positions_x, dim=0)
            return pos_y, pos_x

    def forward(self, B, Hp, Wp, mask=None, **kwargs):
        pos_y, pos_x = self.interpolate_pos_encoding(B, Hp, Wp, mask=mask)

        # Combine embeddings
        pos = torch.cat([pos_y[:, :, :, None].expand(-1, -1, -1, Wp),
                         pos_x[:, :, None, :].expand(-1, -1, Hp, -1)],
                        dim=1)
        pos = self.token_projection(pos)
        return pos


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
