import argparse
import torch
import torchvision.models
from standalone_nets import swin_transformer

BASE_FACTORIES = {
    'resnet18': lambda **kwargs: Resnet('resnet18', torchvision.models.resnet18, 512, **kwargs),
    'resnet50': lambda **kwargs: Resnet('resnet50', torchvision.models.resnet50, **kwargs),
    'resnet101': lambda **kwargs: Resnet('resnet101', torchvision.models.resnet101, **kwargs),
    'resnet152': lambda **kwargs: Resnet('resnet152', torchvision.models.resnet152, **kwargs),
    # Swin architectures: swin_t is roughly equivalent to unmodified resnet50
    'swin_t': lambda **kwargs: SwinTransformer(
        'swin_t', swin_transformer.swin_tiny_patch4_window7, **kwargs),
    'swin_s': lambda **kwargs: SwinTransformer(
        'swin_s', swin_transformer.swin_small_patch4_window7, **kwargs),
    'swin_b': lambda **kwargs: SwinTransformer(
        'swin_b', swin_transformer.swin_base_patch4_window7, **kwargs),
}

class BaseNetwork(torch.nn.Module):
    """Common base network.

    :param name: a short name for the base network, e.g. resnet50
    :param stride: total stride from input to output
    :param out_features: number of output features
    """

    def __init__(self, name: str, *, stride: int, out_features: int):
        super().__init__()
        self.name = name
        self.stride = stride
        self.out_features = out_features

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""

    def forward(self, *args):
        raise NotImplementedError




class Resnet(BaseNetwork):
    def __init__(self, name, torchvision_resnet, out_features=2048, pretrained=True,
                 pool0_stride=2, input_conv_stride=2, input_conv2_stride=0,
                 remove_last_block=False, block5_dilation=1):

        self.pretrained = pretrained
        self.pool0_stride = pool0_stride
        self.input_conv_stride = input_conv_stride
        self.input_conv2_stride = input_conv2_stride
        self.remove_last_block = remove_last_block
        self.block5_dilation = block5_dilation

        modules = list(torchvision_resnet(self.pretrained).children())
        stride = 32

        input_modules = modules[:4]

        # input pool
        if self.pool0_stride:
            if self.pool0_stride != 2:
                # pylint: disable=protected-access
                input_modules[3].stride = torch.nn.modules.utils._pair(self.pool0_stride)
                stride = int(stride * 2 / self.pool0_stride)
        else:
            input_modules.pop(3)
            stride //= 2

        # input conv
        if self.input_conv_stride != 2:
            # pylint: disable=protected-access
            input_modules[0].stride = torch.nn.modules.utils._pair(self.input_conv_stride)
            stride = int(stride * 2 / self.input_conv_stride)

        # optional use a conv in place of the max pool
        if self.input_conv2_stride:
            assert not self.pool0_stride  # this is only intended as a replacement for maxpool
            channels = input_modules[0].out_channels
            conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(channels, channels, 3, 2, 1, bias=False),
                torch.nn.BatchNorm2d(channels),
                torch.nn.ReLU(inplace=True),
            )
            input_modules.append(conv2)
            stride *= 2

        # block 5
        block5 = modules[7]
        if self.remove_last_block:
            block5 = None
            stride //= 2
            out_features //= 2

        if self.block5_dilation != 1:
            stride //= 2
            for m in block5.modules():
                if not isinstance(m, torch.nn.Conv2d):
                    continue

                # also must be changed for the skip-conv that has kernel=1
                m.stride = torch.nn.modules.utils._pair(1)

                if m.kernel_size[0] == 1:
                    continue

                m.dilation = torch.nn.modules.utils._pair(self.block5_dilation)
                padding = (m.kernel_size[0] - 1) // 2 * self.block5_dilation
                m.padding = torch.nn.modules.utils._pair(padding)

        super().__init__(name, stride=stride, out_features=out_features)
        self.input_block = torch.nn.Sequential(*input_modules)
        self.block2 = modules[4]
        self.block3 = modules[5]
        self.block4 = modules[6]
        self.block5 = block5

    def forward(self, *args):
        x = args[0]
        x = self.input_block(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


class FPN(torch.nn.Module):
    """ Feature Pyramid Network (https://arxiv.org/abs/1612.03144), modified to only
    refine and return the feature map of a single pyramid level.
    This implementation is more computationally efficient than torchvision's
    FeaturePyramidNetwork when only a single feature map is needed, as it avoids refining
    (i.e. applying a 3x3 conv on) feature maps that aren't used later on.
    For example, for Swin, if only the feature map of stride 8 (fpn_level=2) is needed,
    the feature maps of stride 4, 16 and 32 won't get refined with this implementation.
    """

    def __init__(self, in_channels, out_channels, fpn_level=3):

        super().__init__()

        self.num_upsample_ops = len(in_channels) - fpn_level

        self.lateral_convs = torch.nn.ModuleList()

        # Start from the higher-level features (start from the smaller feature maps)
        for i in range(1, 2 + self.num_upsample_ops):
            lateral_conv = torch.nn.Conv2d(in_channels[-i], out_channels, 1)
            self.lateral_convs.append(lateral_conv)

        # Only one single refine conv for the largest feature map
        self.refine_conv = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, inputs):
        # FPN top-down pathway
        # Start from the higher-level features (start from the smaller feature maps)
        laterals = [lateral_conv(x)
                    for lateral_conv, x in zip(self.lateral_convs, inputs[::-1])]

        for i in range(1, 1 + self.num_upsample_ops):
            laterals[i] += torch.nn.functional.interpolate(
                laterals[i - 1], size=laterals[i].shape[2:], mode='nearest')

        x = self.refine_conv(laterals[-1])
        return x


class SwinTransformer(BaseNetwork):
    """Swin Transformer, with optional FPN and input upsampling to obtain higher resolution
    feature maps"""

    def __init__(self, name, swin_net, pretrained=True, drop_path_rate=0.2,
                 input_upsample=False, use_fpn=False, fpn_level=3, fpn_out_channels=None):
        self.pretrained = pretrained
        self.drop_path_rate = drop_path_rate
        self.input_upsample = input_upsample
        self.use_fpn = use_fpn
        self.fpn_level = fpn_level
        self.fpn_out_channels = fpn_out_channels

        embed_dim = swin_net().embed_dim

        if not self.use_fpn or self.fpn_out_channels is None:
            self.out_features = 8 * embed_dim
        else:
            self.out_features = self.fpn_out_channels

        stride = 32

        if self.input_upsample:
            stride //= 2

        if self.use_fpn:
            stride //= 2 ** (4 - self.fpn_level)

        super().__init__(name, stride=stride, out_features=self.out_features)

        self.input_upsample_op = None
        if self.input_upsample:
            self.input_upsample_op = torch.nn.Upsample(scale_factor=2)

        if not self.use_fpn:
            out_indices = [3, ]
        else:
            out_indices = list(range(self.fpn_level - 1, 4))

        self.backbone = swin_net(pretrained=self.pretrained,
                                 drop_path_rate=self.drop_path_rate,
                                 out_indices=out_indices)

        self.fpn = None
        if self.use_fpn:
            self.fpn = FPN([embed_dim, 2 * embed_dim, 4 * embed_dim, 8 * embed_dim],
                           self.out_features, self.fpn_level)

    def forward(self, x):
        if self.input_upsample_op is not None:
            x = self.input_upsample_op(x)

        outs = self.backbone(x)

        if self.fpn is not None:
            x = self.fpn(outs)
        else:
            x = outs[-1]

        return x
