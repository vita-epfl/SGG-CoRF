from openpifpaf.network import HeadNetwork
import logging
import argparse
import torch
from torchvision.ops import DeformConv2d
from torch import nn
import numpy as np
import math

import openpifpaf
from openpifpaf import headmeta
from .headmeta import Raf_CAF, Raf_laplace
from .transformer_refine import TransformerRefine
LOG = logging.getLogger(__name__)

class convolution(nn.Module):
    def __init__(self, k, inp_dim, out_dim, stride=1, with_bn=True):
        super(convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = torch.nn.Conv2d(inp_dim, out_dim, (k, k), padding=(pad, pad), stride=(stride, stride), bias=not with_bn)
        self.bn   = torch.nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        #self.bn   = torch.nn.GroupNorm(num_groups=32, num_channels=out_dim) if with_bn else nn.Sequential()
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv(x)
        bn   = self.bn(conv)
        relu = self.relu(bn)
        return relu

class deform_convolution(nn.Module):
    def __init__(self, k, k_deform, inp_dim, out_dim, stride=1, with_bn=True):
        super(deform_convolution, self).__init__()

        pad = (k - 1) // 2
        self.conv = torch.nn.Conv2d(inp_dim, 2*k*k, (k, k), padding=(pad, pad), stride=(stride, stride))
        self.deform = DeformConv2d(inp_dim, out_dim, kernel_size=k_deform, padding=(pad, pad))
        self.bn   = torch.nn.BatchNorm2d(out_dim) if with_bn else nn.Sequential()
        #self.bn   = torch.nn.GroupNorm(num_groups=32, num_channels=out_dim) if with_bn else nn.Sequential()
        self.relu = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.conv(x)
        bn   = self.deform(x, offset)
        bn   = self.bn(bn)
        relu = self.relu(bn)
        return relu

def make_kp_layer(cnv_dim, curr_dim, out_dim, with_bn=False):
    return torch.nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=with_bn),
        torch.nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

def make_hm_layer(cnv_dim, curr_dim, out_dim, with_bn=False):
    return torch.nn.Sequential(
        convolution(1, cnv_dim, curr_dim, with_bn=with_bn),
        torch.nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

def make_5deep_layer(cnv_dim, curr_dim, out_dim, with_bn=False):
    return torch.nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=with_bn),
        convolution(3, curr_dim, curr_dim, with_bn=with_bn),
        convolution(3, curr_dim, curr_dim, with_bn=with_bn),
        convolution(3, curr_dim, curr_dim, with_bn=with_bn),
        torch.nn.Conv2d(curr_dim, out_dim, (1, 1))
    )

def make_corrdeep_layer(cnv_dim, curr_dim, with_bn=False):
    return torch.nn.Sequential(
        convolution(3, cnv_dim, curr_dim, with_bn=with_bn),
        convolution(3, curr_dim, curr_dim, with_bn=with_bn),
        convolution(3, curr_dim, curr_dim, with_bn=with_bn),
        convolution(3, curr_dim, curr_dim, with_bn=with_bn),
    )

def make_4deep_layer(cnv_dim, curr_dim, with_bn=False):
    return torch.nn.Sequential(
        convolution(3, cnv_dim, curr_dim, stride=2, with_bn=with_bn),
        convolution(3, curr_dim, curr_dim, with_bn=with_bn),
        convolution(3, curr_dim, curr_dim, with_bn=with_bn),
        convolution(3, curr_dim, curr_dim, with_bn=with_bn),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    )

def make_4deform_layer(cnv_dim, curr_dim, with_bn=False):
    return torch.nn.Sequential(
        deform_convolution(k=3, k_deform=3, inp_dim=cnv_dim, out_dim=curr_dim, with_bn=with_bn),
        deform_convolution(k=3, k_deform=3, inp_dim=curr_dim, out_dim=curr_dim, with_bn=with_bn),
        deform_convolution(k=3, k_deform=3, inp_dim=curr_dim, out_dim=curr_dim, with_bn=with_bn),
        deform_convolution(k=3, k_deform=3, inp_dim=curr_dim, out_dim=curr_dim, with_bn=with_bn),
    )

class CenterNetHead(HeadNetwork):

    deep4_head = False
    withBN = False
    positional_encoding = 'fourierv2'
    single_conv = False
    downup4 = False
    joint_transformer = False
    solo_transformer = False
    prior_token = None
    prior_offset = 'abs_offset'
    def __init__(self,
                 meta: headmeta.Base,
                 in_features, *,
                 kernel_size=1, padding=0, dilation=1):
        super().__init__(meta, in_features)

        LOG.debug('%s config: fields = %d, confidences = %d, vectors = %d, scales = %d '
                  'kernel = %d, padding = %d, dilation = %d',
                  meta.name, meta.n_fields, meta.n_confidences, meta.n_vectors, meta.n_scales,
                  kernel_size, padding, dilation)
        if self.downup4:
            self.downsample = torch.nn.Conv2d(in_features, in_features, (4, 4), stride=4)
            self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        if self.deform4_head:
            self.pre_head = make_4deform_layer(in_features, 256, with_bn=self.withBN)
        if self.joint_transformer or self.solo_transformer:
            self.obj_head = torch.nn.Conv2d(256, meta.n_fields + 4, (1, 1))
        elif not self.meta.single_head and not self.single_conv:
            if self.meta.kernel_1:
                self.hm = make_hm_layer(
                                in_features, 256, meta.n_fields)
                self.hm[-1].bias.data.fill_(-2.19)
                self.offset = make_hm_layer(
                                in_features, 256, 2)
                self.wh = make_hm_layer(
                                in_features, 256, 2)
            elif self.deep4_head:
                self.hm = make_5deep_layer(
                                in_features, 256, meta.n_fields * (meta.upsample_stride ** 2), with_bn=self.withBN)
                self.hm[-1].bias.data.fill_(-2.19)
                self.offset = make_5deep_layer(
                                in_features, 256, 2 * (meta.upsample_stride ** 2), with_bn=self.withBN)
                self.wh = make_5deep_layer(
                                in_features, 256, 2 * (meta.upsample_stride ** 2), with_bn=self.withBN)
            else:
                self.hm = make_kp_layer(
                                in_features, 256, meta.n_fields * (meta.upsample_stride ** 2), with_bn=self.withBN)
                self.hm[-1].bias.data.fill_(-2.19)
                self.offset = make_kp_layer(
                                in_features, 256, 2 * (meta.upsample_stride ** 2), with_bn=self.withBN)
                self.wh = make_kp_layer(
                                in_features, 256, 2 * (meta.upsample_stride ** 2), with_bn=self.withBN)
        elif self.single_conv:
            self.obj_head = torch.nn.Conv2d(in_features, meta.n_fields + 4, (1, 1))
        else:
            self.transformer = TransformerRefine(in_channels=in_features, out_channels=256, embed_dim=256, mlp_ratio=8., num_heads=8, pos_encoding_type=self.positional_encoding, drop_path_rate=0.0)
            # self.obj_head = make_hm_layer(
            #                 in_features, 256, meta.n_fields + 4)
            self.obj_head = torch.nn.Conv2d(256, meta.n_fields + 4, (1, 1))

        assert meta.upsample_stride >= 1
        if not self.prior_token is None:
            prior_input = 4
            if self.prior_token == 'predcls':
                dim_class_emb = 128
                prior_input += dim_class_emb#self.dim_class_emb
                self.class_emb = nn.Embedding(num_embeddings=meta.n_fields+1, embedding_dim=dim_class_emb)#self.dim_class_emb)
            if self.prior_token == 'sgcls_cls':
                dim_class_emb = 64
                prior_input += dim_class_emb#self.dim_class_emb
                self.class_emb = nn.Embedding(num_embeddings=2, embedding_dim=dim_class_emb)
                #self.class_emb = nn.Embedding(num_embeddings=2, embedding_dim=dim_class_emb)
            self.proj_prior = nn.Linear(prior_input, 256)
        # self.upsample_op = None

        # if meta.upsample_stride > 1:
        #     self.upsample_op = torch.nn.PixelShuffle(meta.upsample_stride)

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""
        group = parser.add_argument_group('CompositeField3')
        group.add_argument('--cntrnet-deform-bn',
                            default=False, action='store_true',
                            help='Use BN in the head')
        group.add_argument('--cntrnet-deform-deep4-head',
                            default=False, action='store_true',
                            help='Use 4 3x3 conv')
        group.add_argument('--cntrnet-deform-deform4-head',
                            default=False, action='store_true',
                            help='Use 4 3x3 deform_conv')
        group.add_argument('--cntrnet-deform-positional-encoding',
                           choices=('fourierv2', 'fourier', 'learned1d', 'learned2d'),
                           default=cls.positional_encoding,
                           help=('Positional Encoding used by Transformer module'))
        group.add_argument('--cntrnet-deform-single-conv',
                            default=False, action='store_true',
                            help='Use 1 1x1 deform_conv')
        group.add_argument('--cntrnet-deform-downup4',
                            default=False, action='store_true',
                            help='Downsample and upsample by 4')
        group.add_argument('--cntrnet-deform-joint-transf',
                            default=False, action='store_true',
                            help='Use Joint transformer neck')
        group.add_argument('--cntrnet-deform-solo-transf',
                            default=False, action='store_true',
                            help='Use Solo transformer neck')
        group.add_argument('--cntrnet-deform-prior-token',
                            default=None, type=str, choices=[None, 'predcls', 'sgcls', 'sgcls_cls'],
                            help='Include prior in input to transformer')
        group.add_argument('--cntrnet-deform-prior-offset',
                            default='abs_offset', type=str, choices=['abs_offset', 'rel_offset'],
                            help='Type of prior offset')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""
        cls.deep4_head = args.cntrnet_deform_deep4_head
        cls.deform4_head = args.cntrnet_deform_deform4_head
        cls.withBN = args.cntrnet_deform_bn
        cls.positional_encoding = args.cntrnet_deform_positional_encoding
        cls.single_conv = args.cntrnet_deform_single_conv
        cls.downup4 = args.cntrnet_deform_downup4
        cls.joint_transformer = args.cntrnet_deform_joint_transf
        cls.solo_transformer = args.cntrnet_deform_solo_transf
        cls.prior_token = args.cntrnet_deform_prior_token
        cls.prior_offset = args.cntrnet_deform_prior_offset

    def forward(self, x, targets=None):  # pylint: disable=arguments-differ
        tensor_toconcat = []
        prior_info = None
        if isinstance(x, tuple):
            x = x[0]
        if not targets is None and not self.prior_token is None:
            unsqueeze = False
            width = targets[0][0].shape[-1]
            wh = targets[0][2]#.detach()
            reg = targets[0][1]#.detach()
            if self.prior_offset == 'abs_offset':
                ind_x = (targets[0][4]%width)#.detach()
                ind_y = ((targets[0][4] - ind_x)/width)#.detach()

            if len(wh.shape) == 2:
                unsqueeze = True
                wh = wh.unsqueeze(0).to(device=x.device)
                reg = reg.unsqueeze(0).to(device=x.device)
            if self.prior_offset == 'abs_offset':
                ind_x = ind_x.unsqueeze(0).to(device=x.device)
                ind_y = ind_y.unsqueeze(0).to(device=x.device)
                reg[:,:,0] = reg[:,:,0]  + ind_x
                reg[:,:,1] = reg[:,:,1]  + ind_y

            if self.prior_token == 'predcls' or self.prior_token == 'sgcls_cls':
                if self.prior_token == 'predcls':
                    cls = targets[0][-1]#.detach()
                else:
                    cls = torch.zeros(*targets[0][-1].shape, dtype=torch.long, device=x.device)
                    cls[targets[0][-1]>0] = 1
                if len(cls.shape) == 1:
                    cls = cls.unsqueeze(0).to(device=x.device)
                cls_embed = self.class_emb(cls)
                prior_tokens = torch.cat([reg, wh, cls_embed], dim=2)
            else:
                prior_tokens = torch.cat([reg, wh], dim=2)
            prior_tokens = self.proj_prior(prior_tokens)
            prior_info = (prior_tokens, targets[0][4].unsqueeze(0).to(x.device) if unsqueeze else targets[0][4]) #.detach())
        if self.downup4:
            x = self.downsample(x)
        if self.deform4_head:
            x = self.pre_head(x)

        if self.single_conv or self.joint_transformer or self.solo_transformer:
            x = self.obj_head(x)
            x[:,:self.meta.n_fields, :,:] = torch.sigmoid(x[:,:self.meta.n_fields, :,:])
        elif not self.meta.single_head:
            output_hm = self.hm(x)
            output_hm = torch.sigmoid_(output_hm)


            output_reg = self.offset(x)
            output_wh = self.wh(x)
            if self.meta.giou_loss:
                output_wh = torch.nn.functional.softplus(output_wh)

            # if self.upsample_op is not None:
            #     output_hm = self.upsample_op(output_hm)
            #     output_reg = self.upsample_op(output_reg)
            #     output_wh = self.upsample_op(output_wh)

            tensor_toconcat.append(output_hm)
            tensor_toconcat.append(output_reg)
            tensor_toconcat.append(output_wh)

            x = torch.cat(tensor_toconcat, dim=1)
        else:
            x = self.transformer(x, prior_info=prior_info)
            if self.downup4:
                x = self.upsample(x)
            x = self.obj_head(x)
            x[:,:self.meta.n_fields, :,:] = torch.sigmoid(x[:,:self.meta.n_fields, :,:])

        # if self.upsample_op is not None:
        #     low_cut = (self.meta.upsample_stride - 1) // 2
        #     high_cut = math.ceil((self.meta.upsample_stride - 1) / 2.0)
        #     if self.training:
        #         # negative axes not supported by ONNX TensorRT
        #         x = x[:, :, low_cut:-high_cut, low_cut:-high_cut]
        #     else:
        #         # the int() forces the tracer to use static shape
        #         x = x[:, :, low_cut:int(x.shape[2]) - high_cut, low_cut:int(x.shape[3]) - high_cut]

        x_size = x.size()
        batch_size = int(x_size[0])
        feature_height = int(x_size[2])
        feature_width = int(x_size[3])

        x = x.view(
            batch_size,
            self.meta.n_fields + 4,
            feature_height,
            feature_width
        )

        return x

class CompositeField3_singleDeform_fourier(HeadNetwork):
    inplace_ops = True
    mapping_size = 128
    multipleheads = False
    extra_conv_offset = False
    separate_offset = False
    use_transformer = False
    deep4_head = False
    deform4_head = False
    withBN = False
    positional_encoding = 'fourierv2'
    downup4 = False
    joint_transformer = False
    solo_transformer = False
    prior_token = None
    prior_offset = 'abs_offset'
    rndm_vect = False
    def __init__(self,
                 meta: Raf_CAF,
                 in_features, *,
                 kernel_size=1, padding=0, dilation=1):
        super().__init__(meta, in_features)

        LOG.debug('%s config: fields = %d, confidences = %d, vectors = %d, scales = %d '
                  'kernel = %d, padding = %d, dilation = %d',
                  meta.name, meta.n_fields, meta.n_confidences, meta.n_vectors, meta.n_scales,
                  kernel_size, padding, dilation)

        if self.downup4:
            self.downsample = torch.nn.Conv2d(in_features, in_features, (4, 4), stride=4)
            self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        in_features_refined = in_features

        if self.meta.refine_feature or self.meta.biggerparam or self.meta.deform_conv:
            in_features_refined = 3*in_features
            if self.deep4_head:
                in_features_refined = in_features_refined*2
            if self.meta.concat_offset:
                in_features_refined = in_features_refined + 4
            elif self.meta.fourier_features:
                in_features_refined = in_features_refined + 256

        if isinstance(self.meta, Raf_laplace):
            out_features = meta.n_fields * (meta.n_confidences + meta.n_vectors * 3 + meta.n_scales)
        elif self.separate_offset:
            print("Separating offset from other fields")
            out_features = meta.n_fields * (meta.n_confidences + meta.n_scales)
        else:
            out_features = meta.n_fields * (meta.n_confidences + meta.n_vectors * 2 + meta.n_scales)

        if self.multipleheads:
            self.raf_hm = make_hm_layer(
                            in_features_refined, 256, meta.n_fields)
            self.raf_subj = make_hm_layer(
                            in_features_refined, 256, meta.n_fields*2)
            self.raf_obj = make_hm_layer(
                            in_features_refined, 256, meta.n_fields*2)
            self.raf_scales = make_hm_layer(
                            in_features_refined, 256, meta.n_fields*2)
        elif self.deform4_head:
            self.pre_head = make_4deform_layer(in_features, 256, with_bn=self.withBN)
            in_features = 256
            in_features_refined = 3*in_features
            if self.meta.refine_feature or self.meta.biggerparam or self.meta.deform_conv:
                self.raf_head = make_hm_layer(
                                in_features_refined, in_features, out_features, with_bn=self.withBN)
            else:
                self.raf_head = torch.nn.Conv2d(256, out_features,
                                            kernel_size=1, padding=0, dilation=1)
        elif self.deep4_head:
            #self.pre_head = make_4deep_layer(in_features, in_features*2, with_bn=self.withBN)
            self.pre_head = make_corrdeep_layer(in_features, 256, with_bn=self.withBN)
            #in_features = in_features*2
            in_features = 256
            in_features_refined = 3*in_features
            if self.meta.refine_feature or self.meta.biggerparam or self.meta.deform_conv:
                self.raf_head = make_hm_layer(
                                in_features_refined, in_features, out_features, with_bn=self.withBN)
            else:
                self.raf_head = torch.nn.Conv2d(256, out_features,
                                            kernel_size=1, padding=0, dilation=1)
        elif self.joint_transformer or self.solo_transformer:
            self.raf_head = torch.nn.Conv2d(256, out_features,
                                        kernel_size=1, padding=0, dilation=1)
        else:
            if self.meta.refine_feature or self.meta.biggerparam or self.meta.deform_conv:
                self.raf_head = make_hm_layer(
                                in_features_refined, 256, out_features, with_bn=self.withBN)
            else:
                if self.use_transformer:
                    self.raf_head = torch.nn.Conv2d(256, out_features,
                                                kernel_size=1, padding=0, dilation=1)
                else:
                    self.raf_head = torch.nn.Conv2d(in_features, out_features,
                                                kernel_size=1, padding=0, dilation=1)
            # self.raf_head = make_hm_layer(
            #                 in_features_refined, 256, out_features, with_bn=self.withBN)

        if self.use_transformer:
            #self.transformer = TransformerRefine(in_channels=in_features, out_channels=in_features, embed_dim=768, num_heads=12)
            self.transformer = TransformerRefine(in_channels=in_features, out_channels=256, embed_dim=256, mlp_ratio=8., num_heads=8, pos_encoding_type=self.positional_encoding, drop_path_rate=0.0)

        if self.meta.refine_feature or self.meta.deform_conv:
            if self.separate_offset:
                # self.offset_conv = torch.nn.Conv2d(in_features, meta.n_fields*meta.n_vectors * 2,
                #                             kernel_size=1, padding=0, dilation=1)
                self.offset_conv = make_kp_layer(
                                in_features, 256, meta.n_fields*meta.n_vectors * 2)
            else:
                self.offset_conv = torch.nn.Conv2d(in_features, meta.n_vectors * 2,
                                            kernel_size=1, padding=0, dilation=1)
                # self.offset_conv = make_kp_layer(
                #                 in_features, 256, meta.n_vectors * 2)
            self.deform_single = DeformConv2d(3*in_features, 3*in_features, kernel_size=1, groups=3, bias=False)
            if self.extra_conv_offset:
                if self.separate_offset:
                    self.extra_conv = nn.Conv2d(
                                            meta.n_fields*meta.n_vectors * 2,
                                            meta.n_vectors * 2,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            bias=False)
                else:
                    self.extra_conv = nn.Conv2d(
                                            meta.n_vectors * 2,
                                            meta.n_vectors * 2,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0,
                                            bias=False)
            self.relu_refined = torch.nn.ReLU(inplace=True)

            B = torch.randn((self.mapping_size, 4), device=next(self.offset_conv.parameters()).device)
            self.B = torch.nn.Parameter(B, requires_grad=False)
        assert meta.upsample_stride >= 1
        self.upsample_op = None
        if meta.upsample_stride > 1:
            self.upsample_op = torch.nn.PixelShuffle(meta.upsample_stride)
        if not self.prior_token is None:
            prior_input = 4
            dim_class_emb = 64
            if self.prior_token == 'predcls':
                prior_input += dim_class_emb#self.dim_class_emb
                self.class_emb = nn.Embedding(num_embeddings=len(meta.obj_categories)+1, embedding_dim=dim_class_emb)
            if self.prior_token == 'sgcls_cls' or self.prior_token == 'prior_vect_cls':
                prior_input += dim_class_emb#self.dim_class_emb
                self.class_emb = nn.Embedding(num_embeddings=2, embedding_dim=dim_class_emb)#self.dim_class_emb)
            if self.prior_token == 'prior_vect_detcls':
                self.class_emb_det = nn.Embedding(num_embeddings=len(meta.obj_categories)+1, embedding_dim=dim_class_emb)
                self.proj_prior_det = nn.Linear(68, 256)
            if self.prior_token == 'prior_vect_det':
                self.proj_prior_det = nn.Linear(4, 256)
            self.proj_prior = nn.Linear(prior_input, 256)
    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        """Command line interface (CLI) to extend argument parser."""
        group = parser.add_argument_group('CompositeField3')
        group.add_argument('--cf3-deform-multipleheads',
                            default=cls.multipleheads, action='store_true',
                            help='Have multiple heads for Raf')
        group.add_argument('--cf3-deform-extra-offset-conv',
                            default=False, action='store_true',
                            help='Have multiple heads for Raf')
        group.add_argument('--cf3-deform-separate-offset',
                            default=False, action='store_true',
                            help='Do not add offsets together')
        group.add_argument('--cf3-deform-use-transformer',
                            default=False, action='store_true',
                            help='Use Transformer Refine after backbone')
        group.add_argument('--cf3-deform-bn',
                            default=False, action='store_true',
                            help='Use BN in the head')
        group.add_argument('--cf3-deform-deep4-head',
                            default=False, action='store_true',
                            help='Use 4 3x3 conv')
        group.add_argument('--cf3-deform-deform4-head',
                            default=False, action='store_true',
                            help='Use 4 3x3 deform_conv')
        group.add_argument('--cf3-deform-positional-encoding',
                           choices=('fourierv2', 'fourier', 'learned1d', 'learned2d'),
                           default=cls.positional_encoding,
                           help=('Positional Encoding used by Transformer module'))
        group.add_argument('--cf3-deform-downup4',
                            default=False, action='store_true',
                            help='Downsample and upsample by 4')
        group.add_argument('--cf3-deform-joint-transf',
                            default=False, action='store_true',
                            help='Use Joint transformer neck')
        group.add_argument('--cf3-deform-solo-transf',
                            default=False, action='store_true',
                            help='Use Solo transformer neck')
        group.add_argument('--cf3-deform-prior-token',
                            default=None, type=str, choices=[None, 'predcls', 'sgcls', 'prior_vect', 'prior_vect_cls', 'prior_vect_detcls', 'prior_vect_det', 'sgcls_cls'],
                            help='Include prior in input to transformer')
        group.add_argument('--cf3-deform-prior-offset',
                            default='abs_offset', type=str, choices=['abs_offset', 'rel_offset'],
                            help='Type of prior offset')
        group.add_argument('--cf3-deform-prior-rndm-vect',
                            default=False, action='store_true',
                            help='add rndm vectors to prior')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""
        cls.multipleheads = args.cf3_deform_multipleheads
        cls.extra_conv_offset = args.cf3_deform_extra_offset_conv
        cls.separate_offset = args.cf3_deform_separate_offset
        cls.use_transformer = args.cf3_deform_use_transformer
        cls.deep4_head = args.cf3_deform_deep4_head
        cls.deform4_head = args.cf3_deform_deform4_head
        cls.withBN = args.cf3_deform_bn
        cls.positional_encoding = args.cf3_deform_positional_encoding
        cls.downup4 = args.cf3_deform_downup4
        cls.joint_transformer = args.cf3_deform_joint_transf
        cls.solo_transformer = args.cf3_deform_solo_transf
        cls.prior_token = args.cf3_deform_prior_token
        cls.prior_offset = args.cf3_deform_prior_offset
        cls.rndm_vect = args.cf3_deform_prior_rndm_vect

    def _build_map(self, data_pts, indices, wh):
        #import pdb; pdb.set_trace()
        #output_map = torch.zeros(wh[1], wh[0], wh[2], wh[3])
        output_map = torch.zeros(wh, device=data_pts.device)
        x_indices = indices%wh[3]
        y_indices = (indices - x_indices)//wh[3]
        modified_view = data_pts.permute(0, 2, 1)
        for idx in range(wh[0]):
            output_map[idx, :, y_indices[idx], x_indices[idx]] = modified_view[idx,[1,0],:]

        return output_map

    def location_fourier_mapping(self,x):
        x_proj = (2*np.pi*x).permute(0,2,3,1) @ self.B.t()
        x_proj = x_proj.permute(0,3,1,2)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=1)

    def forward(self, x, targets=None):  # pylint: disable=arguments-differ
        tensor_toconcat = []
        prior_info = None
        if not targets is None and not self.prior_token is None:
            if self.prior_token == 'prior_vect' or self.prior_token == 'prior_vect_cls' or self.prior_token == 'prior_vect_detcls' or self.prior_token == 'prior_vect_det':
                squeezed = False
                start_idx = 0
                if self.meta.pairwise:
                    start_idx = 9
                vect = torch.zeros((x.shape[0], 1024, 4)).to(x.device)
                if len(targets[1].shape) == 4:
                    squeezed = True
                    targets[1] = targets[1].unsqueeze(0)
                mask = (targets[1][:,:,start_idx+0]==1)#.expand(-1, -1, 4, -1,-1)
                pos_mask = ((targets[1][:,:,start_idx+0]).max(dim=1)[0]>0).view(x.shape[0], -1)
                pos_vect = torch.zeros((x.shape[0], 1024)).to(x.device)
                t_range = torch.arange(pos_mask.shape[-1])
                r1, r2 = targets[1].shape[-2:]
                r1 -= 1
                r2 -= 1
                r1 = -r1
                for batch_idx in range(x.shape[0]):
                    vect[batch_idx, :mask[batch_idx].sum(), :] = targets[1][batch_idx,:,(start_idx+1):(start_idx+5)].permute(0,2,3,1)[mask[batch_idx], :]
                    pos_vect[batch_idx, :pos_mask[batch_idx].sum()] = t_range[pos_mask[batch_idx]]
                    if self.rndm_vect:
                        vect[batch_idx, mask[batch_idx].sum():, :] = (r1 - r2) * torch.rand(1024-mask[batch_idx].sum(), 4) + r2
                        pos_vect[batch_idx, pos_mask[batch_idx].sum():] = t_range[~pos_mask[batch_idx]]
                if self.prior_token == 'prior_vect_cls':
                    cls = torch.zeros((*pos_vect.shape), dtype=torch.long, device=x.device)
                    cls[pos_mask] = 1
                    cls_embed = self.class_emb(cls)
                    vect = torch.cat([vect, cls_embed], dim=2)
                if squeezed:
                    targets[1] = targets[1].squeeze(0)
                vect = self.proj_prior(vect)
                if self.prior_token == 'prior_vect_detcls' or self.prior_token == 'prior_vect_det':
                    unsqueeze = False
                    width = targets[0][0].shape[-1]
                    wh = targets[0][2]#.detach()
                    reg = targets[0][1]#.detach()
                    if len(wh.shape) == 2:
                        unsqueeze = True
                        wh = wh.unsqueeze(0).to(device=x.device)
                        reg = reg.unsqueeze(0).to(device=x.device)
                    if self.prior_token == 'prior_vect_detcls':
                        cls = targets[0][-1]#.detach()
                        if len(cls.shape) == 1:
                            cls = cls.unsqueeze(0).to(device=x.device)
                        cls_embed = self.class_emb_det(cls)
                        prior_tokens = torch.cat([reg, wh, cls_embed], dim=2)
                    else:
                        prior_tokens = torch.cat([reg, wh], dim=2)
                    prior_tokens = self.proj_prior_det(prior_tokens)
                    vect = torch.cat([vect, prior_tokens], dim=1)
                    pos_vect = torch.cat([pos_vect, targets[0][4].unsqueeze(0).to(x.device) if unsqueeze else targets[0][4]], dim=1)
                prior_info = (vect, pos_vect.long())
            else:
                unsqueeze = False
                width = targets[0][0].shape[-1]
                wh = targets[0][2]#.detach()
                reg = targets[0][1]#.detach()
                if self.prior_offset == 'abs_offset':
                    ind_x = (targets[0][4]%width)#.detach()
                    ind_y = ((targets[0][4] - ind_x)/width)#.detach()

                if len(wh.shape) == 2:
                    unsqueeze = True
                    wh = wh.unsqueeze(0).to(device=x.device)
                    reg = reg.unsqueeze(0).to(device=x.device)
                if self.prior_offset == 'abs_offset':
                    ind_x = ind_x.unsqueeze(0).to(device=x.device)
                    ind_y = ind_y.unsqueeze(0).to(device=x.device)
                    reg[:,:,0] = reg[:,:,0] + ind_x
                    reg[:,:,1] = reg[:,:,1] + ind_y

                if self.prior_token == 'predcls' or self.prior_token == 'sgcls_cls':
                    if self.prior_token == 'predcls':
                        cls = targets[0][-1]#.detach()
                    else:
                        cls = torch.zeros(*targets[0][-1].shape, dtype=torch.long, device=x.device)
                        cls[targets[0][-1]>0] = 1
                    if len(cls.shape) == 1:
                        cls = cls.unsqueeze(0).to(device=x.device)
                    cls_embed = self.class_emb(cls)
                    prior_tokens = torch.cat([reg, wh, cls_embed], dim=2)
                else:
                    prior_tokens = torch.cat([reg, wh], dim=2)
                prior_tokens = self.proj_prior(prior_tokens)
                prior_info = (prior_tokens, targets[0][4].unsqueeze(0).to(x.device) if unsqueeze else targets[0][4])#.detach())

        if self.downup4:
            x = self.downsample(x)
        if self.deep4_head or self.deform4_head:
            x = self.pre_head(x)
        if self.use_transformer:
            x = self.transformer(x, prior_info=prior_info)
        if self.meta.refine_feature or self.meta.deform_conv:
            # output_hm = self.hm(refined_x)
            offset_1 = self.offset_conv(x)
            if self.extra_conv_offset:
                offset_extra = self.extra_conv(offset_1.detach())
            # if self.meta.detach_offset:
            #     offset_1 = offset_1.detach()
            if self.extra_conv_offset:
                x = self.relu_refined(self.deform_single(x.repeat(1,3,1,1),
                    torch.cat([offset_extra, torch.zeros((offset_extra.shape[0], 2, *offset_extra.shape[-2:]), device=offset_extra.device)], dim=1)))
            else:
                x = self.relu_refined(self.deform_single(x.repeat(1,3,1,1),
                    torch.cat([offset_1[:,[1,0,3,2], :,:], torch.zeros((offset_1.shape[0], 2, *offset_1.shape[-2:]), device=offset_1.device)], dim=1)))


            # if self.meta.hadamard_product:
            #     x = refined_pred*refined_subj*refined_obj

            if self.meta.concat_offset:
                x = torch.cat([x, offset_1], dim=1)
            elif self.meta.fourier_features:
                x = torch.cat([x, self.location_fourier_mapping(offset_1)], dim=1)
        if self.multipleheads:
            raf_hm = self.raf_hm(x)
            raf_subj = self.raf_subj(x)
            raf_obj = self.raf_obj(x)
            raf_scales = self.raf_scales(x)
            x = torch.cat([raf_hm, raf_subj, raf_obj, raf_scales], dim=1)
        else:
            if self.downup4:
                x = self.upsample(x)
            x = self.raf_head(x)

        # upscale
        if self.upsample_op is not None:
            x = self.upsample_op(x)
            low_cut = (self.meta.upsample_stride - 1) // 2
            high_cut = math.ceil((self.meta.upsample_stride - 1) / 2.0)
            if self.training:
                # negative axes not supported by ONNX TensorRT
                x = x[:, :, low_cut:-high_cut, low_cut:-high_cut]
            else:
                # the int() forces the tracer to use static shape
                x = x[:, :, low_cut:int(x.shape[2]) - high_cut, low_cut:int(x.shape[3]) - high_cut]

        x_size = x.size()
        batch_size = int(x_size[0])
        feature_height = int(x_size[2])
        feature_width = int(x_size[3])

        if isinstance(self.meta, Raf_laplace):
            x = x.view(
                batch_size,
                self.meta.n_fields,
                self.meta.n_confidences + self.meta.n_vectors * 3 + self.meta.n_scales,
                feature_height,
                feature_width
            )

            if self.meta.refine_feature:
                if not self.meta.detach_offset and not self.separate_offset:
                    x[:,:,self.meta.n_confidences:(self.meta.n_confidences + self.meta.n_vectors * 2),:,:] = \
                        x[:,:,self.meta.n_confidences:(self.meta.n_confidences + self.meta.n_vectors * 2),:,:] + offset_1.unsqueeze(1)

            if not self.training and self.inplace_ops:
                # classification
                classes_x = x[:, :, 0:self.meta.n_confidences]
                torch.sigmoid_(classes_x)

                # regressions x: add index
                if self.meta.n_vectors > 0:
                    index_field = openpifpaf.network.heads.index_field_torch((feature_height, feature_width), device=x.device)
                    first_reg_feature = self.meta.n_confidences
                    for i, do_offset in enumerate(self.meta.vector_offsets):
                        if not do_offset:
                            continue
                        reg_x = x[:, :, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                        reg_x.add_(index_field)

                # scale
                first_scale_feature = self.meta.n_confidences + self.meta.n_vectors * 3
                scales_x = x[:, :, first_scale_feature:first_scale_feature + self.meta.n_scales]

                scales_x[:] = torch.nn.functional.softplus(scales_x)
            elif not self.training and not self.inplace_ops:
                # TODO: CoreMLv4 does not like strided slices.
                # Strides are avoided when switching the first and second dim
                # temporarily.
                x = torch.transpose(x, 1, 2)

                # classification
                classes_x = x[:, 0:self.meta.n_confidences]
                classes_x = torch.sigmoid(classes_x)

                # regressions x
                first_reg_feature = self.meta.n_confidences
                regs_x = [
                    x[:, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                    for i in range(self.meta.n_vectors)
                ]
                # regressions x: add index
                index_field = openpifpaf.network.heads.index_field_torch(
                    (feature_height, feature_width), device=x.device, unsqueeze=(1, 0))
                # TODO: coreml export does not work with the index_field creation in the graph.
                index_field = torch.from_numpy(index_field.numpy())
                regs_x = [reg_x + index_field if do_offset else reg_x
                          for reg_x, do_offset in zip(regs_x, self.meta.vector_offsets)]

                # regressions logb
                first_reglogb_feature = self.meta.n_confidences + self.meta.n_vectors * 2
                regs_logb = x[:, first_reglogb_feature:first_reglogb_feature + self.meta.n_vectors]

                # scale
                first_scale_feature = self.meta.n_confidences + self.meta.n_vectors * 3
                scales_x = x[:, first_scale_feature:first_scale_feature + self.meta.n_scales]

                scales_x = torch.nn.functional.softplus(scales_x)

                # concat
                x = torch.cat([classes_x, *regs_x, regs_logb, scales_x], dim=1)

                # TODO: CoreMLv4 problem (see above).
                x = torch.transpose(x, 1, 2)

        else:
            if self.separate_offset:
                x = torch.cat([x[:, :self.meta.n_fields*self.meta.n_confidences], offset_1, x[:, self.meta.n_fields*self.meta.n_confidences:]], dim=1)
            x = x.view(
                batch_size,
                self.meta.n_fields,
                self.meta.n_confidences + self.meta.n_vectors * 2 + self.meta.n_scales,
                feature_height,
                feature_width
            )

            if self.meta.refine_feature:
                if not self.meta.detach_offset and not self.separate_offset:
                    x[:,:,self.meta.n_confidences:(self.meta.n_confidences + self.meta.n_vectors * 2),:,:] = \
                        x[:,:,self.meta.n_confidences:(self.meta.n_confidences + self.meta.n_vectors * 2),:,:] + offset_1.unsqueeze(1)

            if not self.training and self.inplace_ops:
                # classification
                classes_x = x[:, :, 0:self.meta.n_confidences]
                torch.sigmoid_(classes_x)

                # regressions x: add index
                if self.meta.n_vectors > 0:
                    index_field = openpifpaf.network.heads.index_field_torch((feature_height, feature_width), device=x.device)
                    first_reg_feature = self.meta.n_confidences
                    for i, do_offset in enumerate(self.meta.vector_offsets):
                        if not do_offset:
                            continue
                        reg_x = x[:, :, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                        reg_x.add_(index_field)

                # scale
                first_scale_feature = self.meta.n_confidences + self.meta.n_vectors * 2
                scales_x = x[:, :, first_scale_feature:first_scale_feature + self.meta.n_scales]

                scales_x[:] = torch.nn.functional.softplus(scales_x)
            elif not self.training and not self.inplace_ops:
                # TODO: CoreMLv4 does not like strided slices.
                # Strides are avoided when switching the first and second dim
                # temporarily.
                x = torch.transpose(x, 1, 2)

                # classification
                classes_x = x[:, 0:self.meta.n_confidences]
                classes_x = torch.sigmoid(classes_x)

                # regressions x
                first_reg_feature = self.meta.n_confidences
                regs_x = [
                    x[:, first_reg_feature + i * 2:first_reg_feature + (i + 1) * 2]
                    for i in range(self.meta.n_vectors)
                ]
                # regressions x: add index
                index_field = openpifpaf.network.heads.index_field_torch(
                    (feature_height, feature_width), device=x.device, unsqueeze=(1, 0))
                # TODO: coreml export does not work with the index_field creation in the graph.
                index_field = torch.from_numpy(index_field.numpy())
                regs_x = [reg_x + index_field if do_offset else reg_x
                          for reg_x, do_offset in zip(regs_x, self.meta.vector_offsets)]

                # regressions logb
                first_reglogb_feature = self.meta.n_confidences + self.meta.n_vectors * 2
                regs_logb = x[:, first_reglogb_feature:first_reglogb_feature + self.meta.n_vectors]

                # scale
                first_scale_feature = self.meta.n_confidences + self.meta.n_vectors * 2
                scales_x = x[:, first_scale_feature:first_scale_feature + self.meta.n_scales]

                scales_x = torch.nn.functional.softplus(scales_x)

                # concat
                x = torch.cat([classes_x, *regs_x, regs_logb, scales_x], dim=1)

                # TODO: CoreMLv4 problem (see above).
                x = torch.transpose(x, 1, 2)
        return x
