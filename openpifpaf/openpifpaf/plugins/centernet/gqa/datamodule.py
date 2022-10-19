gqa-import argparse
import numpy as np
import torch
import torchvision

import openpifpaf

from .visual_genome import VG
from .constants import BBOX_KEYPOINTS, BBOX_HFLIP, OBJ_CATEGORIES, REL_CATEGORIES, REL_CATEGORIES_FLIP
#from ...raf import headmeta
from ..headmeta import CenterNet, CifDet_CN, Raf_CN, Raf_CNs, Raf_CAF,Raf_GDeform, Raf_laplace, Raf_FPN, CenterNet_FPN
from ..encoder import CenterNetEncoder, RafEncoder
from ..prior_encoder import PriorEncoder
from ..cifdet_cn import CifDet_CN as CifDet_CNEncoder
from ..raf_cn import Raf_CN as Raf_CNEncoder
from ...raf.raf import Raf
from ...raf.toannotations import Raf_HFlip
from ..transforms.toannotation import ToRafAnnotations
from ...raf.vg_h5 import metric
from ..transforms import Crop, Prior_HM
from ..maskhm_encoder import MaskHMEncoder

from ..fpn_encoder import SharedRaf_CN, SharedCenterNetEncoder

class GQAModule(openpifpaf.datasets.DataModule):
    data_dir = "data"

    debug = False
    pin_memory = False

    n_images = -1
    square_edge = [512]
    extended_scale = False
    orientation_invariant = 0.0
    blur = 0.0
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    special_preprocess = False
    max_long_edge = False
    no_flipping = False
    use_dcn = False
    supervise_offset = False
    only_det = False
    ignore_rel = False
    caf_bigger = False
    refine_feature = False
    laplace = False
    detach_deform = False
    detach_offset = False
    remove_duplicate = True
    filter_non_overlap = False
    hadamard_product = False
    gqa_rgb = False
    concat_offset = False
    group_deform=False
    fourier_features = False
    prior = None
    giou_loss = False
    mask_raf = False
    no_scale = False
    kernel_1 = False
    single_head = False
    single_supervision = False

    eval_long_edge = None
    eval_orientation_invariant = 0.0
    eval_extended_scale = False
    obj_categories = OBJ_CATEGORIES
    rel_categories = REL_CATEGORIES
    #gqa_512 = False
    use_fpn = False

    def __init__(self):
        super().__init__()
        self.head_metas = []
        # if self.use_dcn or self.supervise_offset:
        #     raf_1 = headmeta.Raf_dcn('raf_1', 'gqa_cn', self.obj_categories, self.rel_categories)
        #     raf_1.n_offsets = 2 if self.supervise_offset else 0
        #     raf_1.ignore_rel = self.ignore_rel
        #     raf_1.upsample_stride = self.upsample_stride
        #
        #     raf_2 = headmeta.Raf_dcn('raf_2', 'gqa_cn', self.obj_categories, self.rel_categories)
        #     raf_2.n_offsets = 2 if self.supervise_offset else 0
        #     raf_2.ignore_rel = self.ignore_rel
        #     raf_2.upsample_stride = self.upsample_stride
        # else:
        name = 'raf'
        if self.caf_bigger:
            chosen_raf = Raf_CAF
        elif self.laplace:
            chosen_raf = Raf_laplace
        elif self.group_deform:
            chosen_raf = Raf_GDeform
        elif self.use_fpn:
            chosen_raf = Raf_FPN
        else:
            chosen_raf = Raf_CN

        # raf_1 = Raf_CNs('rafs_1', 'gqa_cn', self.obj_categories, self.rel_categories)
        # raf_2 = Raf_CNs('rafs_2', 'gqa_cn', self.obj_categories, self.rel_categories)
        raf_1 = chosen_raf(name+'_1', 'gqa_cn', self.obj_categories, self.rel_categories)
        raf_2 = chosen_raf(name+'_2', 'gqa_cn', self.obj_categories, self.rel_categories)
        raf_1.refine_feature = self.refine_feature
        raf_2.refine_feature = self.refine_feature
        raf_1.detach_offset = self.detach_offset
        raf_2.detach_offset = self.detach_offset
        raf_1.detach_deform = self.detach_deform
        raf_2.detach_deform = self.detach_deform
        raf_1.hadamard_product = self.hadamard_product
        raf_2.hadamard_product = self.hadamard_product
        raf_1.fourier_features = self.fourier_features
        raf_2.fourier_features = self.fourier_features
        raf_1.concat_offset = self.concat_offset
        raf_2.concat_offset = self.concat_offset
        if self.no_scale:
            raf_1.n_scales = 0
            raf_2.n_scales = 0
        if self.use_fpn:
            raf_1.fpn_idx = ['0', '1', '2', '3']
            raf_1.fpn_stride = {'0':8, '1':4, '2':2, '3':1}

            raf_2.fpn_idx = ['0', '1', '2', '3']
            raf_2.fpn_stride = {'0':8, '1':4, '2':2, '3':1}



        raf_1.upsample_stride = self.upsample_stride
        raf_2.upsample_stride = self.upsample_stride

        if not self.cifdet_cn:
            if self.use_fpn:
                cifdet_1 = CenterNet_FPN('centernet_1', 'gqa_cn', self.obj_categories)
                cifdet_2 = CenterNet_FPN('centernet_2', 'gqa_cn', self.obj_categories)

                cifdet_1.fpn_idx = ['0', '1', '2', '3']
                cifdet_1.fpn_stride = {'0':8, '1':4, '2':2, '3':1}

                cifdet_2.fpn_idx = ['0', '1', '2', '3']
                cifdet_2.fpn_stride = {'0':8, '1':4, '2':2, '3':1}
            else:
                cifdet_1 = CenterNet('centernet_1', 'gqa_cn', self.obj_categories)
                cifdet_2 = CenterNet('centernet_2', 'gqa_cn', self.obj_categories)

            cifdet_1.prior = self.prior
            cifdet_2.prior = self.prior
            cifdet_1.kernel_1 = self.kernel_1
            cifdet_2.kernel_1 = self.kernel_1
            cifdet_1.single_head = self.single_head
            cifdet_2.single_head = self.single_head
        else:
            if self.laplace:
                cifdet_1 = openpifpaf.headmeta.CifDet('cifdet_1', 'gqa_cn', self.obj_categories)
                cifdet_2 = openpifpaf.headmeta.CifDet('cifdet_2', 'gqa_cn', self.obj_categories)
            else:
                cifdet_1 = CifDet_CN('cifdetcn_1', 'gqa_cn', self.obj_categories)
                cifdet_2 = CifDet_CN('cifdetcn_2', 'gqa_cn', self.obj_categories)
        #cifdet = headmeta.CifDet_deep('cifdet', 'vg', self.obj_categories)
        #cifdet = headmeta.CifDet_deepShared('cifdet', 'vg', self.obj_categories)
        cifdet_1.giou_loss = self.giou_loss
        cifdet_2.giou_loss = self.giou_loss
        cifdet_1.upsample_stride = self.upsample_stride
        cifdet_2.upsample_stride = self.upsample_stride
        if not self.single_supervision:
            self.head_metas.append(cifdet_1)
            self.head_metas.append(cifdet_2)

            if not self.only_det:
                self.head_metas.append(raf_1)
                self.head_metas.append(raf_2)
        else:
            cifdet_1.name = 'centernet' if not self.cifdet_cn else 'cifdet'
            self.head_metas.append(cifdet_1)

            if not self.only_det:
                raf_1.name = name
                self.head_metas.append(raf_1)

        # if not self.only_det:
        #     self._get_fg_matrix()

        print("Chosen HeadMeat: ", [type(hn) for hn in self.head_metas])
    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module Visual Genome')

        group.add_argument('--gqa-cn-data-dir',
                           default=cls.data_dir)

        group.add_argument('--gqa-cn-n-images',
                           default=cls.n_images, type=int,
                           help='number of images to sample')
        group.add_argument('--gqa-cn-square-edge',
                           default=cls.square_edge, type=int, nargs='+',
                           help='square edge of input images')
        assert not cls.extended_scale
        group.add_argument('--gqa-cn-extended-scale',
                           default=False, action='store_true',
                           help='augment with an extended scale range')
        group.add_argument('--gqa-cn-orientation-invariant',
                           default=cls.orientation_invariant, type=float,
                           help='augment with random orientations')
        group.add_argument('--gqa-cn-blur',
                           default=cls.blur, type=float,
                           help='augment with blur')
        assert cls.augmentation
        group.add_argument('--gqa-cn-no-augmentation',
                           dest='gqa_cn_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--gqa-cn-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')

        group.add_argument('--gqa-cn-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--gqa-cn-special-preprocess',
                           dest='gqa_cn_special_preprocess',
                           default=False, action='store_true',
                           help='do not apply data augmentation')
        group.add_argument('--gqa-cn-max-long-edge',
                           dest='gqa_cn_max_long_edge',
                           default=False, action='store_true',
                           help='do not apply data augmentation')
        group.add_argument('--gqa-cn-no-flipping',
                           dest='gqa_cn_no_flipping',
                           default=False, action='store_true',
                           help='do not apply data augmentation')
        group.add_argument('--gqa-cn-use-dcn',
                dest='gqa_cn_use_dcn',
                default=False, action='store_true',
                help='use deformable Conv in head')
        group.add_argument('--gqa-cn-supervise-offset',
                dest='gqa_cn_supervise_offset',
                default=False, action='store_true',
                help='Supervise offset of deformable Conv in head')
        group.add_argument('--gqa-cn-ignore-rel',
                dest='gqa_cn_ignore_rel',
                default=False, action='store_true',
                help='ignore relationship everywhere')

        group.add_argument('--gqa-cn-det',
                dest='gqa_cn_det',
                default=False, action='store_true',
                help='only detection')
        # group.add_argument('--gqa-cn-use-512',
        #         dest='gqa_cn_512',
        #         default=False, action='store_true',
        #         help='only detection')
        group.add_argument('--gqa-cn-cifdet',
                           default=False, action='store_true',
                           help='Use CifDet_CN')
        group.add_argument('--gqa-cn-refine-feature',
                           default=False, action='store_true',
                           help='Refine full feature')
        group.add_argument('--gqa-cn-laplace',
                           default=False, action='store_true',
                           help='Use heads with laplace')
        group.add_argument('--gqa-cn-detach-offset',
                           default=False, action='store_true',
                           help='Detach offset training')
        group.add_argument('--gqa-cn-detach-deform',
                           default=False, action='store_true',
                           help='Detach deform training')
        group.add_argument('--gqa-cn-no-filter-nonoverlap',
                           default=False, action='store_true',
                           help='Filter out objects with no overlap')
        group.add_argument('--gqa-cn-keep-duplicate',
                           default=False, action='store_true',
                           help='Keep many relationships between objects')
        group.add_argument('--gqa-cn-hadamard-refined',
                           default=False, action='store_true',
                           help='hadamard Product to refine (not concat)')
        group.add_argument('--gqa-cn-caf-bigger',
                dest='gqa_cn_caf_bigger',
                default=False, action='store_true',
                help='Use RAF CAF Bigger')
        group.add_argument('--gqa-cn-rgb',
                dest='gqa_cn_rgb',
                default=False, action='store_true',
                help='Convert GQA from BGR to RGB')
        group.add_argument('--gqa-cn-concat-offset',
                           default=False, action='store_true',
                           help='Concatenate Offset with refined feature')
        group.add_argument('--gqa-cn-group-deform',
                dest='gqa_cn_gdeform',
                default=False, action='store_true',
                help='Use Single Group Deform and longer head')
        group.add_argument('--gqa-cn-fourier-features',
                dest='gqa_cn_fourier_features',
                default=False, action='store_true',
                help='Use fourier features to encode (x,y)')
        group.add_argument('--gqa-cn-prior',
                dest='gqa_cn_prior',
                default=None, type=str, choices=[None, 'predcls', 'sgcls'],
                help='Include prior in input')
        group.add_argument('--gqa-cn-eval-long-edge', default=cls.eval_long_edge, type=int,
                           help='set to zero to deactivate rescaling')
        group.add_argument('--gqa-cn-giou-loss',
                dest='gqa_cn_giou_loss',
                default=False, action='store_true',
                help='Use GIOU Loss')
        group.add_argument('--gqa-cn-mask-raf',
                dest='gqa_cn_mask_raf',
                default=False, action='store_true',
                help='Use Mask Raf')
        group.add_argument('--gqa-cn-no-scale',
                dest='gqa_cn_no_scale',
                default=False, action='store_true',
                help='Remove scale from Raf Training')
        group.add_argument('--gqa-cn-kernel_1',
                dest='gqa_cn_kernel_1',
                default=False, action='store_true',
                help='Set Kernel of CenterNet head as 1')
        group.add_argument('--gqa-cn-single-head',
                dest='gqa_cn_single_head',
                default=False, action='store_true',
                help='Use only single head for CenterNet')
        group.add_argument('--gqa-cn-single-supervision',
                dest='gqa_cn_single_supervision',
                default=False, action='store_true',
                help='Use only one supervision signal')
        group.add_argument('--gqa-cn-fpn',
                dest='gqa_cn_fpn',
                default=False, action='store_true',
                help='Using FPN')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # visual genome specific
        # cls.train_annotations = args.gqa_cn_train_annotations
        # cls.val_annotations = args.gqa_cn_val_annotations
        # cls.train_image_dir = args.gqa_cn_train_image_dir
        # cls.val_image_dir = args.gqa_cn_val_image_dir
        # cls.eval_image_dir = cls.val_image_dir
        # cls.eval_annotations = cls.val_annotations
        cls.data_dir = args.gqa_cn_data_dir
        cls.n_images = args.gqa_cn_n_images
        cls.square_edge = args.gqa_cn_square_edge
        if len(cls.square_edge)==1:
            cls.square_edge = cls.square_edge[0]
        cls.extended_scale = args.gqa_cn_extended_scale
        cls.orientation_invariant = args.gqa_cn_orientation_invariant
        cls.blur = args.gqa_cn_blur
        cls.augmentation = args.gqa_cn_augmentation
        cls.rescale_images = args.gqa_cn_rescale_images
        cls.upsample_stride = args.gqa_cn_upsample
        cls.special_preprocess = args.gqa_cn_special_preprocess
        cls.max_long_edge = args.gqa_cn_max_long_edge
        cls.no_flipping = args.gqa_cn_no_flipping
        cls.caf_bigger = args.gqa_cn_caf_bigger

        cls.use_dcn = args.gqa_cn_use_dcn
        cls.supervise_offset = args.gqa_cn_supervise_offset

        cls.only_det = args.gqa_cn_det
        cls.ignore_rel = args.gqa_cn_ignore_rel
        #cls.gqa_512 = args.gqa_cn_512
        cls.cifdet_cn = args.gqa_cn_cifdet
        cls.refine_feature = args.gqa_cn_refine_feature
        cls.laplace = args.gqa_cn_laplace
        cls.detach_deform = args.gqa_cn_detach_deform
        cls.detach_offset = args.gqa_cn_detach_offset
        cls.filter_non_overlap = not args.gqa_cn_no_filter_nonoverlap
        cls.remove_duplicate = not args.gqa_cn_keep_duplicate
        cls.hadamard_product = args.gqa_cn_hadamard_refined
        cls.gqa_rgb = args.gqa_cn_rgb
        cls.group_deform = args.gqa_cn_gdeform
        cls.fourier_features = args.gqa_cn_fourier_features
        cls.prior = args.gqa_cn_prior
        cls.eval_long_edge = args.gqa_cn_eval_long_edge
        cls.giou_loss = args.gqa_cn_giou_loss
        cls.mask_raf = args.gqa_cn_mask_raf
        cls.concat_offset = args.gqa_cn_concat_offset
        cls.no_scale = args.gqa_cn_no_scale
        cls.kernel_1 = args.gqa_cn_kernel_1
        cls.single_head = args.gqa_cn_single_head
        cls.single_supervision = args.gqa_cn_single_supervision
        cls.use_fpn = args.gqa_cn_fpn

    @staticmethod
    def _convert_data(parent_data, meta):
        image, category_id = parent_data

        anns = [{
            'bbox': np.asarray([5, 5, 21, 21], dtype=np.float32),
            'category_id': category_id + 1,
        }]

        return image, anns, meta

    def _preprocess(self):
        # encoders = (openpifpaf.encoder.CifDet(self.head_metas[0]),
        #             Raf(self.head_metas[1]),)
        if not self.cifdet_cn:
            if self.only_det:
                chosen_objdetEnc = CenterNetEncoder
                if self.use_fpn:
                    chosen_objdetEnc = SharedCenterNetEncoder
                if not self.single_supervision:
                    encoders = (chosen_objdetEnc(self.head_metas[0]),chosen_objdetEnc(self.head_metas[1]),)
                else:
                    encoders = (chosen_objdetEnc(self.head_metas[0]),)
            else:
                print("Setting up CenterNetEncoder and Raf")
                chosen_objdetEnc = CenterNetEncoder
                chosen_rafEnc = Raf_CNEncoder
                if self.use_fpn:
                    chosen_objdetEnc = SharedCenterNetEncoder
                    chosen_rafEnc = SharedRaf_CN

                if self.laplace:
                    chosen_rafEnc = Raf
                # encoders = (CenterNetEncoder(self.head_metas[0]),
                #             CenterNetEncoder(self.head_metas[1]),
                #             Raf_CNEncoder(self.head_metas[2], offset=self.supervise_offset),
                #             Raf_CNEncoder(self.head_metas[3], offset=self.supervise_offset))
                if not self.single_supervision:
                    encoders = (chosen_objdetEnc(self.head_metas[0]),
                                chosen_objdetEnc(self.head_metas[1]),
                                chosen_rafEnc(self.head_metas[2]),
                                chosen_rafEnc(self.head_metas[3]))
                else:
                    encoders = (chosen_objdetEnc(self.head_metas[0]),
                                chosen_rafEnc(self.head_metas[1]),)
        else:
            chosen_rafEnc = Raf_CNEncoder
            chosen_cifdetEnc = CifDet_CNEncoder
            if self.laplace:
                chosen_rafEnc = Raf
                chosen_cifdetEnc = openpifpaf.encoder.CifDet
            if self.only_det:
                if not self.single_supervision:
                    encoders = (chosen_cifdetEnc(self.head_metas[0]),CifDet_CNEncoder(self.head_metas[1]),)
                else:
                    encoders = (chosen_cifdetEnc(self.head_metas[0]),)
            else:
                print("Setting up CifDet_CN and Raf")
                if not self.single_supervision:
                    encoders = (chosen_cifdetEnc(self.head_metas[0]),
                                chosen_cifdetEnc(self.head_metas[1]),
                                chosen_rafEnc(self.head_metas[2], offset=self.supervise_offset),
                                chosen_rafEnc(self.head_metas[3], offset=self.supervise_offset))
                else:
                    encoders = (chosen_cifdetEnc(self.head_metas[0]),
                                chosen_rafEnc(self.head_metas[1], offset=self.supervise_offset),)

        prior_encoders = None
        if self.prior:
            if not self.single_supervision:
                prior_encoders = Prior_HM(PriorEncoder(self.head_metas[1]))
            else:
                prior_encoders = Prior_HM(PriorEncoder(self.head_metas[0]))
        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations_hrnet(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        if self.extended_scale:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.25 * self.rescale_images,
                             2.0 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))
        else:
            rescale_t = openpifpaf.transforms.RescaleRelative(
                scale_range=(0.7 * self.rescale_images,
                             1.5 * self.rescale_images),
                power_law=True, stretch_range=(0.75, 1.33))

        orientation_t = None
        if self.orientation_invariant:
            orientation_t = openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.RotateBy90(), self.orientation_invariant)

        if self.special_preprocess:
            return openpifpaf.transforms.Compose([
                #openpifpaf.transforms.NormalizeAnnotations_hrnet(),
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RandomApply(Raf_HFlip(BBOX_KEYPOINTS, BBOX_HFLIP, REL_CATEGORIES, REL_CATEGORIES_FLIP), 0.5),
                #rescale_t,
                openpifpaf.transforms.RescaleRelative(scale_range=(0.8 * self.rescale_images,
                             1.3* self.rescale_images), absolute_reference=self.square_edge),
                openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
                openpifpaf.transforms.CenterPad(self.square_edge),
                orientation_t,
                #openpifpaf.transforms.MinSize(min_side=4.0),
                openpifpaf.transforms.UnclippedArea(threshold=0.8),
                # transforms.UnclippedSides(),
                openpifpaf.transforms.TRAIN_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        if self.no_flipping:
            return openpifpaf.transforms.Compose([
                #openpifpaf.transforms.NormalizeAnnotations_hrnet(),
                openpifpaf.transforms.NormalizeAnnotations(),
                #rescale_t,
                #openpifpaf.transforms.RescaleRelative(scale_range=(0.7 * self.rescale_images,
                #             1.5* self.rescale_images), absolute_reference=self.square_edge),
                openpifpaf.transforms.RescaleRelative(scale_range=(0.7 * self.rescale_images,
                             1.5* self.rescale_images)),
                openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
                openpifpaf.transforms.CenterPad(self.square_edge),
                #orientation_t,
                openpifpaf.transforms.MinSize(min_side=4.0),
                openpifpaf.transforms.UnclippedArea(threshold=0.75),
                # transforms.UnclippedSides(),
                openpifpaf.transforms.TRAIN_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        if self.max_long_edge:
            return openpifpaf.transforms.Compose([
                #openpifpaf.transforms.NormalizeAnnotations_hrnet(),
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RandomApply(Raf_HFlip(BBOX_KEYPOINTS, BBOX_HFLIP, REL_CATEGORIES, REL_CATEGORIES_FLIP), 0.5),
                #rescale_t,
                openpifpaf.transforms.RescaleRelative(scale_range=(0.7 * self.rescale_images,
                             1* self.rescale_images), absolute_reference=self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                orientation_t,
                #openpifpaf.transforms.MinSize(min_side=4.0),
                # transforms.UnclippedSides(),
                openpifpaf.transforms.TRAIN_TRANSFORM,
                openpifpaf.transforms.Encoders(encoders),
            ])

        return openpifpaf.transforms.Compose([
            #openpifpaf.transforms.NormalizeAnnotations_hrnet(),
            openpifpaf.transforms.NormalizeAnnotations(),
            #openpifpaf.transforms.AnnotationJitter(),
            #openpifpaf.transforms.RandomApply(openpifpaf.transforms.HFlip(BBOX_KEYPOINTS, BBOX_HFLIP), 0.5),
            openpifpaf.transforms.RandomApply(Raf_HFlip(BBOX_KEYPOINTS, BBOX_HFLIP, REL_CATEGORIES, REL_CATEGORIES_FLIP), 0.5),
            rescale_t,
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.Blur(), self.blur),
            #openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            # openpifpaf.transforms.RandomApply(
            #      openpifpaf.transforms.RotateUniform(30.0), 0.5),
            Crop(self.square_edge, use_area_of_interest=True),
            openpifpaf.transforms.CenterPad(self.square_edge),
            orientation_t,
            #openpifpaf.transforms.MinSize(min_side=4.0),
            openpifpaf.transforms.UnclippedArea(threshold=0.75),
            # transforms.UnclippedSides(),
            openpifpaf.transforms.TRAIN_TRANSFORM,
            prior_encoders,
            openpifpaf.transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = GQA(
            data_dir=self.data_dir,
            preprocess=self._preprocess(),
            num_im=self.n_images,
            #use_512=self.gqa_512,
            filter_non_overlap=self.filter_non_overlap,
            filter_duplicate_rels=self.remove_duplicate,
            convert_rgb=self.gqa_rgb
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug and self.augmentation,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = GQA(
            data_dir=self.data_dir,
            preprocess=self._preprocess(),
            num_im=5000,#self.n_images,
            split='test',
            #use_512=self.gqa_512,
            convert_rgb=self.gqa_rgb
        )

        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def _eval_preprocess(self):
        rescale_t = None
        if self.eval_extended_scale:
            assert self.eval_long_edge
            rescale_t = openpifpaf.transforms.DeterministicEqualChoice([
                    openpifpaf.transforms.RescaleAbsolute(self.eval_long_edge),
                    openpifpaf.transforms.RescaleAbsolute((self.eval_long_edge) // 2),
                ], salt=1)
        elif self.eval_long_edge:
            rescale_t = openpifpaf.transforms.RescaleAbsolute(self.eval_long_edge)
        padding_t = None
        if self.batch_size == 1:
            padding_t = openpifpaf.transforms.CenterPadTight(128)
            #padding_t = openpifpaf.transforms.CenterPad(512)
        else:
            assert self.eval_long_edge
            padding_t = openpifpaf.transforms.CenterPad(self.eval_long_edge)

        orientation_t = None
        if self.eval_orientation_invariant:
            orientation_t = openpifpaf.transforms.DeterministicEqualChoice([
                    None,
                    openpifpaf.transforms.RotateBy90(fixed_angle=90),
                    openpifpaf.transforms.RotateBy90(fixed_angle=180),
                    openpifpaf.transforms.RotateBy90(fixed_angle=270),
                ], salt=3)

        prior_encoders = None
        mask_encoder = None
        if self.prior:
            if not self.single_supervision:
                prior_encoders = Prior_HM(PriorEncoder(self.head_metas[1]))
            else:
                prior_encoders = Prior_HM(PriorEncoder(self.head_metas[0]))
            if self.mask_raf:
                mask_encoder = openpifpaf.transforms.Encoders([MaskHMEncoder(self.head_metas[-1])])

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            rescale_t,
            padding_t,
            orientation_t,
            # openpifpaf.transforms.ToAnnotations([
            #     ToRafAnnotations(self.obj_categories, self.rel_categories),
            #     openpifpaf.transforms.ToCrowdAnnotations(self.obj_categories),
            # ]),
            openpifpaf.transforms.EVAL_TRANSFORM,
            prior_encoders,
            mask_encoder,
            openpifpaf.transforms.ToAnnotations([
                ToRafAnnotations(self.obj_categories, self.rel_categories),
            ]),
        ])
    def _get_fg_matrix(self):
        # train_data = VisualRelationship(
        #     image_dir=self.train_image_dir,
        #     ann_file=self.train_annotations
        # )
        train_data = GQA(
            data_dir=self.data_dir,
        )


        self.head_metas[-1].fg_matrix, self.head_metas[-1].bg_matrix, self.head_metas[-1].smoothing_pred = train_data.get_frequency_prior(self.obj_categories, self.rel_categories)


    def eval_loader(self):
        eval_data = GQA(
            data_dir=self.data_dir,
            preprocess=self._eval_preprocess(),
            num_im=self.n_images,
            split='test',
            #use_512=self.gqa_512,
            eval_mode=True,
            convert_rgb=self.gqa_rgb
        )
        if not self.only_det:
            self._get_fg_matrix()

        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        # eval_data = VG(
        #     data_dir=self.data_dir,
        #     preprocess=self._eval_preprocess(),
        #     num_im=self.n_images,
        #     split='test',
        #     use_512=self.gqa_512,
        #     eval_mode=True,
        # )

        eval_data = None
        if self.only_det:
            return [metric.VG(obj_categories=self.obj_categories, rel_categories=self.rel_categories, mode='sgdet', iou_types=['bbox'], gqa_eval=eval_data)]
        return [metric.VG(obj_categories=self.obj_categories, rel_categories=self.rel_categories, mode='sgdet', iou_types=['bbox', 'relations'] if (not self.only_det) else ['bbox'], gqa_eval=eval_data),
                metric.VG(obj_categories=self.obj_categories, rel_categories=self.rel_categories, mode='sgcls', iou_types=['bbox','relations'] if (not self.only_det) else ['bbox'], gqa_eval=eval_data),
                metric.VG(obj_categories=self.obj_categories, rel_categories=self.rel_categories, mode='predcls', iou_types=['bbox','relations'] if (not self.only_det) else ['bbox'], gqa_eval=eval_data)
                ]
