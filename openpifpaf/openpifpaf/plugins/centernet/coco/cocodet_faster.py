import argparse
import numpy as np

import torch
import torchvision

import openpifpaf
from ..headmeta import CenterNet, CifDet_CN
from .. import encoder
from ..cifdet_cn import CifDet_CN as CifDet_CNEncoder

from ...coco.cocokp import CocoKp
from .constants import (
    COCO_CATEGORIES,
    COCO_KEYPOINTS,
    HFLIP,
)
from .dataset import CocoDataset
from .image_utils import AffineTransform, EigenTransform, HFlipTensor, CenterPadTensor

try:
    import pycocotools.coco
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass

def collate_images_targets_meta(batch):
    targets = []
    images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    for i in range(len(batch[0][1])):
        targets.append([])
        for j, b in enumerate(batch):
            targets[i].append(b[1][i])
    #targets = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    metas = [b[2] for b in batch]
    return images, targets, metas


class CocoDet(openpifpaf.datasets.DataModule):
    # cli configurable
    train_annotations = 'data-mscoco/annotations/instances_train2017.json'
    val_annotations = 'data-mscoco/annotations/instances_val2017.json'
    eval_annotations = val_annotations
    train_image_dir = 'data-mscoco/images/train2017/'
    val_image_dir = 'data-mscoco/images/val2017/'
    eval_image_dir = val_image_dir

    square_edge = 512
    upsample_stride = 1
    cifdet_cn = False
    eval_annotation_filter = False

    _eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    _eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)

    def __init__(self):
        super().__init__()
        if not self.cifdet_cn:
            cifdet_1 = CenterNet('cifdet_1', 'cocodet_cn', COCO_CATEGORIES)
            cifdet_2 = CenterNet('cifdet_2', 'cocodet_cn', COCO_CATEGORIES)
        else:
            cifdet_1 = CifDet_CN('cifdetcn_1', 'cocodet_cn', COCO_CATEGORIES)
            cifdet_2 = CifDet_CN('cifdetcn_2', 'cocodet_cn', COCO_CATEGORIES)
        #cifdet = headmeta.CifDet_deep('cifdet', 'cocodet', COCO_CATEGORIES)
        cifdet_1.upsample_stride = self.upsample_stride
        cifdet_2.upsample_stride = self.upsample_stride
        self.head_metas = [cifdet_1, cifdet_2]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module CocoDet')

        group.add_argument('--cn-fstr-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--cn-fstr-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--cn-fstr-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--cn-fstr-val-image-dir',
                           default=cls.val_image_dir)
        group.add_argument('--cn-fstr-eval-annotation-filter',
                           dest='cn_fstr_eval_annotation_filter',
                           default=cls.eval_annotation_filter, action='store_true')
        group.add_argument('--cn-fstr-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')

        group.add_argument('--cn-fstr-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--cn-fstr-cifdet',
                           default=False, action='store_true',
                           help='Use CifDet_CN')


    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # cocodet_cn specific
        cls.train_annotations = args.cn_fstr_train_annotations
        cls.val_annotations = args.cn_fstr_val_annotations
        cls.train_image_dir = args.cn_fstr_train_image_dir
        cls.val_image_dir = args.cn_fstr_val_image_dir

        cls.square_edge = args.cn_fstr_square_edge
        cls.upsample_stride = args.cn_fstr_upsample

        cls.eval_annotation_filter = args.cn_fstr_eval_annotation_filter
        cls.cifdet_cn = args.cn_fstr_cifdet

    def _preprocess(self):
        if not self.cifdet_cn:
            enc = (encoder.CenterNetEncoder(self.head_metas[0]),
                    encoder.CenterNetEncoder(self.head_metas[1]))
        else:
            enc = (CifDet_CNEncoder(self.head_metas[0]),
                    CifDet_CNEncoder(self.head_metas[1]))

        rescale_t = AffineTransform(target_wh=self.square_edge, scale_range=[0.6, 1.4])

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.Crop(self.square_edge, use_area_of_interest=True),
            #rescale_t,
            openpifpaf.transforms.RandomApply(
                openpifpaf.transforms.HFlip(COCO_KEYPOINTS, HFLIP), 0.5),
            openpifpaf.transforms.CenterPad(self.square_edge),
            openpifpaf.transforms.ImageTransform(torchvision.transforms.ToTensor()),
            #EigenTransform(np.random.RandomState(123), self._eig_val, self._eig_vec),
            openpifpaf.transforms.ImageTransform(
                torchvision.transforms.Normalize(mean=[0.40789654, 0.44719302, 0.47026115],
                                                 std=[0.28863828, 0.27408164, 0.27809835]),
            ),
            openpifpaf.transforms.Encoders(enc),
        ])

    def train_loader(self):
        train_data = CocoDataset(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            preprocess=self._preprocess(),
            annotation_filter=self.eval_annotation_filter,
            category_ids=[],
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = CocoDataset(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            preprocess=self._preprocess(),
            annotation_filter=self.eval_annotation_filter,
            category_ids=[],
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    @staticmethod
    def _eval_preprocess():

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.NormalizeAnnotations(),
            openpifpaf.transforms.CenterPadTight(128),
            openpifpaf.transforms.ToAnnotations([
                    openpifpaf.transforms.ToDetAnnotations(COCO_CATEGORIES),
                    openpifpaf.transforms.ToCrowdAnnotations(COCO_CATEGORIES),
                ]),
            openpifpaf.transforms.ImageTransform(torchvision.transforms.ToTensor()),
            openpifpaf.transforms.ImageTransform(
                torchvision.transforms.Normalize(mean=[0.40789654, 0.44719302, 0.47026115],
                                                 std=[0.28863828, 0.27408164, 0.27809835]),
            ),
        ])

    def eval_loader(self):
        eval_data = CocoDataset(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            preprocess=self._eval_preprocess(),
            annotation_filter=self.eval_annotation_filter,
            category_ids=[],
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        return [openpifpaf.metric.Coco(
            pycocotools.coco.COCO(self.eval_annotations),
            max_per_image=100,
            category_ids=[],
            iou_type='bbox',
        )]
