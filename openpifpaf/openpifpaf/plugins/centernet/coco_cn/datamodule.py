import argparse

import torch
import torchvision

import openpifpaf
from ..headmeta import CenterNet, CifDet_CN
from .. import encoder
from ..cifdet_cn import CifDet_CN as CifDet_CNEncoder

from ..coco.constants import (
    COCO_CATEGORIES,
)

from .cocodet_cn import COCO as CocoDataset

try:
    import pycocotools.coco
    # monkey patch for Python 3 compat
    pycocotools.coco.unicode = str
except ImportError:
    pass

class CocoDet(openpifpaf.datasets.DataModule):
    # cli configurable
    train_annotations = 'data-mscoco/annotations/instances_train2017.json'
    val_annotations = 'data-mscoco/annotations/instances_val2017.json'
    eval_annotations = val_annotations
    train_image_dir = 'data-mscoco/images/train2017/'
    val_image_dir = 'data-mscoco/images/val2017/'
    eval_image_dir = val_image_dir

    upsample_stride = 1
    cifdet_cn = False
    opt = None

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
        group = parser.add_argument_group('data module CocoDet CN Original')

        group.add_argument('--cn-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--cn-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--cn-train-image-dir',
                           default=cls.train_image_dir)
        group.add_argument('--cn-val-image-dir',
                           default=cls.val_image_dir)

        group.add_argument('--keep_res', action='store_true',
                             help='keep the original resolution'
                                  ' during validation.')
        group.add_argument('--not_rand_crop', action='store_true',
                             help='not use the random crop data augmentation'
                                  'from CornerNet.')
        group.add_argument('--shift', type=float, default=0.1,
                             help='when not using random crop'
                                  'apply shift augmentation.')
        group.add_argument('--scale', type=float, default=0.4,
                             help='when not using random crop'
                                  'apply scale augmentation.')
        group.add_argument('--flip', type = float, default=0.5,
                             help='probability of applying flip augmentation.')
        group.add_argument('--no_color_aug', action='store_true',
                             help='not use the color augmenation '
                                  'from CornerNet')
        group.add_argument('--input_h', type=int, default=512,
                             help='input height. -1 for default from dataset.')
        group.add_argument('--input_w', type=int, default=512,
                             help='input width. -1 for default from dataset.')
        group.add_argument('--pad', type=int, default=127,
                             help='specify padding')
        group.add_argument('--cn-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--cn-cifdet',
                           default=False, action='store_true',
                           help='Use CifDet_CN')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # cocodet_cn specific
        cls.train_annotations = args.cn_train_annotations
        cls.val_annotations = args.cn_val_annotations
        cls.train_image_dir = args.cn_train_image_dir
        cls.val_image_dir = args.cn_val_image_dir

        cls.upsample_stride = args.cn_upsample

        cls.cifdet_cn = args.cn_cifdet
        cls.opt = args

    def _preprocess(self):
        if not self.cifdet_cn:
            enc = (encoder.CenterNetEncoder(self.head_metas[0]),
                    encoder.CenterNetEncoder(self.head_metas[1]))
        else:
            enc = (CifDet_CNEncoder(self.head_metas[0]),
                    CifDet_CNEncoder(self.head_metas[1]))

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.Encoders(enc),
        ])

    @staticmethod
    def _eval_preprocess():

        return openpifpaf.transforms.Compose([
            openpifpaf.transforms.ToAnnotations([
                    openpifpaf.transforms.ToDetAnnotations(COCO_CATEGORIES),
                    openpifpaf.transforms.ToCrowdAnnotations(COCO_CATEGORIES),
                ]),
            #openpifpaf.transforms.ImageTransform(torchvision.transforms.ToTensor())
        ])

    def train_loader(self):
        train_data = CocoDataset(
            image_dir=self.train_image_dir,
            ann_file=self.train_annotations,
            opt=self.opt,
            preprocess=self._preprocess(),
            split='train',
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def val_loader(self):
        val_data = CocoDataset(
            image_dir=self.val_image_dir,
            ann_file=self.val_annotations,
            opt=self.opt,
            preprocess=self._preprocess(),
            split='test',
        )

        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=openpifpaf.datasets.collate_images_targets_meta)

    def eval_loader(self):
        eval_data = CocoDataset(
            image_dir=self.eval_image_dir,
            ann_file=self.eval_annotations,
            opt=self.opt,
            preprocess=self._eval_preprocess(),
            split='test',
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
