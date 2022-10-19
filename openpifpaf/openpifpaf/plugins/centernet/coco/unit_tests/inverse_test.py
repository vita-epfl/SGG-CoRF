from pycocotools.coco import COCO
import os

from openpifpaf.plugins.coco.constants import COCO_CATEGORIES
import openpifpaf
from PIL import Image

from openpifpaf.transforms.preprocess import Preprocess
import random
import torch
import torchvision
import cv2
import copy
import numpy as np
import logging
from PIL import Image

class AffineTransform(Preprocess):
    def __init__(self, target_wh, scale_range=(0.5, 1.0), *,
                 rot=0,
                 shift=np.array([0, 0], dtype=np.float32),
                 fast=False):
        self.scale_range = scale_range
        self.fast = fast
        self.rot = rot
        self.shift = shift
        self.target_wh = target_wh
        if not isinstance(self.target_wh, np.ndarray) and not isinstance(self.target_wh, list):
            self.target_wh = np.array([self.target_wh, self.target_wh], dtype=np.int)

    def get_dir(self, src_point, rot_rad):
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)

        src_result = [0, 0]
        src_result[0] = src_point[0] * cs - src_point[1] * sn
        src_result[1] = src_point[0] * sn + src_point[1] * cs

        return src_result

    def get_3rd_point(self, a, b):
        direct = a - b
        return b + np.array([-direct[1], direct[0]], dtype=np.float32)

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def get_N(self, W, H):
        """N that maps from unnormalized to normalized coordinates"""
        N = np.zeros((3, 3), dtype=np.float64)
        N[0, 0] = 2.0 / W
        N[0, 1] = 0
        N[1, 1] = 2.0 / H
        N[1, 0] = 0
        N[0, -1] = -1.0
        N[1, -1] = -1.0
        N[-1, -1] = 1.0
        return N


    def get_N_inv(self, W, H):
        """N that maps from normalized to unnormalized coordinates"""
        # TODO: do this analytically maybe?
        N = self.get_N(W, H)
        return np.linalg.inv(N)

    def cvt_MToTheta(self, M, w, h):
        """convert affine warp matrix `M` compatible with `opencv.warpAffine` to `theta` matrix
        compatible with `torch.F.affine_grid`

        Parameters
        ----------
        M : np.ndarray
            affine warp matrix shaped [2, 3]
        w : int
            width of image
        h : int
            height of image

        Returns
        -------
        np.ndarray
            theta tensor for `torch.F.affine_grid`, shaped [2, 3]
        """
        M_aug = np.concatenate([M, np.zeros((1, 3))], axis=0)
        M_aug[-1, -1] = 1.0
        N = self.get_N(w, h)
        N_inv = self.get_N_inv(w, h)
        theta = N @ M_aug @ N_inv
        theta = np.linalg.inv(theta)
        return theta[:2, :]

    def affine_transform(self, pt, t):
        new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
        new_pt = np.dot(t, new_pt)
        return new_pt[:2]

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        center = np.array([image.size[1] / 2., image.size[0] / 2.], dtype=np.float32)
        s = max(image.size[1], image.size[0]) * 1.0
        scale = 1#np.random.choice(np.arange(self.scale_range[0], self.scale_range[1], 0.1))
        w_border = self._get_border(128, image.size[1])
        h_border = self._get_border(128, image.size[0])
        center[0] = np.random.randint(low=w_border, high=image.size[1] - w_border)
        center[1] = np.random.randint(low=h_border, high=image.size[0] - h_border)
        if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
            scale = np.array([scale, scale], dtype=np.float32)
        if isinstance(scale, list):
            scale = np.array(scale)
        scale_tmp = s*scale
        src_w = scale_tmp[0]
        #self.target_wh = [image.size[1], image.size[0]]
        dst_w = self.target_wh[0]
        dst_h = self.target_wh[1]

        rot_rad = np.pi * self.rot / 180
        src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
        dst_dir = np.array([0, dst_w * -0.5], np.float32)

        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center + scale_tmp * self.shift
        src[1, :] = center + src_dir + scale_tmp * self.shift
        dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

        src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
        dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])

        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
        trans_inverted = cv2.invertAffineTransform(trans)
        image = image.transform(
                (dst_w, dst_h),
                Image.AFFINE,
                trans_inverted.flatten(),
                resample=Image.BILINEAR
            )
        anns_ret = []
        for ann in anns:
            if 'bbox' in ann:
                bbox = self._coco_box_to_bbox(ann['bbox'])
                bbox[:2] = self.affine_transform(bbox[:2], trans)
                bbox[2:] = self.affine_transform(bbox[2:], trans)
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, dst_w - 1)
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, dst_h - 1)
                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    ann['bbox'] = np.asarray([bbox[0], bbox[1], w, h], dtype=np.float32)
                    if not 'keypoints' in ann:
                        ann['keypoints'] = np.asarray([], dtype=np.float32).reshape(-1, 3)
                    anns_ret.append(ann)

        meta['rotation']['angle'] += self.rot
        meta['scale'] *= scale #[self.target_wh]
        prefactor = np.array([1, 1])
        if meta['offset'][0] == 0:
            prefactor[0] = -1
        if meta['offset'][1] == 0:
            prefactor[1] = -1
        meta['offset'] = self.affine_transform(meta['offset'], trans)*prefactor
        meta['valid_area'][:2] = np.maximum(0.0, self.affine_transform(meta['valid_area'][:2], trans))
        meta['valid_area'][2:] = np.minimum(self.affine_transform(meta['valid_area'][2:], trans), [dst_w - 1, dst_h - 1])
        return image, anns_ret, meta

ann_file = "data-coco/annotations/instances_train2017.json"
image_dir = 'data-coco/images/train2017/'

coco = COCO(ann_file)
index = 0
ids = coco.getImgIds()
image_id = ids[index]
ann_ids = coco.getAnnIds(imgIds=image_id, catIds=[])
anns = coco.loadAnns(ann_ids)
image_info = coco.loadImgs(image_id)[0]
local_file_path = os.path.join(image_dir, image_info['file_name'])

meta = {
    'dataset_index': index,
    'image_id': image_id,
    'file_name': image_info['file_name'],
    'local_file_path': local_file_path,
}


with open(local_file_path, 'rb') as f:
    image = Image.open(f).convert('RGB')

preprocess = openpifpaf.transforms.Compose([
    openpifpaf.transforms.NormalizeAnnotations(),
    AffineTransform(target_wh=512, scale_range=[0.6, 1.4]),
    openpifpaf.transforms.ToAnnotations([
        openpifpaf.transforms.ToDetAnnotations(COCO_CATEGORIES),
        ]),
    ])

image, anns_modified, meta = preprocess(image, anns, meta)

import pdb; pdb.set_trace()
pred = [ann.inverse_transform(meta) for ann in anns_modified]
