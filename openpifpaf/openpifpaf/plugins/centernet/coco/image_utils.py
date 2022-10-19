from openpifpaf.transforms.preprocess import Preprocess
import random
import torch
import torchvision
import cv2
import copy
import numpy as np
import logging
from PIL import Image

from openpifpaf.transforms.hflip import _HorizontalSwap

LOG = logging.getLogger(__name__)

class CenterPadTensor(Preprocess):
    def __init__(self, target_size):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        LOG.debug('valid area before pad: %s, image size = %s', meta['valid_area'], image.size)
        image, anns, ltrb = self.center_pad(image, anns)
        meta['offset'] -= ltrb[:2]
        meta['valid_area'][:2] += ltrb[:2]
        LOG.debug('valid area after pad: %s, image size = %s', meta['valid_area'], image.size)

        return image, anns, meta

    def center_pad(self, image, anns):
        h, w = image.shape[-2:]

        left = int((self.target_size[0] - w) / 2.0)
        top = int((self.target_size[1] - h) / 2.0)
        if left < 0:
            left = 0
        if top < 0:
            top = 0

        right = self.target_size[0] - w - left
        bottom = self.target_size[1] - h - top
        if right < 0:
            right = 0
        if bottom < 0:
            bottom = 0
        ltrb = (left, top, right, bottom)
        LOG.debug('pad with %s', ltrb)

        # pad image
        image = torchvision.transforms.functional.pad(
            image, ltrb, fill=0)

        # pad annotations
        for ann in anns:
            ann['keypoints'][:, 0] += ltrb[0]
            ann['keypoints'][:, 1] += ltrb[1]
            ann['bbox'][0] += ltrb[0]
            ann['bbox'][1] += ltrb[1]

        return image, anns, ltrb

class HFlipTensor(Preprocess):
    def __init__(self, keypoints, hflip):
        self.swap = _HorizontalSwap(keypoints, hflip)

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        w = image.shape[-1]
        image = torch.from_numpy(image, )
        for ann in anns:
            ann['keypoints'][:, 0] = -ann['keypoints'][:, 0] - 1.0 + w
            if self.swap is not None and not ann['iscrowd']:
                ann['keypoints'] = self.swap(ann['keypoints'])
                meta['horizontal_swap'] = self.swap
            ann['bbox'][0] = -(ann['bbox'][0] + ann['bbox'][2]) - 1.0 + w

        assert meta['hflip'] is False
        meta['hflip'] = True

        meta['valid_area'][0] = -(meta['valid_area'][0] + meta['valid_area'][2]) + w

        return image, anns, meta

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

    # def __call__(self, image, anns, meta):
    #     meta = copy.deepcopy(meta)
    #     anns = copy.deepcopy(anns)
    #
    #     center = np.array([image.shape[2] / 2., image.shape[1] / 2.], dtype=np.float32)
    #     scale = np.random.choice(np.arange(self.scale_range[0], self.scale_range[1], 0.1))
    #     w_border = self._get_border(128, image.shape[2])
    #     h_border = self._get_border(128, image.shape[1])
    #     center[0] = np.random.randint(low=w_border, high=image.shape[2] - w_border)
    #     center[1] = np.random.randint(low=h_border, high=image.shape[1] - h_border)
    #
    #     if not isinstance(self.scale_range, np.ndarray) and not isinstance(self.scale_range, list):
    #         scale = np.array([self.scale_range, self.scale_range], dtype=np.float32)
    #     if isinstance(self.scale_range, list):
    #         scale = np.array(self.scale_range)
    #
    #     scale_tmp = scale
    #     src_w = scale_tmp[0]
    #     dst_w = self.target_wh[0]
    #     dst_h = self.target_wh[1]
    #
    #     rot_rad = np.pi * self.rot / 180
    #     src_dir = self.get_dir([0, src_w * -0.5], rot_rad)
    #     dst_dir = np.array([0, dst_w * -0.5], np.float32)
    #
    #     src = np.zeros((3, 2), dtype=np.float32)
    #     dst = np.zeros((3, 2), dtype=np.float32)
    #     src[0, :] = center + scale_tmp * self.shift
    #     src[1, :] = center + src_dir + scale_tmp * self.shift
    #     dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    #     dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir
    #
    #     src[2:, :] = self.get_3rd_point(src[0, :], src[1, :])
    #     dst[2:, :] = self.get_3rd_point(dst[0, :], dst[1, :])
    #
    #     trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    #     image_unsqueezed = image.unsqueeze(0)
    #     theta = self.cvt_MToTheta(trans, image.shape[2], image.shape[1])
    #     grid = torch.nn.functional.affine_grid(torch.from_numpy(theta).float().unsqueeze(0), torch.Size((1, 3, dst_h, dst_w)))
    #     feature_rotated = torch.nn.functional.grid_sample(image_unsqueezed, grid).squeeze()
    #
    #     for ann in anns:
    #         if 'bbox' in ann:
    #             ann['bbox'][:2] = self.affine_transform(ann['bbox'][:2], trans)
    #             ann['bbox'][2:] = self.affine_transform(ann['bbox'][2:], trans)
    #
    #     return feature_rotated, anns, meta

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        center = np.array([image.size[1] / 2., image.size[0] / 2.], dtype=np.float32)
        s = max(image.size[1], image.size[0]) * 1.0
        scale = np.random.choice(np.arange(self.scale_range[0], self.scale_range[1], 0.1))
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
        meta['scale'] *= scale
        prefactor = np.array([1, 1])
        if meta['offset'][0] == 0:
            prefactor[0] = -1
        if meta['offset'][1] == 0:
            prefactor[1] = -1
        meta['offset'] = self.affine_transform(meta['offset'], trans)*prefactor
        meta['valid_area'][:2] = np.maximum(0.0, self.affine_transform(meta['valid_area'][:2], trans))
        meta['valid_area'][2:] = np.minimum(self.affine_transform(meta['valid_area'][2:], trans), [dst_w - 1, dst_h - 1])
        return image, anns_ret, meta

class EigenTransform(Preprocess):
    def __init__(self, data_rng, eig_val, eig_vec):
        self.eig_val = eig_val
        self.eig_vec = eig_vec
        self.data_rng = data_rng

    def _grayscale(self, image):
        return torchvision.transforms.Grayscale()(image)

    def _lighting(self, data_rng, image, alphastd, eigval, eigvec):
        alpha = data_rng.normal(scale=alphastd, size=(3, ))
        image += torch.matmul(torch.from_numpy(eigvec).float(),  torch.from_numpy(eigval * alpha).float()).unsqueeze(-1).unsqueeze(-1)

    def _blend(self, alpha, image1, image2):
        image1 *= alpha
        image2 *= (1 - alpha)
        image1 += image2

    def _saturation(self, data_rng, image, gs, gs_mean, var):
        alpha = 1. + data_rng.uniform(low=-var, high=var)
        self._blend(alpha, image, gs)

    def _brightness(self, data_rng, image, gs, gs_mean, var):
        alpha = 1. + data_rng.uniform(low=-var, high=var)
        image *= alpha

    def _contrast(self, data_rng, image, gs, gs_mean, var):
        alpha = 1. + data_rng.uniform(low=-var, high=var)
        self._blend(alpha, image, gs_mean)

    def _color_aug(self, data_rng, image, eig_val, eig_vec):
        functions = [self._brightness, self._contrast,self._saturation]
        random.shuffle(functions)

        gs = self._grayscale(image)
        gs_mean = gs.mean()
        for f in functions:
            f(data_rng, image, gs, gs_mean, 0.4)
        self._lighting(data_rng, image, 0.1, eig_val, eig_vec)

    def __call__(self, image, anns, meta):
        self._color_aug(self.data_rng, image, self.eig_val, self.eig_vec)
        return image, anns, meta
