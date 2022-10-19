import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import logging

import torch.utils.data as data

import torch
import cv2
import os
from .utils import flip, color_aug
from .utils import get_affine_transform, affine_transform
from .utils import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from .utils import draw_dense_reg
import math

LOG = logging.getLogger(__name__)
STAT_LOG = logging.getLogger(__name__.replace('openpifpaf.', 'openpifpaf.stats.'))


class COCO(data.Dataset):
    num_classes = 80
    default_resolution = [512, 512]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, image_dir, ann_file, opt, preprocess, split):
        super(COCO, self).__init__()
        #self.data_dir = os.path.join(opt.data_dir, 'coco')
        self.img_dir = image_dir#os.path.join(self.data_dir, '{}2017'.format(split))
        # if split == 'test':
        #   self.annot_path = os.path.join(
        #       self.data_dir, 'annotations',
        #       'image_info_test-dev2017.json').format(split)
        # else:
        #   if opt.task == 'exdet':
        #     self.annot_path = os.path.join(
        #       self.data_dir, 'annotations',
        #       'instances_extreme_{}2017.json').format(split)
        #   else:
        #     self.annot_path = os.path.join(
        #       self.data_dir, 'annotations',
        #       'instances_{}2017.json').format(split)
        self.annot_path = ann_file
        self.preprocess = preprocess
        self.max_objs = 128
        self.class_name = [
          '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
          'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
          'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
          'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
          'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
          'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
          'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
          'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
          'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
          'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
          'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
          'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
          'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self._valid_ids = [
          1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
          14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
          24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
          37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
          48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
          58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
          72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
          82, 84, 85, 86, 87, 88, 89, 90]
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                          for v in range(1, self.num_classes + 1)]
        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
        # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

        self.split = split
        self.opt = opt

        #print('==> initializing coco 2017 {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        #print('Loaded {} {} samples'.format(split, self.num_samples))
        LOG.info('Images: %d', self.num_samples)

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def __len__(self):
        return self.num_samples

    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def __getitem__(self, index):
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        num_objs = min(len(anns), self.max_objs)

        img = cv2.imread(img_path)

        height, width = img.shape[0], img.shape[1]
        if self.split == 'test':
            scale = 1
            new_height = int(height * scale)
            new_width  = int(width * scale)
            if not self.opt.keep_res:
                inp_height, inp_width = self.opt.input_h, self.opt.input_w
                c = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
                s = max(height, width) * 1.0
            else:
                inp_height = (new_height | self.opt.pad) + 1
                inp_width = (new_width | self.opt.pad) + 1
                c = np.array([new_width // 2, new_height // 2], dtype=np.float32)
                s = np.array([inp_width, inp_height], dtype=np.float32)

            trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
            resized_image = cv2.resize(img, (new_width, new_height))
            inp_image = cv2.warpAffine(
                resized_image, trans_input, (inp_width, inp_height),
                flags=cv2.INTER_LINEAR)
            inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

            images = inp_image.transpose(2, 0, 1).reshape(3, inp_height, inp_width)
            # if self.opt.flip_test:
            #     images = np.concatenate((images, images[:, :, :, ::-1]), axis=0)
            #images = torch.from_numpy(images)
            if not isinstance(s, np.ndarray) and not isinstance(s, list):
                scale_factors = np.array([s, s], dtype=np.float32)
            else:
                scale_factors = s

            meta = {
                'dataset_index': index,
                'image_id': img_id,
                'file_name': file_name,
                'local_file_path': img_path,
                'offset': np.array((0.0, 0.0)),
                'scale': np.array((1.0, 1.0)),
                'rotation': {'angle': 0.0, 'width': None, 'height': None},
                'valid_area': np.array((0.0, 0.0,  new_width - 1, new_height - 1)),
                'hflip': False,
                'width_height': np.array((width, height)),
            }

            # meta = {'c': c, 's': s,
            #         'out_height': inp_height // self.opt.down_ratio,
            #         'out_width': inp_width // self.opt.down_ratio}

            images, anns, meta = self.preprocess(images, anns, meta)

            meta['scale'] *= scale
            meta['scale'] *= scale
            meta['offset'] = - affine_transform(meta['offset'], trans_input)
            meta['valid_area'][:2] = np.maximum(0.0, affine_transform(meta['valid_area'][:2], trans_input))
            meta['valid_area'][2:] = np.minimum(affine_transform(meta['valid_area'][2:], trans_input), [inp_width - 1, inp_height - 1])

            return images, anns, meta

        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
                s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] =  width - c[0] - 1


        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        # output_h = input_h // self.opt.down_ratio
        # output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [input_w, input_h])

        # hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        # wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        # dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        # reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        # ind = np.zeros((self.max_objs), dtype=np.int64)
        # reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        # cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        # cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)
        #
        # draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
        #                 draw_umich_gaussian

        meta = {
            'dataset_index': index,
            'image_id': img_id,
            'file_name': file_name,
            'local_file_path': img_path,
            'offset': np.array((0.0, 0.0)),
            'scale': np.array((1.0, 1.0)),
            'rotation': {'angle': 0.0, 'width': None, 'height': None},
            'valid_area': np.array((0.0, 0.0,  width - 1, height - 1)),
            'hflip': False,
            'width_height': np.array((width, height)),
        }

        if flipped:
            meta['hflip'] = True
            meta['valid_area'][0] = -(meta['valid_area'][0] + meta['valid_area'][2]) + width
        if not isinstance(s, np.ndarray) and not isinstance(s, list):
            scale_factors = np.array([s, s], dtype=np.float32)
        else:
            scale_factors = s
        meta['offset'] *= scale_factors
        meta['scale'] *= scale_factors
        meta['valid_area'][:2] *= scale_factors
        meta['valid_area'][2:] *= scale_factors

        new_wh = [input_w, input_h]
        ltrb = [c[0]-input_w/2.0, c[1]-input_h/2.0, c[0]+input_w/2.0, c[1]+input_h/2.0]
        original_valid_area = meta['valid_area'].copy()
        meta['offset'] += ltrb[:2]
        meta['valid_area'][:2] = np.maximum(0.0, original_valid_area[:2] - ltrb[:2])
        new_rb_corner = original_valid_area[:2] + original_valid_area[2:] - ltrb[:2]
        new_rb_corner = np.maximum(meta['valid_area'][:2], new_rb_corner)
        new_rb_corner = np.minimum(new_wh, new_rb_corner)
        meta['valid_area'][2:] = new_rb_corner - meta['valid_area'][:2]

        anns_ret = []
        for ann in anns:
            if 'bbox_original' not in ann:
                ann['bbox_original'] = np.copy(ann['bbox'])
            if 'iscrowd' not in ann:
                ann['iscrowd'] = False
            bbox = self._coco_box_to_bbox(ann['bbox'])
            ann['category_id'] = int(self.cat_ids[ann['category_id']]) + 1
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1

            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, input_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, input_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                ann['bbox'] = np.asarray([bbox[0], bbox[1], w, h], dtype=np.float32)
                ann['keypoints'] = np.asarray([], dtype=np.float32).reshape(-1, 3)
                anns_ret.append(ann)

        inp, anns_ret, meta = self.preprocess(inp, anns_ret, meta)

        LOG.debug(meta)
        # ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        # if self.opt.dense_wh:
        #   hm_a = hm.max(axis=0, keepdims=True)
        #   dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
        #   ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
        #   del ret['wh']
        # elif self.opt.cat_spec_wh:
        #   ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
        #   del ret['wh']
        # if self.opt.reg_offset:
        #   ret.update({'reg': reg})
        # if self.opt.debug > 0 or not self.split == 'train':
        #   gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
        #            np.zeros((1, 6), dtype=np.float32)
        #   meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
        #   ret['meta'] = meta

        return inp, anns_ret, meta
