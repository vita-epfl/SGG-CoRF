import dataclasses
import logging
import numpy as np
import torch
from typing import ClassVar
import math

from math import sqrt

from openpifpaf.encoder.annrescaler import AnnRescalerDet
from .visualizer import CenterNet as CenterNetVisualizer
from .visualizer import Raf as RafCNVisualizer
from ..raf.annrescaler import AnnRescalerRel
#from openpifpaf.utils import  mask_valid_area
from . import headmeta
from .utils import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian

LOG = logging.getLogger(__name__)

@dataclasses.dataclass
class CenterNetEncoder:
    meta: headmeta.CenterNet
    rescaler: AnnRescalerDet = None
    v_threshold: int = 0
    bmin: float = 10.0  #: in pixels
    visualizer: CenterNetVisualizer = None

    side_length: ClassVar[int] = 5
    padding: ClassVar[int] = 10

    def __call__(self, image, anns, meta, fpn_interval=None):
        return CenterNetGenerator(self)(image, anns, meta, fpn_interval)

class CenterNetGenerator():
    def __init__(self, config: CenterNetEncoder):
        self.config = config

        self.rescaler = config.rescaler or AnnRescalerDet(
            config.meta.stride, len(config.meta.categories))

        self.visualizer = config.visualizer or CenterNetVisualizer(config.meta)

        self.hm = None
        self.reg = None
        self.wh = None
        self.reg_mask = None
        self.ind = None
        self.mse_loss = False
        self.max_objs = config.meta.max_objs

        self.draw_gaussian = draw_msra_gaussian if self.mse_loss else \
                    draw_umich_gaussian
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def __call__(self, image, anns, meta, fpn_interval=None):
        if isinstance(image, tuple):
            image, _ = image
        self.anns = anns
        width_height_original = image.shape[2:0:-1]

        detections = self.rescaler.detections(anns, fpn_interval)

        bg_mask = self.rescaler.bg_mask(anns, width_height_original,
                                        crowd_margin=(self.config.side_length - 1) / 2)
        self.num_objs = min(len(detections), self.max_objs)

        valid_area = self.rescaler.valid_area(meta)

        n_fields = len(self.config.meta.categories)

        self.init_fields(n_fields, bg_mask)
        self.fill(detections)

        fields = self.fields(valid_area)

        self.visualizer.processed_image(image[[2,1,0], :, :])
        if self.config.meta.prior is None:
            self.visualizer.targets(fields, annotation_dicts=anns, metas=meta)
        else:
            self.visualizer.targets(fields[1], annotation_dicts=anns, metas=meta)


        return fields

    def init_fields(self, n_fields, bg_mask):
        field_w = bg_mask.shape[-1]
        field_h = bg_mask.shape[-2]
        self.hm = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        self.wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        self.ind = np.zeros((self.max_objs), dtype=np.int64)
        self.reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        self.cls = np.zeros((self.max_objs), dtype=np.int64)

    def fill(self, detections):
        output_w = self.hm.shape[-1]
        output_h = self.hm.shape[-2]
        for k in range(self.num_objs):
            category_id, bbox = detections[k]
            bbox = self._coco_box_to_bbox(bbox)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            if (bbox[3] - bbox[1]) > 0 and (bbox[2] - bbox[0]) > 0:
                self.fill_detection(category_id - 1, k, bbox)

    def fill_detection(self, f, k, bbox):
        output_w = self.hm.shape[-1]
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        #radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
          [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        self.draw_gaussian(self.hm[f], ct_int, radius)
        self.wh[k] = 1. * w, 1. * h
        self.ind[k] = ct_int[1] * output_w + ct_int[0]
        self.reg[k] = ct - ct_int
        self.reg_mask[k] = 1
        self.cls[k] = f+1
        # gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
        #                ct[0] + w / 2, ct[1] + h / 2, 1, f])

    def fields(self, valid_area):
        if not self.config.meta.prior is None:
            return (self.anns, (
                torch.from_numpy(self.hm),
                torch.from_numpy(self.reg),
                torch.from_numpy(self.wh),
                torch.from_numpy(self.reg_mask),
                torch.from_numpy(self.ind),
                torch.from_numpy(self.cls),
            ))
        return (
            torch.from_numpy(self.hm),
            torch.from_numpy(self.reg),
            torch.from_numpy(self.wh),
            torch.from_numpy(self.reg_mask),
            torch.from_numpy(self.ind),
            torch.from_numpy(self.cls),
        )


@dataclasses.dataclass
class RafEncoder:
    meta: headmeta.Raf_CNs
    rescaler: AnnRescalerRel = None
    v_threshold: int = 0
    visualizer: RafCNVisualizer = None

    side_length: ClassVar[int] = 5
    padding: ClassVar[int] = 10

    def __call__(self, image, anns, meta):
        return RafGenerator(self)(image, anns, meta)

class RafGenerator():
    def __init__(self, config: RafEncoder):
        self.config = config

        self.rescaler = config.rescaler or AnnRescalerRel(
            config.meta.stride, len(config.meta.rel_categories))

        self.visualizer = config.visualizer or RafCNVisualizer(config.meta)

        self.hm = None
        self.reg = None
        self.wh = None
        self.reg_mask = None
        self.ind = None
        self.mse_loss = False
        self.max_objs = config.meta.max_objs

        self.draw_gaussian = draw_msra_gaussian if self.mse_loss else \
                    draw_umich_gaussian
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def __call__(self, image, anns, meta):
        if isinstance(image, tuple):
            image, _ = image
        width_height_original = image.shape[2:0:-1]

        detections = self.rescaler.relations(anns)

        bg_mask, bg_mask_offset = self.rescaler.bg_mask(anns, width_height_original, self.config.meta,
                                        crowd_margin=None)

        self.num_objs = min(len(detections), self.max_objs)

        #valid_area = self.rescaler.valid_area(meta)

        n_fields = len(self.config.meta.rel_categories)

        self.init_fields(n_fields, bg_mask)
        self.fill(detections)

        fields = self.fields()

        self.visualizer.processed_image(image)
        self.visualizer.targets(fields, annotation_dicts=anns, metas=meta)

        return fields

    def init_fields(self, n_fields, bg_mask):
        field_w = bg_mask.shape[-1]
        field_h = bg_mask.shape[-2]
        self.hm = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.subj = np.zeros((self.max_objs, 2), dtype=np.float32)
        self.obj = np.zeros((self.max_objs, 2), dtype=np.float32)
        self.scales = np.zeros((self.max_objs, 2), dtype=np.float32)
        self.ind = np.zeros((self.max_objs), dtype=np.int64)
        self.reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

    def fill(self, detections):
        output_w = self.hm.shape[-1]
        output_h = self.hm.shape[-2]
        rel_idx = 0
        for det_index, (k, ann) in enumerate(detections.items()):
            #k, ann = detections[det_index]
            if (not len(ann['object_index']) > 0) or ann['iscrowd']:
                continue
            bbox = ann['bbox']
            det_id = ann['detection_id']
            category_id = ann['category_id']
            bbox = self._coco_box_to_bbox(bbox)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            if (bbox[3] - bbox[1]) > 0 and (bbox[2] - bbox[0]) > 0:
                for object_id, predicate in zip(ann['object_index'], ann['predicate']):
                    if detections[object_id]['iscrowd']:
                        continue
                    bbox_object = detections[object_id]['bbox']
                    object_category = detections[object_id]['category_id']
                    bbox_object = self._coco_box_to_bbox(bbox_object)
                    bbox_object[[0, 2]] = np.clip(bbox_object[[0, 2]], 0, output_w - 1)
                    bbox_object[[1, 3]] = np.clip(bbox_object[[1, 3]], 0, output_h - 1)
                    if (bbox_object[3] - bbox_object[1]) > 0 and (bbox_object[2] - bbox_object[0]) > 0:
                        self.fill_detection(predicate, rel_idx, bbox, bbox_object)
                        rel_idx += 1

    def fill_detection(self, f, k, bbox_subj, bbox_obj):
        output_w = self.hm.shape[-1]
        w_subj, h_subj = (bbox_subj[2] - bbox_subj[0]), (bbox_subj[3] - bbox_subj[1])
        w_obj, h_obj = (bbox_obj[2] - bbox_obj[0]), (bbox_obj[3] - bbox_obj[1])
        scale_subj = 0.1 * np.minimum(w_subj, h_subj)
        scale_obj = 0.1 * np.minimum(w_obj, h_obj)
        ct_subj = np.array(
            [(bbox_subj[0] + bbox_subj[2]) / 2, (bbox_subj[1] + bbox_subj[3]) / 2], dtype=np.float32)
        ct_obj = np.array(
            [(bbox_obj[0] + bbox_obj[2]) / 2, (bbox_obj[1] + bbox_obj[3]) / 2], dtype=np.float32)
        #h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        #radius = gaussian_radius((math.ceil(h), math.ceil(w)))
        #radius = max(0, int(radius))
        radius = 10
        #radius = self.opt.hm_gauss if self.opt.mse_loss else radius
        ct = np.array(
          [(ct_subj[0] + ct_obj[0]) / 2, (ct_subj[1] + ct_obj[1]) / 2], dtype=np.float32)
        ct_int = ct.astype(np.int32)
        self.draw_gaussian(self.hm[f], ct_int, radius)
        x_subj, y_subj =  ct_subj[0] - ct[0], ct_subj[1] - ct[1]
        x_obj, y_obj = ct_obj[0] - ct[0], ct_obj[1] - ct[1]
        self.subj[k] = 1. * x_subj, 1. * y_subj
        self.ind[k] = ct_int[1] * output_w + ct_int[0]
        self.obj[k] = 1. * x_obj, 1. * y_obj
        self.scales[k] = 1. * scale_subj, 1. * scale_obj
        self.reg_mask[k] = 1
        # gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
        #                ct[0] + w / 2, ct[1] + h / 2, 1, f])

    def fields(self):
        return (
            torch.from_numpy(self.hm),
            torch.from_numpy(self.subj),
            torch.from_numpy(self.obj),
            torch.from_numpy(self.scales),
            torch.from_numpy(self.reg_mask),
            torch.from_numpy(self.ind),
        )
