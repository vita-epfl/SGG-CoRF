import logging
import time

import numpy as np
from collections import defaultdict

from .occupancy import Occupancy

LOG = logging.getLogger(__name__)


class Keypoints:
    suppression = 0.0
    instance_threshold = 0.15
    keypoint_threshold = 0.15
    occupancy_visualizer = None

    def annotations(self, anns):
        start = time.perf_counter()

        for ann in anns:
            ann.data[ann.data[:, 2] < self.keypoint_threshold] = 0.0
        anns = [ann for ann in anns if ann.score >= self.instance_threshold]

        if not anns:
            return anns

        # +1 for rounding up
        max_y = int(max(np.max(ann.data[:, 1]) for ann in anns) + 1)
        max_x = int(max(np.max(ann.data[:, 0]) for ann in anns) + 1)
        # +1 because non-inclusive boundary
        shape = (len(anns[0].data), max(1, max_y + 1), max(1, max_x + 1))
        occupied = Occupancy(shape, 2, min_scale=4)

        anns = sorted(anns, key=lambda a: -a.score)
        for ann in anns:
            assert ann.joint_scales is not None
            assert len(occupied) == len(ann.data)
            for f, (xyv, joint_s) in enumerate(zip(ann.data, ann.joint_scales)):
                v = xyv[2]
                if v == 0.0:
                    continue

                if occupied.get(f, xyv[0], xyv[1]):
                    xyv[2] *= self.suppression
                else:
                    occupied.set(f, xyv[0], xyv[1], joint_s)  # joint_s = 2 * sigma

        if self.occupancy_visualizer is not None:
            LOG.debug('Occupied fields after NMS')
            self.occupancy_visualizer.predicted(occupied)

        for ann in anns:
            ann.data[ann.data[:, 2] < self.keypoint_threshold] = 0.0
        anns = [ann for ann in anns if ann.score >= self.instance_threshold]
        anns = sorted(anns, key=lambda a: -a.score)

        LOG.debug('nms = %.3fs', time.perf_counter() - start)
        return anns


class Detection:
    suppression = 0.1
    suppression_soft = 0.3
    instance_threshold = 0.15
    iou_threshold = 0.6
    iou_threshold_soft = 0.7

    @staticmethod
    def bbox_iou(box, other_boxes):
        box = np.expand_dims(box, 0)
        x1 = np.maximum(box[:, 0], other_boxes[:, 0])
        y1 = np.maximum(box[:, 1], other_boxes[:, 1])
        x2 = np.minimum(box[:, 0] + box[:, 2], other_boxes[:, 0] + other_boxes[:, 2])
        y2 = np.minimum(box[:, 1] + box[:, 3], other_boxes[:, 1] + other_boxes[:, 3])
        inter_area = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        box_area = box[:, 2] * box[:, 3]
        other_areas = other_boxes[:, 2] * other_boxes[:, 3]
        return inter_area / (box_area + other_areas - inter_area + 1e-5)

    def annotations_per_category(self, anns, method=2, sigma=0.5, nms_type='both'):
        start = time.perf_counter()

        dict_anns = defaultdict(list)
        all_boxes = defaultdict(list)

        for ann in anns:
            dict_anns[ann.category].append(ann)

        if not anns:
            return anns



        ret_anns = []
        for cat in dict_anns.keys():
            dict_anns[cat] = sorted(dict_anns[cat], key=lambda a: -a.score)
            all_boxes[cat] = np.stack([ann.bbox for ann in dict_anns[cat]])

            for ann_i, ann in enumerate(dict_anns[cat][1:], start=1):
                ious = self.bbox_iou(ann.bbox, all_boxes[cat][:ann_i])
                max_iou = np.max(ious)

                if method == 1:  # linear
                    weight = 1 - max_iou
                elif method == 2:  # gaussian
                    weight = np.exp(-(max_iou * max_iou) / sigma)
                else:  # original NMS
                    weight = 0

                if nms_type in ('both', 'nms') and max_iou > self.iou_threshold:
                    ann.score *= 0
                elif nms_type in ('both', 'snms') and max_iou > self.iou_threshold_soft:
                    ann.score *= weight

            for ann in dict_anns[cat]:
                if ann.score > 0.1: #self.instance_threshold:
                    ret_anns.append(ann)

        anns = sorted(ret_anns, key=lambda a: -a.score)

        LOG.debug('nms = %.3fs', time.perf_counter() - start)
        return anns

    def annotations(self, anns):
        start = time.perf_counter()

        anns = [ann for ann in anns if ann.score >= self.instance_threshold]
        if not anns:
            return anns
        anns = sorted(anns, key=lambda a: -a.score)

        all_boxes = np.stack([ann.bbox for ann in anns])
        for ann_i, ann in enumerate(anns[1:], start=1):
            mask = [ann.score >= self.instance_threshold for ann in anns[:ann_i]]
            ious = self.bbox_iou(ann.bbox, all_boxes[:ann_i][mask])
            max_iou = np.max(ious)

            if max_iou > self.iou_threshold:
                ann.score *= self.suppression
            elif max_iou > self.iou_threshold_soft:
                ann.score *= self.suppression_soft

        anns = [ann for ann in anns if ann.score >= self.instance_threshold]
        anns = sorted(anns, key=lambda a: -a.score)

        LOG.debug('nms = %.3fs', time.perf_counter() - start)
        return anns
