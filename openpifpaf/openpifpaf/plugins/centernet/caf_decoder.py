import argparse
from collections import defaultdict
import heapq
import logging
import time
from typing import List

import numpy as np

from openpifpaf.decoder import Decoder, utils
from .decoder import CifDetHr, CifDetSeeds, CenterNet
from ..annotation import Annotation
from . import headmeta
from . import visualizer as visualizer_centernet
from ..raf import visualizer as visualizer_raf
# pylint: disable=import-error
from openpifpaf.functional import caf_center_s, grow_connection_blend

LOG = logging.getLogger(__name__)

class Cifdetraf_caf(Decoder):
    """Generate CifCaf poses from fields.

    :param: nms: set to None to switch off non-maximum suppression.
    """
    connection_method = 'blend'
    occupancy_visualizer = visualizer.Occupancy()
    force_complete = False
    greedy = False
    keypoint_threshold = 0.15
    keypoint_threshold_rel = 0.5
    dense_coupling = 0.0

    reverse_match = False

    def __init__(self,
                cifdet_metas: List[headmeta.CifDet_CN],
                raf_metas: List[headmeta.Raf_CN],
                *,
                cifdet_visualizers=None,
                raf_visualizers=None):
        super().__init__()
        self.cifdet_metas = cifdet_metas
        self.raf_metas = raf_metas

        self.cifdet_visualizers = cifdet_visualizers

        if isinstance(cifdet_metas[-1], headmeta.CifDet_CN):
            print("Network with CIFDet_CN head")
        else:
            print("Network with CenterNet head")
        if self.cifdet_visualizers is None:
            chosen_visualizer = visualizer.CifDet if isinstance(cifdet_metas[-1], headmeta.CifDet_CN) else visualizer_centernet.CenterNet
            self.cifdet_visualizers = [chosen_visualizer(meta) for meta in cifdet_metas]
        self.raf_visualizers = raf_visualizers
        if self.raf_visualizers is None:
            chosen_vis_raf = visualizer_centernet.Raf if isinstance(self.raf_metas[-1], headmeta.Raf_CNs) else visualizer_raf.Raf
            self.raf_visualizers = [chosen_vis_raf(meta) for meta in raf_metas]

        if self.nms is True:
            self.nms = utils.nms.Detection()


        self.confidence_scales = raf_metas[-1].decoder_confidence_scales
        self.timers = defaultdict(float)

        # init by_target and by_source
        self.by_source = defaultdict(dict)
        for j1 in range(len(self.raf_metas[-1].obj_categories)):
            for j2 in range(len(self.raf_metas[-1].obj_categories)):
                for raf_i in range(len(self.raf_metas[-1].rel_categories)):
                    if self.raf_metas[-1].fg_matrix[j1, j2, raf_i]>0:
                        self.by_source[j1][j2] = (raf_i, True)


    @classmethod
    def factory(cls, head_metas):
        # TODO: multi-scale
        cifdet_metas = []
        raf_metas = []
        for head_meta in head_metas:
            if isinstance(head_meta, headmeta.CifDet_CN) or isinstance(head_meta, headmeta.CenterNet):
                cifdet_metas.append(head_meta)
            elif isinstance(head_meta, headmeta.Raf_CN) or isinstance(head_meta, headmeta.Raf_CNs) or isinstance(head_meta, headmeta.Raf_CAF): #or isinstance(meta_next, headmeta.Raf_dcn)
                raf_metas.append(head_meta)

        if len(cifdet_metas)==0 and len(raf_metas)==0:
            return []

        assert len(cifdet_metas) == len(raf_metas)
        return [
            Cifdetraf_caf(cifdet_metas, raf_metas)
        ]

    def __call__(self, fields, initial_annotations=None):
        start = time.perf_counter()
        if not initial_annotations:
            initial_annotations = []
        LOG.debug('initial annotations = %d', len(initial_annotations))

        for vis, meta in zip(self.cif_visualizers, self.cif_metas):
            vis.predicted(fields[meta.head_index])
        for vis, meta in zip(self.caf_visualizers, self.caf_metas):
            vis.predicted(fields[meta.head_index])

        annotations_det = []
        if isinstance(self.cifdet_metas[-1], headmeta.CifDet_CN):
            cifdethr = CifDetHr().fill(fields, [self.cifdet_metas[-1]])
            seeds = CifDetSeeds(cifdethr.accumulated).fill(fields, [self.cifdet_metas[-1]])
            occupied = utils.Occupancy(cifdethr.accumulated.shape, 2, min_scale=4)
            # def mark_occupied(ann):
            #     for joint_i, xyv in enumerate(ann.data):
            #         if xyv[2] == 0.0:
            #             continue
            #
            #         width = ann.joint_scales[joint_i]
            #         occupied.set(joint_i, xyv[0], xyv[1], width)  # width = 2 * sigma

            for v, f, x, y, w, h in seeds.get():
                if occupied.get(f, x, y):
                    continue
                ann = AnnotationDet(self.cifdet_metas[-1].categories).set(f + 1, v, (x - w/2.0, y - h/2.0, w, h))
                annotations_det.append(ann)
                #mark_occupied(ann)
                occupied.set(f, x, y, 0.1 * min(w, h))
        else:
            annotations_det, cifdethr = CenterNet([self.cifdet_metas[-1]])(fields)

        caf_scored = utils.CafScored(cifdethr.accumulated).fill(fields, self.caf_metas)

        occupied = utils.Occupancy(cifdethr.accumulated.shape, 2, min_scale=4)
        annotations = []

        def mark_occupied(ann):
            for joint_i, xyv in enumerate(ann.data):
                if xyv[2] == 0.0:
                    continue

                width = ann.joint_scales[joint_i]
                occupied.set(joint_i, xyv[0], xyv[1], width)  # width = 2 * sigma

        for ann in initial_annotations:
            self._grow(ann, caf_scored)
            annotations.append(ann)
            mark_occupied(ann)

        for v, f, x, y, s in seeds.get():
            if occupied.get(f, x, y):
                continue

            ann = Annotation(self.keypoints,
                             self.out_skeleton,
                             score_weights=self.score_weights
                             ).add(f, (x, y, v))
            ann.joint_scales[f] = s
            self._grow(ann, caf_scored)
            annotations.append(ann)
            mark_occupied(ann)

            self.occupancy_visualizer.predicted(occupied)

            LOG.debug('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)

            if self.force_complete:
                annotations = self.complete_annotations(cifdethr, fields, annotations)

            if self.nms is not None:
                annotations = self.nms.annotations(annotations)

            LOG.info('%d annotations: %s', len(annotations),
                     [np.sum(ann.data[:, 2] > 0.1) for ann in annotations])

            return annotations, annotations_det

        def connection_value(self, ann, caf_scored, start_i, end_i, *, reverse_match=True):
            caf_i, forward = self.by_source[start_i][end_i]
            caf_f, caf_b = caf_scored.directed(caf_i, forward)
            xyv = ann.data[start_i]
            xy_scale_s = max(0.0, ann.joint_scales[start_i])

            only_max = self.connection_method == 'max'

            new_xysv = grow_connection_blend(
                caf_f, xyv[0], xyv[1], xy_scale_s, only_max)
            if new_xysv[3] == 0.0:
                return 0.0, 0.0, 0.0, 0.0
            keypoint_score = np.sqrt(new_xysv[3] * xyv[2])  # geometric mean
            if keypoint_score < self.keypoint_threshold:
                return 0.0, 0.0, 0.0, 0.0
            if keypoint_score < xyv[2] * self.keypoint_threshold_rel:
                return 0.0, 0.0, 0.0, 0.0
            xy_scale_t = max(0.0, new_xysv[2])

            # reverse match
            if self.reverse_match and reverse_match:
                reverse_xyv = grow_connection_blend(
                    caf_b, new_xysv[0], new_xysv[1], xy_scale_t, only_max)
                if reverse_xyv[2] == 0.0:
                    return 0.0, 0.0, 0.0, 0.0
                if abs(xyv[0] - reverse_xyv[0]) + abs(xyv[1] - reverse_xyv[1]) > xy_scale_s:
                    return 0.0, 0.0, 0.0, 0.0

            return (new_xysv[0], new_xysv[1], new_xysv[2], keypoint_score)

        @staticmethod
        def p2p_value(source_xyv, caf_scored, source_s, target_xysv, caf_i, forward):
            # TODO move to Cython (see grow_connection_blend)
            caf_f, _ = caf_scored.directed(caf_i, forward)
            xy_scale_s = max(0.0, source_s)

            # source value
            caf_field = caf_center_s(caf_f, source_xyv[0], source_xyv[1],
                                     sigma=2.0 * xy_scale_s)
            if caf_field.shape[1] == 0:
                return 0.0

            # distances
            d_source = np.linalg.norm(
                ((source_xyv[0],), (source_xyv[1],)) - caf_field[1:3], axis=0)
            d_target = np.linalg.norm(
                ((target_xysv[0],), (target_xysv[1],)) - caf_field[5:7], axis=0)

            # combined value and source distance
            xy_scale_t = max(0.0, target_xysv[2])
            sigma_s = 0.5 * xy_scale_s
            sigma_t = 0.5 * xy_scale_t
            scores = (
                np.exp(-0.5 * d_source**2 / sigma_s**2)
                * np.exp(-0.5 * d_target**2 / sigma_t**2)
                * caf_field[0]
            )
            return np.sqrt(source_xyv[2] * max(scores))

        def _grow(self, ann, caf_scored, *, reverse_match=True):
            frontier = []
            in_frontier = set()

            def add_to_frontier(start_i):
                for end_i, (caf_i, _) in self.by_source[start_i].items():
                    if ann.data[end_i, 2] > 0.0:
                        continue
                    if (start_i, end_i) in in_frontier:
                        continue

                    max_possible_score = np.sqrt(ann.data[start_i, 2])
                    if self.confidence_scales is not None:
                        max_possible_score *= self.confidence_scales[caf_i]
                    heapq.heappush(frontier, (-max_possible_score, None, start_i, end_i))
                    in_frontier.add((start_i, end_i))
                    ann.frontier_order.append((start_i, end_i))

            def frontier_get():
                while frontier:
                    entry = heapq.heappop(frontier)
                    if entry[1] is not None:
                        return entry

                    _, __, start_i, end_i = entry
                    if ann.data[end_i, 2] > 0.0:
                        continue

                    new_xysv = self.connection_value(
                        ann, caf_scored, start_i, end_i, reverse_match=reverse_match)
                    if new_xysv[3] == 0.0:
                        continue
                    score = new_xysv[3]
                    if self.greedy:
                        return (-score, new_xysv, start_i, end_i)
                    if self.confidence_scales is not None:
                        caf_i, _ = self.by_source[start_i][end_i]
                        score = score * self.confidence_scales[caf_i]
                    heapq.heappush(frontier, (-score, new_xysv, start_i, end_i))

            # seeding the frontier
            for joint_i, v in enumerate(ann.data[:, 2]):
                if v == 0.0:
                    continue
                add_to_frontier(joint_i)

            while True:
                entry = frontier_get()
                if entry is None:
                    break

                _, new_xysv, jsi, jti = entry
                if ann.data[jti, 2] > 0.0:
                    continue

                ann.data[jti, :2] = new_xysv[:2]
                ann.data[jti, 2] = new_xysv[3]
                ann.joint_scales[jti] = new_xysv[2]
                ann.decoding_order.append(
                    (jsi, jti, np.copy(ann.data[jsi]), np.copy(ann.data[jti])))
                add_to_frontier(jti)

        def _flood_fill(self, ann):
            frontier = []

            def add_to_frontier(start_i):
                for end_i, (caf_i, _) in self.by_source[start_i].items():
                    if ann.data[end_i, 2] > 0.0:
                        continue
                    start_xyv = ann.data[start_i].tolist()
                    score = xyv[2]
                    if self.confidence_scales is not None:
                        score = score * self.confidence_scales[caf_i]
                    heapq.heappush(frontier, (-score, end_i, start_xyv, ann.joint_scales[start_i]))

            for start_i, xyv in enumerate(ann.data):
                if xyv[2] == 0.0:
                    continue
                add_to_frontier(start_i)

            while frontier:
                _, end_i, xyv, s = heapq.heappop(frontier)
                if ann.data[end_i, 2] > 0.0:
                    continue
                ann.data[end_i, :2] = xyv[:2]
                ann.data[end_i, 2] = 0.00001
                ann.joint_scales[end_i] = s
                add_to_frontier(end_i)

        def complete_annotations(self, cifhr, fields, annotations):
            start = time.perf_counter()

            caf_scored = utils.CafScored(cifhr.accumulated, score_th=0.001).fill(
                fields, self.caf_metas)
            for ann in annotations:
                unfilled_mask = ann.data[:, 2] == 0.0
                self._grow(ann, caf_scored, reverse_match=False)
                now_filled_mask = ann.data[:, 2] > 0.0
                updated = np.logical_and(unfilled_mask, now_filled_mask)
                ann.data[updated, 2] = np.minimum(0.001, ann.data[updated, 2])

                # some joints might still be unfilled
                if np.any(ann.data[:, 2] == 0.0):
                    self._flood_fill(ann)

            LOG.debug('complete annotations %.3fs', time.perf_counter() - start)
            return annotations
