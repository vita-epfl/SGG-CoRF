import argparse
from collections import defaultdict
import heapq
import logging
import time
import math
import copy
from typing import List

import heapq
import numpy as np
from scipy.special import softmax, expit

from openpifpaf.annotation import AnnotationDet
from .annotation import AnnotationRaf
from .raf_analyzer import RafAnalyzer
from openpifpaf.decoder import Decoder, utils
from .headmeta import Raf
from openpifpaf import headmeta,visualizer
from . import visualizer as visualizer_raf

from openpifpaf.functional import grow_connection_blend
LOG = logging.getLogger(__name__)

class CifDetRaf(Decoder):
    """Generate CifDetRaf from fields.

    :param: nms: set to None to switch off non-maximum suppression.
    """
    connection_method = 'blend'
    occupancy_visualizer = visualizer.Occupancy()
    force_complete = False
    greedy = False
    nms = True

    lamb = 0.7
    pred_dist = None
    add_weights = False
    add_reverse = False
    max_score = True

    def __init__(self,
                cifdet_metas: List[headmeta.CifDet],
                raf_metas: List[Raf],
                *,
                cifdet_visualizers=None,
                raf_visualizers=None):
        super().__init__()
        self.cifdet_metas = cifdet_metas
        self.raf_metas = raf_metas

        self.cifdet_visualizers = cifdet_visualizers
        if self.cifdet_visualizers is None:
            self.cifdet_visualizers = [visualizer.CifDet(meta) for meta in cifdet_metas]
        self.raf_visualizers = raf_visualizers
        if self.raf_visualizers is None:
            self.raf_visualizers = [visualizer_raf.Raf(meta) for meta in raf_metas]

        if self.nms is True:
            self.nms = utils.nms.Detection()


        self.confidence_scales = raf_metas[0].decoder_confidence_scales

        self.timers = defaultdict(float)

        # init by_target and by_source
        self.by_target = defaultdict(dict)

        self.by_source = defaultdict(dict)
        for j1 in range(len(self.raf_metas[0].obj_categories)):
            for j2 in range(len(self.raf_metas[0].obj_categories)):
                for raf_i in range(len(self.raf_metas[0].rel_categories)):
                    self.by_source[j1][j2] = (raf_i, True)
                    self.by_source[j2][j1] = (raf_i, True)

    @classmethod
    def factory(cls, head_metas):
        # TODO: multi-scale
        return [
            CifDetRaf([meta], [meta_next])
            for meta, meta_next in zip(head_metas[:-1], head_metas[1:])
            if (isinstance(meta, headmeta.CifDet)
                and isinstance(meta_next, Raf))
        ]
    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""
        # check consistency

        cls.greedy = args.greedy
        cls.connection_method = args.connection_method

    def __call__(self, fields, initial_annotations=None, meta=None):
        start = time.perf_counter()
        if not initial_annotations:
            initial_annotations = []
        LOG.debug('initial annotations = %d', len(initial_annotations))

        for vis, meta in zip(self.cifdet_visualizers, self.cifdet_metas):
            vis.predicted(fields[meta.head_index])
        for vis, meta in zip(self.raf_visualizers, self.raf_metas):
            vis.predicted(fields[meta.head_index])

        cifdethr = utils.CifDetHr().fill(fields, self.cifdet_metas)
        seeds = utils.CifDetSeeds(cifdethr.accumulated).fill(fields, self.cifdet_metas)

        raf_analyzer = RafAnalyzer(cifdethr.accumulated).fill(fields, self.raf_metas)

        occupied = utils.Occupancy(cifdethr.accumulated.shape, 2, min_scale=4)
        annotations_det = []
        if self.pred_dist is None and not self.raf_metas[0].fg_matrix is None:
            self.pred_dist = np.log(self.raf_metas[0].fg_matrix / (self.raf_metas[0].fg_matrix.sum(2)[:, :, None] + 1e-08) + 1e-3)
            self.pred_dist = softmax(self.pred_dist, axis=2)

        def mark_occupied(ann):
            for joint_i, xyv in enumerate(ann.data):
                if xyv[2] == 0.0:
                    continue

                width = ann.joint_scales[joint_i]
                occupied.set(joint_i, xyv[0], xyv[1], width)  # width = 2 * sigma

        for v, f, x, y, w, h in seeds.get():
            if occupied.get(f, x, y):
                continue
            ann = AnnotationDet(self.cifdet_metas[0].categories).set(f + 1, v, (x - w/2.0, y - h/2.0, w, h))
            annotations_det.append(ann)
            #mark_occupied(ann)
            occupied.set(f, x, y, 0.1 * min(w, h))

        dict_rel = {}
        dict_rel_cnt = {}
        if self.nms is not None:
            #annotations_det = self.nms.annotations(annotations_det)
            annotations_det = self.nms.annotations_per_category(annotations_det, nms_type='snms')

        annotations = []
        for raf_v, index_s, x_s, y_s, raf_i, index_o, x_o, y_o in raf_analyzer.triplets[(-raf_analyzer.triplets[:,0]).argsort()]:
            s_idx = None
            o_idx = None
            min_value_s = None
            min_value_o = None
            for ann_idx, ann in enumerate(annotations_det):
                # if not(ann.category_id-1 == index_s or ann.category_id-1 == index_o):
                #     continue
                a = ann.bbox[0] + ann.bbox[2]/2.0
                b = ann.bbox[1] + ann.bbox[3]/2.0
                curr_dist = (1/(raf_v*ann.score+0.00001))*(math.sqrt((a - x_s)**2+(b - y_s)**2))
                if min_value_s is None or curr_dist<min_value_s:
                    min_value_s = curr_dist
                    s_idx = ann_idx
                curr_dist = (1/(raf_v*ann.score+0.00001))*(math.sqrt((a - x_o)**2+(b - y_o)**2))
                if min_value_o is None or curr_dist<min_value_o:
                    min_value_o = curr_dist
                    o_idx = ann_idx
            if (s_idx, raf_i, o_idx) in dict_rel:
                indx = dict_rel[(s_idx, raf_i, o_idx)]-1
                cnt = dict_rel_cnt[(s_idx, raf_i, o_idx)]
                category_id_obj = annotations[indx].category_id_obj
                category_id_sub = annotations[indx].category_id_sub
                if not self.pred_dist is None:
                    weight = self.lamb*(self.pred_dist[category_id_sub-1, category_id_obj-1, int(raf_i)])
                    weight = (weight + (1-self.lamb)*self.raf_metas[0].smoothing_pred[int(raf_i)])
                if self.add_weights:
                    score_rel = expit(5*(raf_v + weight-1))
                else:
                    score_rel = raf_v*weight
                if self.max_score:
                    annotations[indx].score_rel = max(annotations[indx].score_rel, score_rel)
                else:
                    annotations[indx].score_rel = (annotations[indx].score_rel*cnt + score_rel)/(cnt+1)
                dict_rel_cnt[(s_idx, raf_i, o_idx)] = cnt + 1
                if self.add_reverse:
                    indx = dict_rel[(o_idx, raf_i, s_idx)]-1
                    cnt = dict_rel_cnt[(o_idx, raf_i, s_idx)]
                    if not self.pred_dist is None:
                        weight = self.lamb*(self.pred_dist[category_id_obj-1, category_id_sub-1, int(raf_i)])
                        weight = (weight + (1-self.lamb)*self.raf_metas[0].smoothing_pred[int(raf_i)])
                    if self.add_weights:
                        score_rel = expit(5*(raf_v + weight-1))
                    else:
                        score_rel = raf_v*weight

                    annotations[indx].score_rel = (annotations[indx].score_rel*cnt + score_rel)/(cnt+1)
                    dict_rel_cnt[(o_idx, raf_i, s_idx)] = cnt + 1

            else:
                if s_idx is not None and o_idx is not None:
                    category_id_obj = annotations_det[o_idx].category_id
                    category_id_sub = annotations_det[s_idx].category_id
                    category_id_rel = int(raf_i) + 1
                    score_sub = annotations_det[s_idx].score
                    weight = 1.0
                    if not self.pred_dist is None:
                        weight = self.lamb*(self.pred_dist[category_id_sub-1, category_id_obj-1, int(raf_i)])
                        weight = (weight + (1-self.lamb)*self.raf_metas[0].smoothing_pred[int(raf_i)])
                    if self.add_weights:
                        score_rel = expit(5*(raf_v + weight-1))
                    else:
                        score_rel = raf_v * weight
                    score_obj = annotations_det[o_idx].score
                    bbox_sub = copy.deepcopy(annotations_det[s_idx].bbox)
                    bbox_obj = copy.deepcopy(annotations_det[o_idx].bbox)
                    ann = AnnotationRaf(self.raf_metas[0].obj_categories,
                                        self.raf_metas[0].rel_categories).set(
                                            category_id_obj, category_id_sub,
                                            category_id_rel, score_sub,
                                            score_rel, score_obj,
                                            bbox_sub, bbox_obj, idx_subj=s_idx, idx_obj=o_idx)
                    annotations.append(ann)
                    dict_rel[(s_idx, raf_i, o_idx)] = len(annotations)
                    dict_rel_cnt[(s_idx, raf_i, o_idx)] = 1
                    if self.add_reverse:
                        if not self.pred_dist is None:
                            weight = self.lamb*(self.pred_dist[category_id_obj-1, category_id_sub-1, int(raf_i)])
                            weight = (weight + (1-self.lamb)*self.raf_metas[0].smoothing_pred[int(raf_i)])
                        if self.add_weights:
                            score_rel = expit(5*(raf_v + weight-1))
                        else:
                            score_rel = raf_v * weight
                        ann = AnnotationRaf(self.raf_metas[0].obj_categories,
                                            self.raf_metas[0].rel_categories).set(
                                                category_id_sub, category_id_obj,
                                                category_id_rel, score_obj,
                                                score_rel, score_sub,
                                                bbox_obj, bbox_sub, idx_subj=s_idx, idx_obj=o_idx)
                        annotations.append(ann)
                        dict_rel[(o_idx, raf_i, s_idx)] = len(annotations)
                        dict_rel_cnt[(o_idx, raf_i, s_idx)] = 1

        self.occupancy_visualizer.predicted(occupied)

        LOG.info('annotations %d, %.3fs', len(annotations), time.perf_counter() - start)

        return annotations, annotations_det
