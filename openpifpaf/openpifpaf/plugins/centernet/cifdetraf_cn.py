import argparse
from collections import defaultdict
import heapq
import logging
import time
import math
import copy
from typing import List
import itertools
import copy

import heapq
import numpy as np
from scipy.special import softmax, expit, log_softmax, logit
from scipy.spatial.distance import cdist
from .visualizer import Prior as PriorVisualizer

from openpifpaf.annotation import AnnotationDet
from ..raf.annotation import AnnotationRaf_updated as AnnotationRaf
#from .raf_analyzerUpdated import RafAnalyzer as RafAnalyzer_updated
from .rafAnalyzer_cuda import RafAnalyzer as RafAnalyzer_updated
from openpifpaf.decoder import Decoder, utils
from . import headmeta
from . import visualizer as visualizer_centernet
from openpifpaf import visualizer
from ..raf import visualizer as visualizer_raf
from .decoder import CenterNet
from .losses_util import _transpose_and_gather_feat, _gather_feat
import torch
from openpifpaf.functional import grow_connection_blend

from .matching_utils import match_bboxes

LOG = logging.getLogger(__name__)

class CifDetRaf_CN(Decoder):
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
    prob_dist_rel = None
    add_weights = False
    add_reverse = False
    max_score = True
    timing = True
    fpn_interval_bbox = {'0': [0, 32**2], '1': [32**2, 64**2], '2': [64**2, 128**2], '3': [128**2, 512**2]}
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
        self.prior_visualizer = PriorVisualizer(self.cifdet_metas[-1])
        if self.nms is True:
            self.nms = utils.nms.Detection()


        self.confidence_scales = raf_metas[-1].decoder_confidence_scales

        self.timers = defaultdict(float)
        self.per_class = False
        # init by_target and by_source
        self.by_target = defaultdict(dict)

        self.by_source = defaultdict(dict)
        for j1 in range(len(self.raf_metas[-1].obj_categories)):
            for j2 in range(len(self.raf_metas[-1].obj_categories)):
                for raf_i in range(len(self.raf_metas[-1].rel_categories)):
                    self.by_source[j1][j2] = (raf_i, True)
                    self.by_source[j2][j1] = (raf_i, True)

    @classmethod
    def factory(cls, head_metas):
        # TODO: multi-scale
        cifdet_metas = []
        raf_metas = []
        for head_meta in head_metas:
            if isinstance(head_meta, headmeta.CifDet_CN) or isinstance(head_meta, headmeta.CenterNet) or isinstance(head_meta, headmeta.CenterNet_FPN):
                cifdet_metas.append(head_meta)
            elif isinstance(head_meta, headmeta.Raf_CN) or isinstance(head_meta, headmeta.Raf_CNs) or isinstance(head_meta, headmeta.Raf_CAF) or isinstance(head_meta, headmeta.Raf_GDeform) or isinstance(head_meta, headmeta.Raf_FPN): #or isinstance(meta_next, headmeta.Raf_dcn)
                raf_metas.append(head_meta)

        if len(cifdet_metas)==0 and len(raf_metas)==0:
            return []

        assert len(cifdet_metas) == len(raf_metas)
        return [
            CifDetRaf_CN(cifdet_metas, raf_metas)
        ]
    @classmethod
    def configure(cls, args: argparse.Namespace):
        """Take the parsed argument parser output and configure class variables."""
        # check consistency

        cls.greedy = args.greedy
        cls.connection_method = args.connection_method

    def solve_RAFCN_fasterer_cuda(self, raf_analyzer, annotations_det, predcls=False):
        dict_rel = {}
        annotations = []
        sorted_raf = raf_analyzer.triplets[torch.argsort(raf_analyzer.triplets[:,0], descending=True)]
        if len(sorted_raf)==0:
            return annotations
        #LOG.info('Raf Analyzer output %d', sorted_raf.shape[0])
        pred_bboxes = []
        for ann in annotations_det:
            bbox = ann.bbox#.copy()
            #bbox[:2] = bbox[:2] + bbox[2:]/2.0
            pred_bboxes.append(torch.tensor([[bbox[0] + bbox[2]/2.0, bbox[1] + bbox[3]/2.0, bbox[2], bbox[3], ann.category_id, ann.score]], device=torch.device('cuda:0')))
        pred_bboxes = torch.cat(pred_bboxes, dim=0)#.float()

        denom = pred_bboxes[:, -1:]*sorted_raf[:, 0].T
        cdist_subj = torch.cdist(pred_bboxes[:, :2].unsqueeze(0), sorted_raf[:, 2:4].unsqueeze(0).double())[0]
        cdist_obj = torch.cdist(pred_bboxes[:, :2].unsqueeze(0), sorted_raf[:, 6:8].unsqueeze(0).double())[0]
        curr_dist_subj = (1/(denom+0.00001))* cdist_subj
        curr_dist_obj = (1/(denom+0.00001))* cdist_obj

        sorted_raf = sorted_raf.cpu().numpy()
        id_s_perRel = torch.argmin(curr_dist_subj, 0)
        id_o_perRel = torch.argmin(curr_dist_obj, 0)

        id_s_perRel = id_s_perRel.cpu().numpy()
        id_o_perRel = id_o_perRel.cpu().numpy()
        pred_bboxes = pred_bboxes.cpu().numpy()
        # combined_rafs = np.concatenate([sorted_raf[:,4:5], np.expand_dims(id_s_perRel,1), np.expand_dims(id_o_perRel,1), sorted_raf[:,0:1]], axis=1)
        # sorted_idx = np.lexsort((combined_rafs[:,0], combined_rafs[:,1],combined_rafs[:,2], combined_rafs[:, 3]))
        # combined_rafs = combined_rafs[sorted_idx]
        # sorted_raf = sorted_raf[sorted_idx]
        # combined_rafs_ids = unique_return_inverse_2D_viewbased(combined_rafs)
        # sorted_raf = np.split(sorted_raf, np.unique(combined_rafs_ids, return_index=True)[1][1:])

        dict_uniques = defaultdict(list)
        for idx, (s_idx, o_idx, rel) in enumerate(zip(id_s_perRel, id_o_perRel, sorted_raf)):
            if s_idx == o_idx:
                continue
            if not (s_idx, o_idx, rel[4]) in dict_uniques:
                dict_uniques[(s_idx, o_idx, rel[4])] = [idx, rel[0]]
            elif dict_uniques[(s_idx, o_idx, rel[4])][1] < rel[0]:
                dict_uniques[(s_idx, o_idx, rel[4])] = [idx, rel[0]]

        if len(dict_uniques)==0:
            return annotations
        mask_idx = np.asarray(list(dict_uniques.values()))[:, 0].astype(int)

        if False:
            #no_rel_mask = (self.prob_dist_rel[pred_bboxes[id_s_perRel[mask_idx], 4].astype(int)-1, pred_bboxes[id_o_perRel[mask_idx], 4].astype(int)-1, 1:]>0).any(1)
            #mask_idx *= no_rel_mask
            weights = self.pred_dist[pred_bboxes[id_s_perRel[mask_idx], 4].astype(int)-1, pred_bboxes[id_o_perRel[mask_idx], 4].astype(int)-1, sorted_raf[mask_idx, 4].astype(int)+1]
            trimmed_rafs = sorted_raf[mask_idx]#[no_rel_mask]
            id_s_perRel = id_s_perRel[mask_idx]#[no_rel_mask]
            id_o_perRel = id_o_perRel[mask_idx]#[no_rel_mask]
            trimmed_rafs[:, 0] = trimmed_rafs[:, 0]*(weights)#[no_rel_mask])
        else:
            trimmed_rafs = sorted_raf[mask_idx]
            id_s_perRel = id_s_perRel[mask_idx]
            id_o_perRel = id_o_perRel[mask_idx]
        if not predcls:
            mask_score = trimmed_rafs[:, 0] > np.minimum(pred_bboxes[id_s_perRel, 5], pred_bboxes[id_o_perRel, 5])/2.0
            trimmed_rafs = trimmed_rafs[mask_score]
            id_s_perRel = id_s_perRel[mask_score]
            id_o_perRel = id_o_perRel[mask_score]

        # ind = np.argsort(trimmed_rafs[:,0], axis=0)[:100]
        # id_s_perRel = id_s_perRel[ind]
        # id_o_perRel = id_o_perRel[ind]

        #mask_remove =
        count_rel = 0
        for idx_rel, rel in enumerate(trimmed_rafs):
            s_idx = id_s_perRel[idx_rel]
            o_idx = id_o_perRel[idx_rel]
            # if len(annotations)==100:
            #     break

            if (s_idx, o_idx) in dict_rel:
                indx = dict_rel[(s_idx, o_idx)]-1
                annotations[indx].rel[int(rel[4])] = max(annotations[indx].rel[int(rel[4])], rel[0])
            else:
                ann = AnnotationRaf(self.raf_metas[-1].obj_categories,
                                    self.raf_metas[-1].rel_categories).set(
                                        obj=annotations_det[o_idx], subj=annotations_det[s_idx],
                                        category_id_rel=int(rel[4]) + 1,
                                        score_rel=rel[0], idx_subj=s_idx, idx_obj=o_idx)
                annotations.append(ann)
                dict_rel[(s_idx, o_idx)] = len(annotations)

        return annotations

    def __call__(self, fields, initial_annotations=None, meta=None):
        start = time.perf_counter()
        gt_anns = None
        if isinstance(fields, tuple):
            gt_anns = fields[1]
            if len(fields) == 3:
                inp_hm = fields[2]
                self.prior_visualizer.predicted(inp_hm)
            fields = fields[0]
            if len(gt_anns[0])==3:
                anns_gt, anns_det_gt, mask_hm = gt_anns[0]
                # fields[self.raf_metas[-1].head_index][:self.raf_metas[-1].n_fields, 0, mask_hm==0] = 0
            elif len(gt_anns[0])==4:
                anns_gt, anns_det_gt, gt_det, gt_raf = gt_anns[0]
                # gt_raf = gt_raf.to(fields[self.raf_metas[-1].head_index].device)
                # fields[self.raf_metas[-1].head_index][:, 1:5].permute(0,2,3,1)[gt_raf[:,0]==1,:] = gt_raf[:,1:5].permute(0,2,3,1)[gt_raf[:,0]==1,:]
            else:
                anns_gt, anns_det_gt = gt_anns[0]


        def apply(f, items):
            """Apply f in a nested fashion to all items that are not list or tuple."""
            if items is None:
                return None
            if isinstance(items, (list, tuple)):
                return [apply(f, i) for i in items]
            return f(items)
            #return anns_gt, anns_det_gt
        offset_1 = [None, None]

        # if self.raf_metas[-1].refine_feature:
        #     for i in range(len(fields))[-2:]:
        #         offset_1[i%2] = fields[i][:,-4:]
        #         fields[i] = fields[i][:,:-4]

        if not initial_annotations:
            initial_annotations = []
        LOG.debug('initial annotations = %d', len(initial_annotations))

        #if not self.timing:
        for vis, meta in zip(self.cifdet_visualizers, self.cifdet_metas):
            vis.predicted(fields[meta.head_index])
        for vis, meta in zip(self.raf_visualizers, self.raf_metas):
            vis.predicted(fields[meta.head_index], offset_1=offset_1[meta.head_index%2])

        annotations_det = []
        annotations_det = CenterNet([self.cifdet_metas[-1]])(fields)

        # with torch.autograd.profiler.record_function('tonumpy'):
        #     fields = apply(lambda x: x.cpu().numpy(), fields)

        ### BEWARE FOR VISUALIZATION
        # annotations_det = self.nms.annotations_per_category(annotations_det, nms_type='nms')
        # annotations_det = [ann for ann in annotations_det if ann.score >0.08]

        if self.pred_dist is None and not self.raf_metas[-1].fg_matrix is None:
            self.pred_dist = np.power(1.0001, self.raf_metas[-1].fg_matrix)
            #self.prob_dist_rel = log_softmax(self.raf_metas[-1].fg_matrix / (self.raf_metas[-1].fg_matrix.sum(2)[:, :, None] + 1e-03))
            self.prob_dist_rel = self.raf_metas[-1].fg_matrix / (self.raf_metas[-1].fg_matrix.sum(2)[:, :, None] + 1e-03)


        raf_analyzer = RafAnalyzer_updated().fill(fields, [self.raf_metas[-1]])
        annotations = self.solve_RAFCN_fasterer_cuda(raf_analyzer, annotations_det)

        if gt_anns:
            annotations_predcls = self.solve_RAFCN_fasterer_cuda(raf_analyzer, anns_det_gt, predcls=True)
            if isinstance(self.cifdet_metas[-1], headmeta.CifDet_CN):
                anns_det_sgcls = self.overlap_gt_pred_hr(anns_det_gt, cifdethr.accumulated)
            else:
                if isinstance(fields[self.cifdet_metas[-1].head_index], dict):
                    anns_det_sgcls = self.overlap_gt_pred_bbox(anns_det_gt, annotations_det)
                else:
                    # anns_det_sgcls = self.overlap_gt_pred(anns_det_gt, fields[self.cifdet_metas[-1].head_index][:-4].cpu().numpy())
                    anns_det_sgcls = self.overlap_gt_pred_bbox(anns_det_gt, annotations_det)
            annotations_sgcls = self.solve_RAFCN_fasterer_cuda(raf_analyzer, anns_det_sgcls)

        LOG.info('Detection annotations %d, %.3fs', len(annotations_det), time.perf_counter() - start)
        LOG.info('Relation annotations %d, %.3fs', len(annotations), time.perf_counter() - start)

        if gt_anns:
            return (annotations, annotations_det), (annotations_predcls, anns_det_gt), (annotations_sgcls, anns_det_sgcls)
        return annotations, annotations_det

    def overlap_gt_pred(self, anns_det_gt, hm):
        anns_det_gt = copy.deepcopy(anns_det_gt)

        # pred_bboxes = []
        # for ann in annotations_det:
        #     bbox = ann.bbox.copy()
        #     bbox[2:] = bbox[:2] + bbox[2:]
        #     pred_bboxes.append(bbox)
        # pred_bboxes = np.asarray(pred_bboxes)

        for ann in anns_det_gt:
            bbox = ann.bbox#.copy()
            ct = ((bbox[:2] + bbox[2:]/2)/self.cifdet_metas[-1].stride).astype(np.int32)
            hm_max = np.amax(hm[:, max(0,(ct[1]-1)):min(hm.shape[1],(ct[1]+2)), max(0, (ct[0]-1)):min(hm.shape[2],(ct[0]+2))], axis=(1,2))
            #hm_max = np.argmax(hm_max, axis=0)
            ann.category_id = np.argmax(hm_max, axis=0) + 1
            ann.score = hm_max[np.argmax(hm_max, axis=0)]

        return anns_det_gt

    def overlap_gt_pred_bbox(self, anns_det_gt, anns_det_pred):
        anns_det_gt = copy.deepcopy(anns_det_gt)

        pred_bboxes = []
        for ann in anns_det_pred:
            bbox = ann.bbox.copy()
            bbox[2:] = bbox[:2] + bbox[2:]
            pred_bboxes.append(bbox)
        pred_bboxes = np.asarray(pred_bboxes)

        gt_bboxes = []
        for ann in anns_det_gt:
            bbox = ann.bbox.copy()
            bbox[2:] = bbox[:2] + bbox[2:]
            gt_bboxes.append(bbox)
        gt_bboxes = np.asarray(gt_bboxes)

        idx_gt_actual, idx_pred_actual, ious_actual, labels = match_bboxes(gt_bboxes, pred_bboxes)
        for idx_gt, idx_pred in zip(idx_gt_actual, idx_pred_actual):
            anns_det_gt[idx_gt].category_id = anns_det_pred[idx_pred].category_id
            anns_det_gt[idx_gt].score = anns_det_pred[idx_pred].score

        return anns_det_gt
