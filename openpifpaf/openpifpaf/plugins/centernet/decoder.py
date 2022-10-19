from collections import defaultdict
import logging
import time
from typing import List
import numpy as np
import torch

import openpifpaf
from openpifpaf.decoder import Decoder, utils
from openpifpaf.annotation import AnnotationDet
from openpifpaf import visualizer
from .visualizer import CenterNet as CenterNetVisualizer
from .losses_util import _transpose_and_gather_feat, _gather_feat
from . import headmeta

LOG = logging.getLogger(__name__)

class CenterNet(Decoder):
    def __init__(self, head_metas: List[headmeta.CenterNet], *, visualizers=None):
        super().__init__()
        self.metas = head_metas

        self.visualizers = visualizers
        if self.visualizers is None:
            self.visualizers = [CenterNetVisualizer(meta) for meta in self.metas]

        self.timers = defaultdict(float)

    @classmethod
    def factory(cls, head_metas):
        # TODO: multi-scale
        decoder_meta = []
        for meta in head_metas:
            if isinstance(meta, headmeta.CenterNet):
                decoder_meta.append(meta)
        return [
            CenterNet(decoder_meta)
        ]

    def _ctdet_decode(self, heat, wh, reg=None, cat_spec_wh=False, K=100):
        batch, cat, height, width = heat.size()

        # heat = torch.sigmoid(heat)
        # perform nms on heatmaps
        def _nms(heat, kernel=3):
            pad = (kernel - 1) // 2

            hmax = torch.nn.functional.max_pool2d(
                heat, (kernel, kernel), stride=1, padding=pad)
            keep = (hmax == heat).float()
            return heat * keep

        def _topk(scores, K=40):
            batch, cat, height, width = scores.size()

            # Modified this from Centernet to support small maps
            topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), min(K, scores.shape[-2]*scores.shape[-1]))

            topk_inds = topk_inds % (height * width)
            topk_ys   = (topk_inds // width).int().float()
            topk_xs   = (topk_inds % width).int().float()

            topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
            topk_clses = (topk_ind // K).int()
            topk_inds = _gather_feat(
                topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
            topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
            topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

            return topk_score, topk_inds, topk_clses, topk_ys, topk_xs
        heat = _nms(heat)

        scores, inds, clses, ys, xs = _topk(heat, K=K)
        if reg is not None:
            reg = _transpose_and_gather_feat(reg, inds)
            reg = reg.view(batch, K, 2)
            xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
            ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        else:
            xs = xs.view(batch, K, 1) + 0.5
            ys = ys.view(batch, K, 1) + 0.5
        wh = _transpose_and_gather_feat(wh, inds)
        if cat_spec_wh:
            wh = wh.view(batch, K, cat, 2)
            clses_ind = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2).long()
            wh = wh.gather(2, clses_ind).view(batch, K, 2)
        else:
            wh = wh.view(batch, K, 2)
        clses  = clses.view(batch, K, 1).float()
        scores = scores.view(batch, K, 1)
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        detections = torch.cat([bboxes, scores, clses], dim=2)

        return detections

    def scale_scalar(self, field, stride):
        field = np.repeat(field, stride, 0)
        field = np.repeat(field, stride, 1)

        # center (the result is technically still off by half a pixel)
        half_stride = stride // 2
        return field[half_stride:-half_stride + 1, half_stride:-half_stride + 1]

    def __call__(self, fields, ret_hmhr=False):
        start = time.perf_counter()

        if self.visualizers:
            for vis, meta in zip(self.visualizers, self.metas):
                vis.predicted(fields[meta.head_index])
        meta = self.metas[-1]
        annotations = []
        field = fields[meta.head_index]
        hm = torch.unsqueeze(field[:meta.n_fields], 0)#.sigmoid_()
        reg = torch.unsqueeze(field[meta.n_fields:meta.n_fields+2], 0)
        wh = torch.unsqueeze(field[meta.n_fields+2:meta.n_fields+4], 0)
        detections = self._ctdet_decode(hm, wh, reg=reg, cat_spec_wh=False).cpu().numpy()
        for x1, y1, x2, y2, scores, clses in detections[0]:
            x1 *= meta.stride
            x2 *= meta.stride
            y1 *= meta.stride
            y2 *= meta.stride
            ann = AnnotationDet(self.metas[-1].categories).set(
                int(clses) + 1, scores, (x1, y1, x2-x1, y2-y1))
            annotations.append(ann)

        #annotations = utils.nms.Detection().annotations_per_category(annotations, nms_type='snms')

        LOG.info('annotations %d, decoder = %.3fs', len(annotations), time.perf_counter() - start)
        if ret_hmhr:
            return annotations, self.scale_scalar(hm.numpy()[0], meta.stride)
        return annotations
