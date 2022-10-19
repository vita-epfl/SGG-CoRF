import logging
import time
import torch
from typing import List

import numpy as np

# pylint: disable=import-error
from . import headmeta

LOG = logging.getLogger(__name__)

class RafAnalyzer:
    default_score_th = 0.1
    def __init__(self, *, score_th=None, cif_floor=0.1):
        self.score_th = score_th or self.default_score_th
        self.cif_floor = cif_floor

        #self.triplets = defaultdict([])
        self.triplets = torch.empty((0,8), device=torch.device('cuda:0'))


    def fill_single(self, all_fields, meta: headmeta.Raf_CN):
        start = time.perf_counter()

        raf = all_fields[meta.head_index]
        mask = raf[:,0] > self.score_th
        if not torch.any(mask):
            return self
        if raf.shape[1]==5:
            raf = torch.cat((raf,
                    torch.zeros((raf.shape[0], 1, *raf.shape[2:]), device=torch.device('cuda:0')),
                    torch.zeros((raf.shape[0], 1, *raf.shape[2:]), device=torch.device('cuda:0'))), dim=1)
        raf_idx = torch.arange(raf.shape[0], device=torch.device('cuda:0'))[(...,) +(None,)*3].expand(raf.shape[0], 1, *raf.shape[2:])
        raf = torch.cat((raf, raf_idx), dim=1)
        raf[:, (1, 2, 3, 4, 5, 6), :] *= meta.stride
        #nine = raf[mask.expand(raf.size())]
        self.triplets = raf.permute(1,0,2,3)[:, mask].T
        self.triplets = self.triplets[:, (0,5,1,2,7,6,3,4)]#.cpu().numpy()

        return self



    def fill(self, all_fields, metas: List[headmeta.Raf_CN]):
        for meta in metas:
            self.fill_single(all_fields, meta)

        return self
