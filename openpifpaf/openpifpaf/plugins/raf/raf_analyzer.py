import logging
import time

from typing import List

import numpy as np

# pylint: disable=import-error
from .functional import scalar_values_3d
from . import headmeta

LOG = logging.getLogger(__name__)

class RafAnalyzer:
    default_score_th = 0.1

    def __init__(self, cifhr, *, score_th=None, cif_floor=0.1):
        self.cifhr = cifhr
        self.score_th = score_th or self.default_score_th
        self.cif_floor = cif_floor

        #self.triplets = defaultdict([])
        self.triplets = np.empty((0,8))

    def scalar_values_3d_py(self, field, x, y, default_v=-1, scale=None):
        values_np = np.full((field.shape[0], x.shape[0],), default_v, dtype=np.float32)

        for i in range(values_np.shape[1]):
            if scale is not None:
              csigma = scale[i]
            else:
              csigma = 1.0
            cx = x[i]
            cy = y[i]
            minx = (np.clip(cx - 0.5*csigma, a_min=0, a_max=field.shape[2] - 1)).astype(np.int64)
            maxx = (np.clip(cx + 0.5*csigma, a_min=minx + 1, a_max=field.shape[2])).astype(np.int64)
            miny = (np.clip(cy - 0.5*csigma, a_min=0, a_max=field.shape[1] - 1)).astype(np.int64)
            maxy = (np.clip(cy + 0.5*csigma, a_min=miny + 1, a_max=field.shape[1])).astype(np.int64)

            values_np[:, i] = np.amax(field[:, miny:maxy, minx:maxx], axis=(1,2))

        return values_np
    def fill_single(self, all_fields, meta: headmeta.Raf):
        start = time.perf_counter()
        raf = all_fields[meta.head_index]

        for raf_i, nine in enumerate(raf):
            mask = nine[0] > self.score_th
            if not np.any(mask):
                continue
            nine = nine[:, mask]
            if nine.shape[0] == 9:
                nine[(1, 2, 3, 4, 5, 6, 7, 8), :] *= meta.stride
            else:
                nine[(1, 2, 3, 4, 5, 6), :] *= meta.stride
            cifhr_values = self.scalar_values_3d_py(self.cifhr, nine[1], nine[2], default_v=0.0, scale=nine[7] if nine.shape[0] == 9 else None)
            cifhr_s = np.max(cifhr_values, axis=0)
            index_s = np.argmax(cifhr_values, axis=0)
            #raf_s = nine[0] * (self.cif_floor + (1.0 - self.cif_floor) * cifhr_s)
            raf_s = (2/3)*nine[0] + (1/3)*cifhr_s

            cifhr_values = self.scalar_values_3d_py(self.cifhr, nine[3], nine[4], default_v=0.0, scale=nine[8] if nine.shape[0] == 9 else None)
            cifhr_o = np.max(cifhr_values, axis=0)
            index_o = np.argmax(cifhr_values, axis=0)
            #nine[0] = 0.5*(raf_s + nine[0] * (self.cif_floor + (1.0 - self.cif_floor) * cifhr_o))
            nine[0] = raf_s + (1/3)* cifhr_o

            mask = nine[0] > self.score_th
            index_s = index_s[mask]
            index_o = index_o[mask]
            nine = nine[:, mask]
            #self.triplets[index_s].append([nine[0], index_s, nine[1], nine[2], raf_i, index_o, nine[3], nine[4], False])
            self.triplets = np.concatenate((self.triplets, np.column_stack([nine[0], index_s, nine[1], nine[2], [raf_i]*nine[0].shape[0], index_o, nine[3], nine[4]])))

        return self



    def fill(self, all_fields, metas: List[headmeta.Raf]):
        for meta in metas:
            self.fill_single(all_fields, meta)

        return self
