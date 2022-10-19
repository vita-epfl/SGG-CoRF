import logging
import time

from typing import List

import numpy as np

# pylint: disable=import-error
from .functional import scalar_values_3d
from . import headmeta

LOG = logging.getLogger(__name__)

class RafScored:
    default_score_th = 0.2

    def __init__(self, cifhr, *, score_th=None, cif_floor=0.1):
        self.cifhr = cifhr
        self.score_th = score_th or self.default_score_th
        self.cif_floor = cif_floor

        self.forward = None
        self.backward = None

    def directed(self, caf_i, forward):
        if forward:
            return self.forward[caf_i], self.backward[caf_i]

        return self.backward[caf_i], self.forward[caf_i]

    def rescore(self, nine, joint_t):
        if self.cif_floor < 1.0 and joint_t < len(self.cifhr):
            cifhr_t = scalar_values(self.cifhr[joint_t], nine[3], nine[4], default=0.0)
            nine[0] = nine[0] * (self.cif_floor + (1.0 - self.cif_floor) * cifhr_t)
        return nine[:, nine[0] > self.score_th]

    def fill_single(self, all_fields, meta: headmeta.Raf):
        start = time.perf_counter()
        raf = all_fields[meta.head_index]

        if self.forward is None:
            self.forward = [np.empty((9, 0), dtype=raf.dtype) for _ in raf]
            self.backward = [np.empty((9, 0), dtype=raf.dtype) for _ in raf]

        for raf_i, nine in enumerate(raf):
            assert nine.shape[0] == 9
            mask = nine[0] > self.score_th
            if not np.any(mask):
                continue
            nine = nine[:, mask]

            if meta.decoder_min_distance:
                dist = np.linalg.norm(nine[1:3] - nine[5:7], axis=0)
                mask_dist = dist > meta.decoder_min_distance / meta.stride
                nine = nine[:, mask_dist]

            if meta.decoder_max_distance:
                dist = np.linalg.norm(nine[1:3] - nine[5:7], axis=0)
                mask_dist = dist < meta.decoder_max_distance / meta.stride
                nine = nine[:, mask_dist]

            nine[(1, 2, 3, 4, 5, 6, 7, 8), :] *= meta.stride

            scores = nine[0]
            cifhr_values = scalar_values_3d(self.cifhr, nine[1], nine[2], default=0.0)
            cifhr_s = np.max(cifhr_values, axis=0)
            index_s = np.amax(cifhr_values, axis=0)
            scores_s = scores * (self.cif_floor + (1.0 - self.cif_floor) * cifhr_s)
            mask_s = scores_s > self.score_th
            d9_s = np.copy(nine[:, mask_s][(0, 5, 6, 7, 8, 1, 2, 3, 4), :])
            d9_s[0] = scores_s[mask_s]
            self.backward[raf_i] = np.concatenate((self.backward[raf_i], d9_s), axis=1)

            cifhr_values = scalar_values_3d(self.cifhr, nine[5], nine[6], default=0.0)
            cifhr_o = np.max(cifhr_values, axis=0)
            index_o = np.amax(cifhr_values, axis=0)
            scores_o = scores * (self.cif_floor + (1.0 - self.cif_floor) * cifhr_o)
            mask_o = scores_o > self.score_th
            d9_o = np.copy(nine[:, mask_o])
            d9_o[0] = scores_o[mask_o]
            self.forward[raf_i] = np.concatenate((self.forward[raf_i], d9_o), axis=1)

        LOG.debug('scored caf (%d, %d) in %.3fs',
                  sum(f.shape[1] for f in self.forward),
                  sum(b.shape[1] for b in self.backward),
                  time.perf_counter() - start)
        return self

    def fill(self, all_fields, metas: List[headmeta.Raf]):
        for meta in metas:
            self.fill_single(all_fields, meta)

        return self
