from dataclasses import dataclass, field
from openpifpaf import headmeta
from typing import Any, ClassVar, List, Tuple

@dataclass
class Raf(headmeta.Base):
    obj_categories: List[str]
    rel_categories: List[str]
    sigmas: List[float] = None
    only_in_field_of_view: bool = False

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 2
    ignore_rel = False
    pairwise_ignore = False
    
    vector_offsets = [True, True]
    decoder_min_distance = 0.0
    decoder_max_distance = float('inf')
    decoder_confidence_scales: List[float] = None
    fg_matrix = None
    bg_matrix = None
    smoothing_pred = None

    training_weights: List[float] = None
    @property
    def n_fields(self):
        return len(self.rel_categories)

    @staticmethod
    def concatenate(metas):
        # TODO: by keypoint name, update skeleton indices if meta.keypoints
        # is not the same for all metas.
        concatenated = Raf(
            name='_'.join(m.name for m in metas),
            dataset=metas[0].dataset,
            obj_categories=metas[0].obj_categories,
            rel_categories=metas[0].rel_categories,
            sigmas=metas[0].sigmas,
            only_in_field_of_view=metas[0].only_in_field_of_view,
            decoder_confidence_scales=[
                s
                for meta in metas
                for s in (meta.decoder_confidence_scales
                          if meta.decoder_confidence_scales
                          else [1.0 for _ in meta.skeleton])
            ]
        )
        return concatenated

@dataclass
class Raf_dcn(headmeta.Base):
    obj_categories: List[str]
    rel_categories: List[str]
    sigmas: List[float] = None
    only_in_field_of_view: bool = False
    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_offsets: ClassVar[int] = 0
    n_scales: ClassVar[int] = 2
    ignore_rel = False
    upsample_stride = 1
    pairwise_ignore = False

    vector_offsets = [True, True]
    decoder_min_distance = 0.0
    decoder_max_distance = float('inf')
    decoder_confidence_scales: List[float] = None
    fg_matrix = None
    bg_matrix = None
    smoothing_pred = None

    @property
    def n_fields(self):
        return len(self.rel_categories)

    @staticmethod
    def concatenate(metas):
        # TODO: by keypoint name, update skeleton indices if meta.keypoints
        # is not the same for all metas.
        concatenated = Raf(
            name='_'.join(m.name for m in metas),
            dataset=metas[0].dataset,
            obj_categories=metas[0].obj_categories,
            rel_categories=metas[0].rel_categories,
            sigmas=metas[0].sigmas,
            only_in_field_of_view=metas[0].only_in_field_of_view,
            decoder_confidence_scales=[
                s
                for meta in metas
                for s in (meta.decoder_confidence_scales
                          if meta.decoder_confidence_scales
                          else [1.0 for _ in meta.skeleton])
            ]
        )
        return concatenated

@dataclass
class CifDet_deep(headmeta.CifDet):
    @property
    def n_fields(self):
        return len(self.categories)

@dataclass
class CifDet_deepShared(headmeta.CifDet):
    @property
    def n_fields(self):
        return len(self.categories)
