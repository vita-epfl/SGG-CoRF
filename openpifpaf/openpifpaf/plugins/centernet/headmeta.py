from dataclasses import dataclass, field
from openpifpaf import headmeta
from typing import Any, ClassVar, List, Tuple

@dataclass
class Raf_laplace(headmeta.Base):
    obj_categories: List[str]
    rel_categories: List[str]
    sigmas: List[float] = None
    only_in_field_of_view: bool = False

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 2
    ignore_rel = False
    pairwise_ignore = False
    detach_deform = False
    detach_offset = False

    vector_offsets = [True, True]
    decoder_min_distance = 0.0
    decoder_max_distance = float('inf')
    decoder_confidence_scales: List[float] = None
    fg_matrix = None
    bg_matrix = None
    smoothing_pred = None
    refine_hm = False
    deform_conv = False
    biggerparam = False
    use_gtoffset = False
    hadamard_product = False
    concat_offset = False
    refine_feature = False
    fourier_features = False
    giou_loss = False

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
class CenterNet(headmeta.Base):
    """Head meta data for a Composite Intensity Field (CIF) for Detection."""

    categories: List[str]

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 0
    max_objs = 128
    vector_offsets = [True, False]
    vector_softplus = [False, True]
    decoder_min_scale = 0.0
    prior = None
    giou_loss = False
    training_weights: List[float] = None
    kernel_1 = False
    single_head = False

    @property
    def n_fields(self):
        return len(self.categories)


@dataclass
class CifDet_CN(headmeta.CifDet):
    """Head meta data for a Composite Intensity Field (CIF) for Detection."""

    categories: List[str]

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 0
    max_objs = 128
    vector_offsets = [True, False]
    vector_softplus = [False, False]
    decoder_min_scale = 0.0
    giou_loss = False

    training_weights: List[float] = None
    refine_feature = False

    @property
    def n_fields(self):
        return len(self.categories)

@dataclass
class Raf_CNs(headmeta.Base):
    obj_categories: List[str]
    rel_categories: List[str]
    sigmas: List[float] = None
    only_in_field_of_view: bool = False
    max_objs = 128
    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 3
    n_scales: ClassVar[int] = 0

    vector_offsets = [True, True, False]
    decoder_min_distance = 0.0
    decoder_max_distance = float('inf')
    decoder_confidence_scales: List[float] = None
    fg_matrix = None
    bg_matrix = None
    smoothing_pred = None
    refine_hm = False
    deform_conv = False
    biggerparam = False
    use_gtoffset = False
    refine_offset = False

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
class Raf_CN(headmeta.Base):
    obj_categories: List[str]
    rel_categories: List[str]
    sigmas: List[float] = None
    only_in_field_of_view: bool = False

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 2
    ignore_rel = False
    pairwise_ignore = False
    detach_deform = False
    detach_offset = False

    vector_offsets = [True, True]
    decoder_min_distance = 0.0
    decoder_max_distance = float('inf')
    decoder_confidence_scales: List[float] = None
    fg_matrix = None
    bg_matrix = None
    smoothing_pred = None
    refine_hm = False
    deform_conv = False
    biggerparam = False
    use_gtoffset = False
    hadamard_product = False
    concat_offset = False
    giou_loss = False
    refine_feature = False

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
class Raf_GDeform(headmeta.Base):
    obj_categories: List[str]
    rel_categories: List[str]
    sigmas: List[float] = None
    only_in_field_of_view: bool = False

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 2
    ignore_rel = False
    pairwise_ignore = False
    detach_deform = False
    detach_offset = False

    vector_offsets = [True, True]
    decoder_min_distance = 0.0
    decoder_max_distance = float('inf')
    decoder_confidence_scales: List[float] = None
    fg_matrix = None
    bg_matrix = None
    smoothing_pred = None
    refine_hm = False
    deform_conv = False
    biggerparam = False
    use_gtoffset = False
    hadamard_product = False
    concat_offset = False
    refine_feature = False
    fourier_features = False
    giou_loss = False
    pairwise = False


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
class Raf_CAF(headmeta.Base):
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
    refine_hm = False
    deform_conv = False
    biggerparam = False
    use_gtoffset = False

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
class CenterNet_FPN(headmeta.Base):
    """Head meta data for a Composite Intensity Field (CIF) for Detection."""

    categories: List[str]

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 0
    max_objs = 128
    vector_offsets = [True, False]
    vector_softplus = [False, True]
    decoder_min_scale = 0.0
    prior = None
    giou_loss = False
    training_weights: List[float] = None
    kernel_1 = False
    single_head = False
    fpn_idx = []
    fpn_stride = []
    @property
    def n_fields(self):
        return len(self.categories)


@dataclass
class Raf_FPN(headmeta.Base):
    obj_categories: List[str]
    rel_categories: List[str]
    sigmas: List[float] = None
    only_in_field_of_view: bool = False

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 2
    ignore_rel = False
    pairwise_ignore = False
    detach_deform = False
    detach_offset = False

    vector_offsets = [True, True]
    decoder_min_distance = 0.0
    decoder_max_distance = float('inf')
    decoder_confidence_scales: List[float] = None
    fg_matrix = None
    bg_matrix = None
    smoothing_pred = None
    refine_hm = False
    deform_conv = False
    biggerparam = False
    use_gtoffset = False
    hadamard_product = False
    concat_offset = False
    giou_loss = False
    refine_feature = False
    fpn_idx = []
    fpn_stride = []

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
