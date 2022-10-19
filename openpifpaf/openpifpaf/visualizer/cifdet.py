import copy
import logging
import numpy as np

from .base import Base
from ..annotation import AnnotationDet
from .. import headmeta, show

try:
    import matplotlib.cm
    CMAP_GREENS_NAN = copy.copy(matplotlib.cm.get_cmap('Greens'))
    CMAP_GREENS_NAN.set_bad('white', alpha=0.5)
except ImportError:
    CMAP_GREENS_NAN = None

LOG = logging.getLogger(__name__)


class CifDet(Base):
    def __init__(self, meta: headmeta.CifDet):
        super().__init__(meta.name)
        self.meta = meta
        self.annotation_painter = show.AnnotationPainter()

    def targets(self, field, *, annotation_dicts, metas=None):
        assert self.meta.categories is not None
        if len(self.indices('confidence'))==0 and len(self.indices('regression'))==0:
            return
        annotations = [
            AnnotationDet(self.meta.categories).set(ann['category_id'], None, ann['bbox'])
            for ann in annotation_dicts
        ]

        self._confidences(field[:, 0], annotation_dicts, metas)
        self._regressions(field[:, 1:3], field[:, 3:5],
                          confidence_fields=field[:, 0],
                          annotations=annotations)

    def predicted(self, field):
        self._confidences(field[:, 0], None, None)
        self._regressions(field[:, 1:3], field[:, 3:5],
                          annotations=self._ground_truth,
                          confidence_fields=field[:, 0],
                          uv_is_offset=False)

    def _confidences(self, confidences, annotations, metas=None):
        f_range = self.indices('confidence')

        if len(self.indices('confidence'))==1 and self.indices('confidence')[0] == -1:
            f_range = np.arange(confidences.shape[0])[np.nanmax(confidences, axis=(1,2))>0.2]

        for f in f_range:
            LOG.debug('%s', self.meta.categories[f])

            with self.image_canvas(self._processed_image, margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                im = ax.imshow(self.scale_scalar(confidences[f], self.meta.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap=CMAP_GREENS_NAN)
                self.colorbar(ax, im)

    def _regressions(self, regression_fields, wh_fields, *,
                     annotations=None, confidence_fields=None, uv_is_offset=True):

        f_range = self.indices('regression')
        if len(self.indices('regression'))==1 \
            and self.indices('regression')[0] == -1 \
            and not (confidence_fields is None):
            f_range = np.arange(confidence_fields.shape[0])[np.nanmax(confidence_fields, axis=(1,2))>0.2]
        for f in f_range:
            LOG.debug('%s', self.meta.categories[f])
            confidence_field = confidence_fields[f] if confidence_fields is not None else None

            with self.image_canvas(self._processed_image, margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                show.white_screen(ax, alpha=0.5)
                if annotations:
                    self.annotation_painter.annotations(ax, annotations, color='lightgray')
                q = show.quiver(ax,
                                regression_fields[f, :2],
                                confidence_field=confidence_field,
                                xy_scale=self.meta.stride, threshold=0.3, uv_is_offset=uv_is_offset,
                                cmap='Greens', clim=(0.5, 1.0), width=0.001)
                show.boxes_wh(ax, wh_fields[f, 0], wh_fields[f, 1],
                              confidence_field=confidence_field, threshold=0.3,
                              regression_field=regression_fields[f, :2],
                              xy_scale=self.meta.stride, cmap='Greens',
                              fill=False, linewidth=2,
                              regression_field_is_offset=uv_is_offset)
                if f in self.indices('margin', with_all=False):
                    show.margins(ax, regression_fields[f, :6], xy_scale=self.meta.stride)

                self.colorbar(ax, q)
