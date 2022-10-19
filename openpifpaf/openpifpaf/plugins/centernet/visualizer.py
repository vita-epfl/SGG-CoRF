import copy
import logging
import os
import numpy as np
import torch

from openpifpaf.visualizer import Base
from openpifpaf.annotation import AnnotationDet
from openpifpaf import show
from . import headmeta
#from .painters import CenterNetPainter
try:
    import matplotlib.cm
    CMAP_GREENS_NAN = copy.copy(matplotlib.cm.get_cmap('Greens'))
    CMAP_GREENS_NAN.set_bad('white', alpha=0.5)
except ImportError:
    CMAP_GREENS_NAN = None

LOG = logging.getLogger(__name__)


class CenterNet(Base):

    def __init__(self, meta: headmeta.CenterNet):
        super().__init__(meta.name)
        self.meta = meta
        self.annotation_painter = show.AnnotationPainter()

    def targets(self, field, *, annotation_dicts, metas):
        if len(self.indices('confidence'))==0 and len(self.indices('regression'))==0:
            return
        if isinstance(field, torch.Tensor):
            field = field.cpu().numpy()
        annotations = [
            AnnotationDet(self.meta.categories).set(ann['category_id'], None, ann['bbox'])
            for ann in annotation_dicts
        ]

        #self._confidences(field[0], annotations, metas)
        self._regressions(field[1], field[2], field[4], confidence_fields=field[0], annotations=annotations, metas=metas)

    def predicted(self, field):
        if len(self.indices('confidence'))==0 and len(self.indices('regression'))==0:
            return
        if isinstance(field, torch.Tensor):
            field = field.cpu().numpy()
        self._confidences(field[:-4], None, None)
        self._regressions(field[-4:-2], field[-2:], None, confidence_fields=field[:-4], annotations=None,
                          uv_is_offset=True)

    def _confidences(self, confidences, annotations, metas):
        f_range = self.indices('confidence')
        if len(self.indices('confidence'))==1 and self.indices('confidence')[0] == -1:
            f_range = np.arange(confidences.shape[0])[np.nanmax(confidences, axis=(1,2))>0.2]
        if len(f_range) <=0:
            return
        for f in f_range:
            LOG.debug('%s', self.meta.categories[f])
            with self.image_canvas(self._processed_image, margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                ax.text(0, 0, '{}'.format(self.meta.categories[f]), fontsize=14, color='red')
                if not metas is None:
                    ax.text(200, 0, '{}'.format(metas['image_id']), fontsize=14, color='red')
                im = ax.imshow(self.scale_scalar(confidences[f], self.meta.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap=CMAP_GREENS_NAN)
                self.colorbar(ax, im)
    def _build_map(self, data_pts, indices, wh):
        output_map = np.full((data_pts.shape[1], wh[1], wh[2]), np.nan, dtype=np.float32)
        x_indices = indices%wh[2]
        y_indices = (indices - x_indices)/wh[2]
        output_map[:, y_indices.type(torch.int32), x_indices.type(torch.int32)] = data_pts.T

        return torch.from_numpy(output_map)

    def _regressions(self, regression_fields, wh_fields, obj_indices=None,*,
                     annotations=None, confidence_fields=None, uv_is_offset=True, metas=None):

        f_range = self.indices('regression')
        if len(self.indices('regression'))==1 \
            and self.indices('regression')[0] == -1 \
            and not (confidence_fields is None):
            f_range = np.arange(confidence_fields.shape[0])[np.nanmax(confidence_fields, axis=(1,2))>0.2]
        if len(f_range) <=0:
            return
        if not obj_indices is None:
            regression_map = self._build_map(regression_fields, obj_indices, confidence_fields.shape)
            wh_map = self._build_map(wh_fields, obj_indices, confidence_fields.shape)
        else:
            regression_map = regression_fields
            wh_map = wh_fields
        #for f in f_range:
        #LOG.debug('%s', self.meta.categories[f])
        if isinstance(confidence_fields, np.ndarray):
            confidence_field = np.amax(confidence_fields, 0) if confidence_fields is not None else None
        else:
            confidence_field = confidence_fields.amax(0) if confidence_fields is not None else None


        with self.image_canvas(self._processed_image, margin=[0.0, 0.01, 0.05, 0.01]) as ax:
            show.white_screen(ax, alpha=0.5)
            if not metas is None:
                ax.text(200, 0, '{}'.format(metas['image_id']), fontsize=14, color='red')
            if annotations:
                self.annotation_painter.annotations(ax, annotations, color='lightgray')
            q = show.quiver(ax,
                            regression_map,
                            confidence_field=confidence_field,
                            xy_scale=self.meta.stride, threshold=0.2, uv_is_offset=uv_is_offset,
                            cmap='Greens', clim=(0.5, 1.0), width=0.001)
            show.boxes_wh(ax, wh_map[0], wh_map[1],
                          confidence_field=confidence_field, threshold=0.2,
                          regression_field=regression_map,
                          xy_scale=self.meta.stride, cmap='Greens',
                          fill=False, linewidth=2,
                          regression_field_is_offset=uv_is_offset)

            self.colorbar(ax, q)

class Raf(Base):

    def __init__(self, meta: headmeta.Raf_CN):
        super().__init__(meta.name)
        self.meta = meta
        self.annotation_painter = show.AnnotationPainter()

    def targets(self, field, *, annotation_dicts, metas):
        if len(self.indices('confidence'))==0 and len(self.indices('regression'))==0:
            return
        if isinstance(field, torch.Tensor):
            field = field.cpu().numpy()
        # annotations = [
        #     AnnotationDet(self.meta.categories).set(ann['category_id'], None, ann['bbox'])
        #     for ann in annotation_dicts
        # ]

        annotations = None
        self._confidences(field[0], annotations, metas)
        self._regressions(field[1], field[2], field[3], field[5], confidence_fields=field[0], annotations=annotations)

    def predicted(self, field):
        if len(self.indices('confidence'))==0 and len(self.indices('regression'))==0:
            return
        if isinstance(field, torch.Tensor):
            field = field.cpu().numpy()
        self._confidences(field[:-6], None, None)
        self._regressions(field[-6:-4], field[-4:-2], field[-2:], None, confidence_fields=field[:-6], annotations=None,
                          uv_is_offset=True)

    def _confidences(self, confidences, annotations, metas):
        f_range = self.indices('confidence')

        if len(self.indices('confidence'))==1 and self.indices('confidence')[0] == -1:
            f_range = np.arange(confidences.shape[0])[np.nanmax(confidences, axis=(1,2))>0.2]
        if len(f_range) <=0:
            return
        for f in f_range:
            LOG.debug('%s', self.meta.rel_categories[f])
            with self.image_canvas(self._processed_image, margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                ax.text(0, 0, '{}'.format(self.meta.rel_categories[f]), fontsize=14, color='red')
                im = ax.imshow(self.scale_scalar(confidences[f], self.meta.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap=CMAP_GREENS_NAN)
                self.colorbar(ax, im)
    def _build_map(self, data_pts, indices, wh):
        output_map = np.full((data_pts.shape[1], wh[1], wh[2]), np.nan, dtype=np.float32)
        x_indices = indices%wh[2]
        y_indices = (indices - x_indices)/wh[2]
        output_map[:, y_indices.type(torch.int32), x_indices.type(torch.int32)] = data_pts.T

        return torch.from_numpy(output_map)

    def _regressions(self, regression_fields, wh_fields, scales, obj_indices,*,
                     annotations=None, confidence_fields=None, uv_is_offset=True):

        f_range = self.indices('regression')
        if len(self.indices('regression'))==1 \
            and self.indices('regression')[0] == -1 \
            and not (confidence_fields is None):
            f_range = np.arange(confidence_fields.shape[0])[np.nanmax(confidence_fields, axis=(1,2))>0.2]

        if len(f_range) <=0:
            return
        if not obj_indices is None:
            regression_mapSub = self._build_map(regression_fields, obj_indices, confidence_fields.shape)
            regression_mapObj = self._build_map(wh_fields, obj_indices, confidence_fields.shape)
        else:
            regression_mapSub = regression_fields
            regression_mapObj = wh_fields

        if isinstance(confidence_fields, np.ndarray):
            confidence_field = np.amax(confidence_fields, 0) if confidence_fields is not None else None
        else:
            confidence_field = confidence_fields.amax(0) if confidence_fields is not None else None

        with self.image_canvas(self._processed_image, margin=[0.0, 0.01, 0.05, 0.01]) as ax:
            show.white_screen(ax, alpha=0.5)
            if annotations:
                self.annotation_painter.annotations(ax, annotations, color='lightgray')
            q = show.quiver(ax,
                            regression_mapSub,
                            confidence_field=confidence_field,
                            xy_scale=self.meta.stride, threshold=0.2, uv_is_offset=uv_is_offset,
                            cmap='Reds', clim=(0.5, 1.0), width=0.001)
            show.quiver(ax,
                        regression_mapObj,
                        confidence_field=confidence_field,
                        xy_scale=self.meta.stride, threshold=0.2, uv_is_offset=uv_is_offset,
                        cmap='Greens', clim=(0.5, 1.0), width=0.001)


            self.colorbar(ax, q)

class Prior(Base):

    def __init__(self, meta: headmeta.CenterNet):
        super().__init__(meta.name)
        self.meta = meta
        self.annotation_painter = show.AnnotationPainter()

    def targets(self, field, *, annotation_dicts, metas):
        if len(self.indices('confidence'))==0 and len(self.indices('regression'))==0:
            return
        if isinstance(field, torch.Tensor):
            field = field.cpu().numpy()
        annotations = [
            AnnotationDet(self.meta.categories).set(ann['category_id'], None, ann['bbox'])
            for ann in annotation_dicts
        ]

        self._confidences(field, annotations, metas)

    def predicted(self, field):
        if len(self.indices('confidence'))==0 and len(self.indices('regression'))==0:
            return
        if isinstance(field, torch.Tensor):
            field = field.cpu().numpy()
        self._confidences(field, None, None)

    def _confidences(self, confidences, annotations, metas):
        f_range = self.indices('confidence')

        if len(self.indices('confidence'))==1 and self.indices('confidence')[0] == -1:
            f_range = np.arange(confidences.shape[0]//2)[np.nanmax(confidences[::2], axis=(1,2))>0.2]
        if len(f_range) <=0:
            return
        for f in f_range:
            #LOG.debug('%s', self.meta.categories[f])

            with self.image_canvas(self._processed_image, margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                ax.text(0, 0, 'INP_HM_{}'.format(f), fontsize=14, color='red')
                im = ax.imshow(self.scale_scalar(confidences[2*f], self.meta.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap=CMAP_GREENS_NAN)
                self.colorbar(ax, im)

            with self.image_canvas(self._processed_image, margin=[0.0, 0.01, 0.05, 0.01]) as ax:
                ax.text(0, 0, 'INP_HM_{}'.format(f), fontsize=14, color='red')
                im = ax.imshow(self.scale_scalar(confidences[2*f+1], self.meta.stride),
                               alpha=0.9, vmin=0.0, vmax=1.0, cmap=CMAP_GREENS_NAN)
                self.colorbar(ax, im)
