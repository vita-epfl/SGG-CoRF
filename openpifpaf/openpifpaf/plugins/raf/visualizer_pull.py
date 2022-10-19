import logging
import os
import numpy as np

from openpifpaf.visualizer import Base
from openpifpaf.annotation import AnnotationDet
from .annotation import AnnotationRaf_updated as AnnotationRaf
from openpifpaf import show
from . import headmeta
from .painters_updated import RelationPainter

LOG = logging.getLogger(__name__)

import colorsys

try:
    import matplotlib
    import matplotlib.animation
    import matplotlib.collections
    import matplotlib.patches
except ImportError:
    matplotlib = None

class Raf(Base):

    def __init__(self, meta: headmeta.Raf):
        super().__init__(meta.name)
        self.meta = meta
        self.detection_painter = RelationPainter(eval=True)#RelationPainter(xy_scale=meta.stride)

    def targets(self, field, *, annotation_dicts, metas):
        if len(self.indices('confidence'))==0 and len(self.indices('regression'))==0:
            return
        annotations = []

        dict_id2index = {}
        for det_index, ann in enumerate(annotation_dicts):
            dict_id2index[ann['detection_id']] = det_index

        for s_idx, ann in enumerate(annotation_dicts):
            if ann['iscrowd'] or not np.any(ann['bbox']) or not len(ann['object_index']) > 0:
                continue
            bbox = ann['bbox']
            category_id = ann['category_id']
            for object_id, predicate in zip(ann['object_index'], ann['predicate']):
                if not(object_id in dict_id2index) or annotation_dicts[dict_id2index[object_id]]['iscrowd']:
                    continue
                object_idx = dict_id2index[object_id]
                bbox_object = annotation_dicts[object_idx]['bbox']
                object_category = annotation_dicts[object_idx]['category_id']
                annotations.append(AnnotationRaf(self.meta.obj_categories,
                                    self.meta.rel_categories).set(
                                    AnnotationDet(self.meta.obj_categories).set(object_category, 1, bbox_object),
                                    AnnotationDet(self.meta.obj_categories).set(ann['category_id'], 1, ann['bbox']),
                                    predicate+1, 1, s_idx, object_idx,
                                    ))
        self._confidences(field[:, 0])
        self._regressions(field[:, 1:3], field[:, 3:5], field[:, 7], field[:, 8], confidence_fields=field[:,0], annotations=annotations, metas=metas)

    def targets_offsets(self, field, *, annotation_dicts):
        if len(self.indices('confidence'))==0 and len(self.indices('regression'))==0:
            return
        self.targets(field[0], annotation_dicts=annotation_dicts)
        self._offsets(field[1][:, 0:2], field[1][:, 2:4], annotations=None)

    def _offsets(self, regression_fields1, regression_fields2, *, annotations=None, confidence_fields=None, uv_is_offset=True):

        if isinstance(confidence_fields, np.ndarray):
            confidence_field = np.amax(confidence_fields, 0) if confidence_fields is not None else None
        else:
            confidence_field = confidence_fields.amax(0) if confidence_fields is not None else None

        with self.image_canvas(self._processed_image) as ax:
            ax.text(0, 0, 'offsets_1', fontsize=14, color='red')
            show.white_screen(ax, alpha=0.5)
            if annotations:
                self.detection_painter.annotations(ax, annotations, color=('mediumblue', 'blueviolet', 'firebrick'))
            q1 = show.quiver(ax,
                             regression_fields1,
                             confidence_field=confidence_field,
                             xy_scale=self.meta.stride, uv_is_offset=uv_is_offset,
                             cmap='Reds', clim=(0.1, 0.5), width=0.001, threshold=0.2)
            show.quiver(ax,
                        regression_fields2,
                        confidence_field=confidence_field,
                        xy_scale=self.meta.stride, uv_is_offset=uv_is_offset,
                        cmap='Greens', clim=(0.1, 0.5), width=0.001, threshold=0.2)

            self.colorbar(ax, q1)

    def predicted(self, field, ground_truth=None, offset_1=None):
        self._confidences(field[:, 0])
        if field.shape[1]>=9:
            self._regressions(field[:, 1:3], field[:, 3:5], field[:, 7], field[:, 8],
                              annotations=self._ground_truth,
                              confidence_fields=field[:, 0],
                              uv_is_offset=False)
        else:
            self._regressions(field[:, 1:3], field[:, 3:5], field[:, 5], field[:, 6],
                              annotations=self._ground_truth,
                              confidence_fields=field[:, 0],
                              uv_is_offset=False)
        if not offset_1 is None:
            self._offsets(offset_1[0, 0:2], offset_1[0, 2:4], annotations=self._ground_truth, confidence_fields=field[:, 0])

    def _confidences(self, confidences):
        f_range = self.indices('confidence')

        if len(self.indices('confidence'))==1 and self.indices('confidence')[0] == -1:
            f_range = np.arange(confidences.shape[0])[np.nanmax(confidences, axis=(1,2))>0.2]
        if len(f_range) <=0:
            return
        for f in f_range:
            LOG.debug('%s', self.meta.rel_categories[f])

            with self.image_canvas(self._processed_image) as ax:
                ax.text(0, 0, '{}'.format(self.meta.rel_categories[f]), fontsize=8, color='red')
                im = ax.imshow(self.scale_scalar(confidences[f], self.meta.stride),
                               alpha=0.7, vmin=0.0, vmax=1.0, cmap='Blues')
                self.colorbar(ax, im)

    def _regressions(self, regression_fields1, regression_fields2,
                     scale_fields1, scale_fields2, *,
                     annotations=None, confidence_fields=None, uv_is_offset=True, metas=None):
        #indices = np.arange(confidence_fields.shape[0])[np.nanmax(confidence_fields, axis=(1,2))>0.2]
        #for f in indices:
        f_range = self.indices('regression')
        if len(self.indices('regression'))==1 \
            and self.indices('regression')[0] == -1 \
            and not (confidence_fields is None):
            f_range = np.arange(confidence_fields.shape[0])[np.nanmax(confidence_fields, axis=(1,2))>0.2]
        if len(f_range) <=0:
            return

        with self.image_canvas(self._processed_image) as ax:
            # if annotations:
            #     self.detection_painter.annotations(ax, annotations, color= None, metas=metas) # ('mediumblue', 'blueviolet', 'firebrick'),
            for f in f_range:
                LOG.debug('%s', self.meta.rel_categories[f])

                confidence_field = confidence_fields[f] if confidence_fields is not None else None


                #ax.text(0, 0, '{}'.format(self.meta.rel_categories[f]), fontsize=14, color='red')
                color_rel = matplotlib.cm.get_cmap('tab10')
                #color_rel = colorsys.rgb_to_hsv(color_rel[0], color_rel[1], color_rel[2])
                q1 = show.quiver_perrel(ax,
                                 regression_fields1[f, :2],
                                 confidence_field=confidence_fields[f],
                                 id_rel=((f % 20 + 0.05) / 20),
                                 xy_scale=self.meta.stride, uv_is_offset=uv_is_offset,
                                 cmap=color_rel, clim=(0.1, 0.5), width=0.001, threshold=0.2)
                show.quiver_perrel(ax,
                            regression_fields2[f, :2],
                            confidence_field=confidence_fields[f],
                            id_rel=((f % 20 + 0.05) / 20),
                            xy_scale=self.meta.stride, uv_is_offset=uv_is_offset,
                            cmap=color_rel, clim=(0.1, 0.5), width=0.001, threshold=0.2)
                # if scale_fields1 is not None and scale_fields2 is not None:
                #     show.boxes(ax, scale_fields1[f]/2.0,
                #                confidence_field=confidence_fields[f],
                #                regression_field=regression_fields1[f, :2],
                #                xy_scale=self.meta.stride, cmap='Reds', clim=(0.1, 0.5), fill=False,
                #                regression_field_is_offset=uv_is_offset, threshold=0.2)
                #     show.boxes(ax, scale_fields2[f]/2.0,
                #                confidence_field=confidence_fields[f],
                #                regression_field=regression_fields2[f, :2],
                #                xy_scale=self.meta.stride, cmap='Greens', clim=(0.1, 0.5), fill=False,
                #                regression_field_is_offset=uv_is_offset, threshold=0.2)

                self.colorbar(ax, q1)
