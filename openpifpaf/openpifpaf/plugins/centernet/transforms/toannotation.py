import copy
import logging

import numpy as np
import PIL

from ...raf.annotation import AnnotationRaf_updated as AnnotationRaf
from openpifpaf.annotation import AnnotationDet

# class ToRafAnnotations:
#     def __init__(self, obj_categories, rel_categories):
#         self.obj_categories = obj_categories
#         self.rel_categories = rel_categories
#
#     def __call__(self, anns):
#         annotations = []
#         annotations_det = []
#         dict_id2idx = {}
#         for ann in anns:
#             if ann['iscrowd']:
#                 continue
#             bbox = ann['bbox']
#             category_id = ann['category_id']
#             dict_id2idx[ann['detection_id']] = len(annotations_det)
#             annotations_det.append(AnnotationDet(self.obj_categories).set(category_id, 1.0, bbox))
#
#         for ann in anns:
#             if ann['iscrowd'] or not np.any(ann['bbox']) or not len(ann['object_index']) > 0:
#                 continue
#             bbox = ann['bbox']
#             category_id = ann['category_id']
#             for object_id, predicate in zip(ann['object_index'], ann['predicate']):
#                 if anns[object_id]['iscrowd']:
#                     continue
#                 idx_subj = dict_id2idx[ann['detection_id']]
#                 ann_subj = annotations_det[idx_subj]
#                 idx_obj = dict_id2idx[object_id]
#                 ann_obj = annotations_det[idx_obj]
#                 annotations.append(AnnotationRaf(self.obj_categories,
#                                     self.rel_categories).set(
#                                     obj=copy.deepcopy(ann_obj), subj=copy.deepcopy(ann_subj),
#                                     category_id_rel=predicate+1,
#                                     score_rel=1.0, idx_subj=idx_subj, idx_obj=idx_obj))
#         return (annotations, annotations_det)


class ToRafAnnotations:
    def __init__(self, obj_categories, rel_categories):
        self.obj_categories = obj_categories
        self.rel_categories = rel_categories

    def __call__(self, anns):
        annotations = []
        annotations_det = []
        dict_id2idx = {}
        mask_hm = None
        mask_raf = None
        if isinstance(anns, list) and len(anns)>0 and isinstance(anns[0], tuple):
            if len(anns[0]) == 2:
                anns, mask_hm = anns[0]
            else:
                anns, mask_hm, mask_raf = anns[0]

        for ann in anns:
            if ann['iscrowd']:
                continue
            bbox = ann['bbox']
            category_id = ann['category_id']
            dict_id2idx[ann['detection_id']] = len(annotations_det)
            annotations_det.append(AnnotationDet(self.obj_categories).set(category_id, 1.0, bbox))
        dict_counter = {}
        for ann in anns:
            if ann['iscrowd'] or not np.any(ann['bbox']) or not len(ann['object_index']) > 0:
                continue
            bbox = ann['bbox']
            category_id = ann['category_id']
            for object_id, predicate in zip(ann['object_index'], ann['predicate']):
                if anns[object_id]['iscrowd']:
                    continue
                idx_subj = dict_id2idx[ann['detection_id']]
                ann_subj = annotations_det[idx_subj]
                idx_obj = dict_id2idx[object_id]
                ann_obj = annotations_det[idx_obj]
                if (ann['detection_id'], object_id, predicate) in dict_counter:
                    annotations[dict_counter[(ann['detection_id'], object_id, predicate)]].set(
                            obj=copy.deepcopy(ann_obj), subj=copy.deepcopy(ann_subj),
                            category_id_rel=predicate+1,
                            score_rel=1.0, idx_subj=idx_subj, idx_obj=idx_obj)
                    continue
                dict_counter[(ann['detection_id'], object_id, predicate)] = len(annotations)
                annotations.append(AnnotationRaf(self.obj_categories,
                              self.rel_categories).set(obj=copy.deepcopy(ann_obj), subj=copy.deepcopy(ann_subj),
                              category_id_rel=predicate+1,
                              score_rel=1.0, idx_subj=idx_subj, idx_obj=idx_obj))
        if not mask_hm is None:
            if not mask_raf is None:
                return  (annotations, annotations_det, mask_hm, mask_raf)
            return (annotations, annotations_det, mask_hm)
        return (annotations, annotations_det)
