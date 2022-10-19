import copy
import logging

import numpy as np
import PIL

from .annotation import AnnotationRaf

from openpifpaf.transforms import Preprocess, Crop
from openpifpaf.transforms.hflip import _HorizontalSwap

class ToRafAnnotations_old:
    def __init__(self, obj_categories, rel_categories):
        self.obj_categories = obj_categories
        self.rel_categories = rel_categories

    def __call__(self, anns):
        annotations = []
        for ann in anns:
            if ann['iscrowd'] or not np.any(ann['bbox']) or not len(ann['object_index']) > 0:
                continue
            bbox = ann['bbox']
            category_id = ann['category_id']
            for object_id, predicate in zip(ann['object_index'], ann['predicate']):
                if anns[object_id]['iscrowd']:
                    continue
                bbox_object = anns[object_id]['bbox']
                object_category = anns[object_id]['category_id']
                annotations.append(AnnotationRaf(self.obj_categories,
                                    self.rel_categories).set(
                                    object_category, category_id,
                                    predicate+1, None, None, None, bbox, bbox_object
                                    ))

        return annotations

class ToRafAnnotations:
    def __init__(self, obj_categories, rel_categories):
        self.obj_categories = obj_categories
        self.rel_categories = rel_categories

    def __call__(self, anns):
        annotations = []
        for ann in anns:
            if ann['iscrowd'] or not np.any(ann['bbox']) or not len(ann['object_index']) > 0:
                continue
            bbox = ann['bbox']
            category_id = ann['category_id']
            for object_id, predicate in zip(ann['object_index'], ann['predicate']):
                if anns[object_id]['iscrowd']:
                    continue
                ann_subj = AnnotationDet(self.obj_categories).set(category_id, 1.0, bbox)
                bbox_object = anns[object_id]['bbox']
                object_category = anns[object_id]['category_id']
                ann_obj = AnnotationDet(self.obj_categories).set(object_category, 1.0, bbox_object)
                annotations.append(AnnotationRaf(self.obj_categories,
                                    self.rel_categories).set(
                                    obj=ann_obj, subj=ann_subj,
                                    category_id_rel=predicate+1,
                                    score_rel=1.0, idx_subj=None, idx_obj=None))

        return annotations

class Raf_HFlip(Preprocess):
    def __init__(self, keypoints, hflip, raf_categ, raf_hflip):
        self.swap = _HorizontalSwap(keypoints, hflip)
        self.raf_hflip = raf_hflip
        self.raf_categ = raf_categ

    def __call__(self, image, anns, meta):
        meta = copy.deepcopy(meta)
        anns = copy.deepcopy(anns)

        w, _ = image.size
        image = image.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        for ann in anns:
            ann['keypoints'][:, 0] = -ann['keypoints'][:, 0] - 1.0 + w
            if self.swap is not None and not ann['iscrowd']:
                ann['keypoints'] = self.swap(ann['keypoints'])
                meta['horizontal_swap'] = self.swap
            ann['bbox'][0] = -(ann['bbox'][0] + ann['bbox'][2]) - 1.0 + w
            for ind, pred in enumerate(ann['predicate']):
                source_name = self.raf_categ[pred]
                target_name = self.raf_hflip.get(source_name)
                if target_name:
                    ann['predicate'][ind] = self.raf_categ.index(target_name)

        assert meta['hflip'] is False
        meta['hflip'] = True

        meta['valid_area'][0] = -(meta['valid_area'][0] + meta['valid_area'][2]) + w

        return image, anns, meta
