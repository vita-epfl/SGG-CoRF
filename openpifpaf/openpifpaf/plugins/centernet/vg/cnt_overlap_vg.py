from collections import defaultdict
import copy
import logging
import os
import numpy as np
import h5py
import json
import torch

def bbox_overlaps(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.size(0)
    K = gt_boxes.size(0)

    gt_boxes_area = ((gt_boxes[:,2] - gt_boxes[:,0] + 1) *
                (gt_boxes[:,3] - gt_boxes[:,1] + 1)).view(1, K)

    anchors_area = ((anchors[:,2] - anchors[:,0] + 1) *
                (anchors[:,3] - anchors[:,1] + 1)).view(N, 1)

    boxes = anchors.view(N, 1, 4).expand(N, K, 4)
    query_boxes = gt_boxes.view(1, K, 4).expand(N, K, 4)

    iw = (torch.min(boxes[:,:,2], query_boxes[:,:,2]) -
        torch.max(boxes[:,:,0], query_boxes[:,:,0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:,:,3], query_boxes[:,:,3]) -
        torch.max(boxes[:,:,1], query_boxes[:,:,1]) + 1)
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps

def load_graphs(graphs_file, images_file, mode='train', num_im=-1, num_val_im=0, filter_empty_rels=True,
                filter_non_overlap=False, use_512=False):
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    :param graphs_file: HDF5
    :param mode: (train, val, or test)
    :param num_im: Number of images we want
    :param num_val_im: Number of validation images
    :param filter_empty_rels: (will be filtered otherwise.)
    :param filter_non_overlap: If training, filter images that dont overlap.
    :return: image_index: numpy array corresponding to the index of images we're using
             boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
             gt_classes: List where each element is a [num_gt] array of classes
             relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    if mode not in ('train', 'val', 'test'):
        raise ValueError('{} invalid'.format(mode))

    roi_h5 = h5py.File(graphs_file, 'r')
    im_h5 = h5py.File(images_file, 'r')

    data_split = roi_h5['split'][:]
    split = 2 if mode == 'test' else 0
    split_mask = data_split == split

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if mode == 'val':
            image_index = image_index[:num_val_im]
        elif mode == 'train':
            image_index = image_index[num_val_im:]


    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    if use_512:
        all_boxes = roi_h5['boxes_{}'.format(512)][:]  # will index later
    else:
        all_boxes = roi_h5['boxes_{}'.format(1024)][:]  # will index later
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    im_widths = im_h5["image_widths"][split_mask]
    im_heights = im_h5["image_heights"][split_mask]
    im_ids = im_h5["image_ids"][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    im_sizes = []
    image_index_valid = []
    boxes = []
    gt_classes = []
    relationships = []
    image_ids = []
    for i in range(len(image_index)):
        boxes_i = all_boxes[im_to_first_box[i]:im_to_last_box[i] + 1, :]
        gt_classes_i = all_labels[im_to_first_box[i]:im_to_last_box[i] + 1]

        if im_to_first_rel[i] >= 0:
            predicates = _relation_predicates[im_to_first_rel[i]:im_to_last_rel[i] + 1]
            obj_idx = _relations[im_to_first_rel[i]:im_to_last_rel[i] + 1] - im_to_first_box[i]
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert mode == 'train'
            inters = bbox_overlaps(torch.from_numpy(boxes_i).float(), torch.from_numpy(boxes_i).float()).numpy()
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue
        image_index_valid.append(image_index[i])
        im_sizes.append(np.array([im_widths[i], im_heights[i]]))
        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        relationships.append(rels)
        image_ids.append(im_ids[i])

    im_sizes = np.stack(im_sizes, 0)
    return split_mask, image_index_valid, im_sizes, boxes, gt_classes, relationships, image_ids


splits = ['train', 'test']
data_dir = "data/visual_genome/"
filter_empty_rels = True
roidb_file = os.path.join(data_dir, "VG-SGG.h5")
use_512 = True
if use_512:
    image_file = os.path.join(data_dir, "imdb_512.h5")
else:
    image_file = os.path.join(data_dir, "imdb_1024.h5")
# read in dataset from a h5 file and a dict (json) file

im_h5 = h5py.File(image_file, 'r')
info = json.load(open(os.path.join(data_dir, "VG-SGG-dicts.json"), 'r'))
im_refs = im_h5['images'] # image data reference
im_scale = im_refs.shape[2]

# add background class
info['label_to_idx']['__background__'] = 0
class_to_ind = info['label_to_idx']
ind_to_classes = sorted(class_to_ind, key=lambda k:
                       class_to_ind[k])
# cfg.ind_to_class = ind_to_classes

predicate_to_ind = info['predicate_to_idx']
predicate_to_ind['__background__'] = 0
ind_to_predicates = sorted(predicate_to_ind, key=lambda k:
                          predicate_to_ind[k])

stride = 4
for split in splits:

    filter_non_overlap = True and split == 'train'
    filter_duplicate_rels = True and split == 'train'

    split_mask, image_index, im_sizes, gt_boxes, gt_classes, relationships, image_ids = load_graphs(
        roidb_file, image_file,
        split, -1, num_val_im=-1,
        filter_empty_rels=filter_empty_rels,
        filter_non_overlap=filter_non_overlap and split == "train",
        use_512=use_512
    )
    cnt_relagnostic = 0
    cnt_rel_distinctSubjObj = 0
    cnt_rel_relnostic = 0
    total_relations = 0
    pair_distinct_rel = 0
    pair_multiple_samerel = 0
    print("{} split contains {} images".format(split, len(image_index)))
    max_obj_perimage = 0
    max_rel_perimage = 0
    for index, image_id in enumerate(image_ids):
        #image_id = image_ids[index]
        obj_boxes = gt_boxes[index].copy()
        obj_labels = gt_classes[index].copy()
        obj_relation_triplets = relationships[index].copy()
        max_obj_perimage = max(max_obj_perimage, len(obj_boxes))
        # Filter out dupes!
        old_size = obj_relation_triplets.shape[0]
        all_rel_sets = defaultdict(int)
        all_rel_sets_pred = defaultdict(int)
        for (o0, o1, r) in obj_relation_triplets:
            if (o0, o1, r) not in all_rel_sets_pred:
                all_rel_sets[(o0, o1)] += 1
            all_rel_sets_pred[(o0, o1, r)] += 1

        obj_relation_triplets = [(k[0], k[1], k[2]) for k,v in all_rel_sets_pred.items()]
        obj_relation_triplets = np.array(obj_relation_triplets)


        max_rel_perimage = max(max_rel_perimage, len(obj_relation_triplets))
        center_subj = (obj_boxes[obj_relation_triplets[:,0], :2] + obj_boxes[obj_relation_triplets[:,0], 2:])/2
        center_obj = (obj_boxes[obj_relation_triplets[:,1], :2] + obj_boxes[obj_relation_triplets[:,1], 2:])/2

        center_rel = (center_subj + center_obj)/2 // stride

        diff = center_rel[np.newaxis,:,:] - center_rel[:, np.newaxis,:]

        res = (diff==0)
        res = res[:,:, 0] * res[:,:, 1]
        triu_indx = np.triu_indices(center_rel.shape[0],1)
        res = res[triu_indx]
        mask_res = (res==1)
        indices_rel_x, indices_rel_y = triu_indx[0][mask_res], triu_indx[1][mask_res]
        same_subjObj = ((obj_relation_triplets[indices_rel_x, :2] - obj_relation_triplets[indices_rel_y, :2])==0)
        same_subjObj = same_subjObj[:, 0] * same_subjObj[:, 1]
        same_rel = (obj_relation_triplets[indices_rel_x ,2] - obj_relation_triplets[indices_rel_y,2])==0

        cnt_relagnostic += sum(mask_res)
        cnt_rel_distinctSubjObj += (sum(mask_res) - sum(same_subjObj))

        cnt_rel_relnostic += (sum(mask_res) - sum(same_rel))
        total_relations += len(obj_relation_triplets)

        pair_distinct_rel += sum(np.asarray(list(all_rel_sets.values()))>1)
        pair_multiple_samerel += sum(np.asarray(list(all_rel_sets_pred.values()))>1)

    relation_pairs = (total_relations*(total_relations-1))/2
    print("found {}/{} relation pair collisions in {} split ({:2f}%)".format(cnt_relagnostic, relation_pairs, split, (cnt_relagnostic/relation_pairs*100)))
    print("found {}/{} distinct relation pair collisions in {} split ({:2f}%)".format(cnt_rel_relnostic, relation_pairs, split, (cnt_rel_relnostic/relation_pairs)*100))
    print("found {}/{} relation collisions pair with distinct subj-obj pair in {} split ({:2f}%)".format(cnt_rel_distinctSubjObj, relation_pairs, split, (cnt_rel_distinctSubjObj/relation_pairs)*100))
    print("found {}/{} pairs have multiple distinct relations in {} split ({:2f}%)".format(pair_distinct_rel,total_relations, split, (pair_distinct_rel/total_relations)*100))
    print("found {}/{} pairs have multiple same relations in {} split ({:2f}%)".format( pair_multiple_samerel, total_relations, split, (pair_multiple_samerel/total_relations)*100))
    print("Max objects in image {} in {} split".format(max_obj_perimage, split))
    print("Max relationships in image {} in {} split".format(max_rel_perimage, split))
