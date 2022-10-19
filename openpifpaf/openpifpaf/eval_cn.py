"""Evaluation on COCO data."""

import argparse
import glob
import json
import logging
import os
import sys
import time
import numpy as np

import PIL
import thop
import torch

from . import datasets, decoder, logger, network, plugin, show, visualizer, __version__
from .plugins.raf.painters_updated import RelationPainter
from maskrcnn_benchmark.utils.miscellaneous import intersect_2d, argsort_desc, bbox_overlaps

LOG = logging.getLogger(__name__)


def default_output_name(args):
    output = '{}.eval-{}'.format(network.Factory.checkpoint, args.dataset)

    # coco
    if args.coco_eval_orientation_invariant or args.coco_eval_extended_scale:
        output += '-coco'
        if args.coco_eval_orientation_invariant:
            output += 'o'
        if args.coco_eval_extended_scale:
            output += 's'
    if args.coco_eval_long_edge is not None and args.coco_eval_long_edge != 641:
        output += '-cocoedge{}'.format(args.coco_eval_long_edge)

    # dense
    if args.dense_connections:
        output += '-dense'
        if args.dense_connections != 1.0:
            output += '{}'.format(args.dense_connections)

    return output

class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter,
                      argparse.RawDescriptionHelpFormatter):
    pass


def cli():  # pylint: disable=too-many-statements,too-many-branches
    plugin.register()

    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.eval',
        usage='%(prog)s [options]',
        description=__doc__,
        formatter_class=CustomFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    datasets.cli(parser)
    decoder.cli(parser)
    logger.cli(parser)
    network.Factory.cli(parser)
    show.cli(parser)
    visualizer.cli(parser)

    parser.add_argument('--output', default=None,
                        help='output filename without file extension')
    parser.add_argument('--skip-existing', default=False, action='store_true',
                        help='skip if output eval file exists already')
    parser.add_argument('--no-skip-epoch0', dest='skip_epoch0',
                        default=True, action='store_false',
                        help='do not skip eval for epoch 0')
    parser.add_argument('--watch', default=False, const=60, nargs='?', type=int)
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--write-predictions', default=False, action='store_true',
                        help='write a json and a zip file of the predictions')
    parser.add_argument('--show-final-image', default=False, action='store_true')
    parser.add_argument('--show-final-ground-truth', default=False, action='store_true')
    parser.add_argument('--flip-test', default=False, action='store_true')
    parser.add_argument('--run-metric', default=False, action='store_true')
    parser.add_argument('--use-gt-image', default=False, action='store_true')
    args = parser.parse_args()

    logger.configure(args, LOG)

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.debug('neural network device: %s', args.device)

    datasets.configure(args)
    decoder.configure(args)
    network.Factory.configure(args)
    show.configure(args)
    visualizer.configure(args)

    return args

def _triplet(relations, classes, boxes, predicate_scores=None, class_scores=None):
    """
    format relations of (sub_id, ob_id, pred_label) into triplets of (sub_label, pred_label, ob_label)
    Parameters:
        relations (#rel, 3) : (sub_id, ob_id, pred_label)
        classes (#objs, ) : class labels of objects
        boxes (#objs, 4)
        predicate_scores (#rel, ) : scores for each predicate
        class_scores (#objs, ) : scores for each object
    Returns:
        triplets (#rel, 3) : (sub_label, pred_label, ob_label)
        triplets_boxes (#rel, 8) array of boxes for the parts
        triplets_scores (#rel, 3) : (sub_score, pred_score, ob_score)
    """
    sub_id, ob_id, pred_label = relations[:, 0], relations[:, 1], relations[:, 2]
    triplets = np.column_stack((classes[sub_id], pred_label, classes[ob_id]))
    triplet_boxes = np.column_stack((boxes[sub_id], boxes[ob_id]))

    triplet_scores = None
    if predicate_scores is not None and class_scores is not None:
        triplet_scores = np.column_stack((
            class_scores[sub_id], predicate_scores, class_scores[ob_id],
        ))

    return triplets, triplet_boxes, triplet_scores

def _compute_pred_matches(gt_triplets, pred_triplets,
                 gt_boxes, pred_boxes, iou_thres, phrdet=False):
    """
    Given a set of predicted triplets, return the list of matching GT's for each of the
    given predictions
    Return:
        pred_to_gt [List of List]
    """
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = intersect_2d(gt_triplets, pred_triplets)
    gt_has_match = keeps.any(1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.where(gt_has_match)[0],
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match],
                                         ):
        boxes = pred_boxes[keep_inds]
        if phrdet:
            # Evaluate where the union box > 0.5
            gt_box_union = gt_box.reshape((2, 4))
            gt_box_union = np.concatenate((gt_box_union.min(0)[:2], gt_box_union.max(0)[2:]), 0)

            box_union = boxes.reshape((-1, 2, 4))
            box_union = np.concatenate((box_union.min(1)[:,:2], box_union.max(1)[:,2:]), 1)

            inds = bbox_overlaps(gt_box_union[None], box_union)[0] >= iou_thres

        else:
            sub_iou = bbox_overlaps(gt_box[None,:4], boxes[:, :4])[0]
            obj_iou = bbox_overlaps(gt_box[None,4:], boxes[:, 4:])[0]

            inds = (sub_iou >= iou_thres) & (obj_iou >= iou_thres)

        for i in np.where(keep_inds)[0][inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt

def get_correct_matches(predictions_det, predictions_rel, ground_truth, nogc=False):
    box = []
    bbox_xyxy = []
    score = []
    label = []

    ground_truth_indices = sorted(range(len(ground_truth)), key=lambda k: ground_truth[k]['id'])

    if ground_truth_indices:
        ground_truth = np.asarray(ground_truth)[ground_truth_indices]
        #predictions_det = np.asarray(predictions_det)[ground_truth_indices]

    for pred in predictions_det:
        box.append(pred.bbox)
        score.append(pred.score)
        label.append(pred.category_id)
        bbox_temp = np.copy(pred.bbox)
        bbox_temp[2:] = bbox_temp[:2] + bbox_temp[2:]
        bbox_xyxy.append(bbox_temp)
    bbox_xyxy = np.asarray(bbox_xyxy)
    score = np.asarray(score)
    label = np.asarray(label)

    gt_rels = []
    gt_dets_bbox = []
    gt_dets_classes = []
    for s_idx, pred in enumerate(ground_truth):
        bbox_temp = np.copy(pred['bbox'])
        bbox_temp[2:] += bbox_temp[:2]
        gt_dets_bbox.append(bbox_temp)
        gt_dets_classes.append(pred['category_id'])
        for rel_idx, rel in enumerate(pred['predicate']):
            o_idx = pred['object_index'][rel_idx]
            if ground_truth_indices:
                o_idx = ground_truth_indices.index(pred['object_index'][rel_idx])
            gt_rels.append([s_idx, o_idx, rel+1])
    gt_rels = np.asarray(gt_rels)
    gt_dets_bbox = np.asarray(gt_dets_bbox)
    gt_dets_classes = np.asarray(gt_dets_classes)

    rel_anns_idxs = []
    rel_anns_rels = []

    for pred in predictions_rel:
        s_idx = pred.idx_subj
        o_idx = pred.idx_obj
        # if ground_truth_indices:
        #     s_idx = ground_truth_indices.index(s_idx)
        #     o_idx = ground_truth_indices.index(o_idx)
        rel_anns_idxs.append([int(s_idx), int(o_idx)])
        rel_anns_rels.append(np.insert(pred.rel, 0, 0, axis=0))

    if len(predictions_rel) == 0:
        rel_anns_idxs.append([len(predictions_det)-1, len(predictions_det)-1])
        rel_temp = np.zeros(51)
        rel_temp[0] = 1
        rel_anns_rels.append(rel_temp)

    rel_anns_idxs = np.asarray(rel_anns_idxs)
    rel_anns_rels = np.asarray(rel_anns_rels)

    gt_triplets, gt_triplet_boxes, _ = _triplet(gt_rels, gt_dets_classes, gt_dets_bbox)

    if not nogc:
        pred_rels = np.column_stack((rel_anns_idxs, 1+rel_anns_rels[:,1:].argmax(1)))
        pred_scores = rel_anns_rels[:,1:].max(1)


    else:
        obj_scores_per_rel = score[rel_anns_idxs].prod(1)
        nogc_overall_scores = obj_scores_per_rel[:,None] * rel_anns_rels[:,1:]
        nogc_score_inds = argsort_desc(nogc_overall_scores)[:100]
        pred_rels = np.column_stack((rel_anns_idxs[nogc_score_inds[:,0]], nogc_score_inds[:,1]+1))
        pred_scores = rel_anns_rels[nogc_score_inds[:,0], nogc_score_inds[:,1]+1]

        # nogc_pred_triplets, nogc_pred_triplet_boxes, _ = _triplet(
        #         nogc_pred_rels, pred_classes, pred_boxes, nogc_pred_scores, obj_scores
        # )

    pred_triplets, pred_triplet_boxes, pred_triplet_scores = _triplet(
            pred_rels, label, bbox_xyxy, pred_scores, score)

    res = _compute_pred_matches(gt_triplets, pred_triplets, gt_triplet_boxes, pred_triplet_boxes, 0.5, phrdet=False,)

    res_bool = [True if len(res_single)>0 else False for res_single in res]
    acc = float(sum(res_bool))/len(gt_triplets)
    return [True if len(res_single)>0 else False for res_single in res], acc

def count_ops(model, height=640, width=640):
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, 3, height, width, device=device)
    gmacs, params = thop.profile(model, inputs=(dummy_input, ))
    LOG.info('GMACs = {0:.2f}, million params = {1:.2f}'.format(gmacs / 1e9, params / 1e6))
    return gmacs, params


# pylint: disable=too-many-statements,too-many-branches
def evaluate(args):
    # generate a default output filename
    if args.output is None:
        args.output = default_output_name(args)

    # skip existing?
    if args.skip_epoch0:
        if network.Factory.checkpoint.endswith('.epoch000'):
            print('Not evaluating epoch 0.')
            return
    if args.skip_existing:
        stats_file = args.output + '.stats.json'
        if os.path.exists(stats_file):
            print('Output file {} exists already. Exiting.'.format(stats_file))
            return
        print('{} not found. Processing: {}'.format(stats_file, network.Factory.checkpoint))

    datamodule = datasets.factory(args.dataset)
    model_cpu, _ = network.Factory().factory(head_metas=datamodule.head_metas)
    #model_cpu, _ = network.Factory().factory()
    model = model_cpu.to(args.device)
    if not args.disable_cuda and torch.cuda.device_count() > 1:
        LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        model.base_net = model_cpu.base_net
        model.head_nets = model_cpu.head_nets

    head_metas = [hn.meta for hn in model.head_nets]
    processor = decoder.factory(head_metas)
    # processor.instance_scorer = decocder.instance_scorer.InstanceScoreRecorder()
    # processor.instance_scorer = torch.load('instance_scorer.pkl')

    metrics = datamodule.metrics()
    total_start = time.time()
    loop_start = time.time()
    nn_time = 0.0
    decoder_time = 0.0
    n_images = 0

    # visualizers
    rel_painter = RelationPainter(eval=True)
    det_painter = show.AnnotationPainter()
    dict_acc = {}
    loader = datamodule.eval_loader()
    for batch_i, (image_tensors, anns_batch, meta_batch) in enumerate(loader):
        LOG.info('batch %d / %d, last loop: %.3fs, batches per second=%.1f',
                 batch_i, len(loader), time.time() - loop_start,
                 batch_i / max(1, (time.time() - total_start)))
        loop_start = time.time()

        pred_batch = processor.batch(model, image_tensors,
                                     device=args.device, gt_anns_batch=anns_batch)
        if args.flip_test:
            pred_batch_flipped = processor.batch(model,\
                                        [torch.flip(image_tensor, (3,)) for image_tensor in image_tensors] if isinstance(image_tensors, list) \
                                                    else torch.flip(image_tensors, (3,)),
                                         device=args.device, gt_anns_batch=anns_batch)

        if isinstance(image_tensors, list):
            image_tensors, _ = image_tensors
        n_images += len(image_tensors)
        decoder_time += processor.last_decoder_time
        nn_time += processor.last_nn_time

        # loop over batch
        assert len(image_tensors) == len(meta_batch)
        for pred_idx, (pred, gt_anns, image_meta) in enumerate(zip(pred_batch, anns_batch, meta_batch)):
            if pred and (isinstance(pred[0], list) or isinstance(pred[0], tuple)):
                if isinstance(pred[0], list):
                    for i in range(len(pred)):
                        pred[i] = [ann.inverse_transform(image_meta) for ann in pred[i]]
                elif isinstance(pred[0], tuple):
                    for i in range(len(pred)):
                        pred[i] = ([ann.inverse_transform(image_meta) for ann in pred[i][0]], [ann.inverse_transform(image_meta) for ann in pred[i][1]])
                for i in range(2):
                    gt_anns[0][i] = [ann.inverse_transform(image_meta) for ann in gt_anns[0][i]]

                # for idx, (gt_ann, pred_ann) in enumerate(zip(gt_anns[0][1], pred[1])):
                #     if not (np.all(np.equal(gt_ann.bbox, pred_ann.bbox)) and gt_ann.category_id == pred_ann.category_id) :
                #         import pdb; pdb.set_trace()
                # for idx, (gt_ann, pred_ann) in enumerate(zip(gt_anns[0][0], pred[0])):
                #     if not (gt_ann.idx_subj == pred_ann.idx_subj and gt_ann.idx_obj == pred_ann.idx_obj and np.all(np.equal(gt_ann.rel, pred_ann.rel))) :
                #         import pdb; pdb.set_trace()
                #
                # for idx, (original_gt, gt_ann) in enumerate(zip(gt_anns[1], gt_anns[0][1])):
                #     if not(np.all(np.equal(original_gt['bbox'], gt_ann.bbox)) and original_gt['category_id']==gt_ann.category_id):
                #         import pdb; pdb.set_trace()
            else:
                image_meta_flipped = dict(image_meta)
                pred = [ann.inverse_transform(image_meta) for ann in pred]
                if args.flip_test:
                    image_meta_flipped['hflip'] = True
                    pred.extend([ann.inverse_transform(image_meta_flipped) for ann in pred_batch_flipped[pred_idx]])
                    pred = decoder.utils.nms.Detection().annotations_per_category(pred, nms_type='snms')[:100]
                    #pred = decoder.utils.nms.Detection().annotations(pred)

            if args.run_metric:
                for metric in metrics:
                    if isinstance(pred[0], tuple):
                        if metric.mode == "sgdet":
                            metric.accumulate(pred[0], image_meta, ground_truth=gt_anns[1])
                        elif metric.mode == "sgcls":
                            metric.accumulate(pred[2], image_meta, ground_truth=gt_anns[1])
                        elif metric.mode == "predcls":
                            metric.accumulate(pred[1], image_meta, ground_truth=gt_anns[1])
                    else:
                        metric.accumulate(pred, image_meta, ground_truth=gt_anns[1])
            if isinstance(pred[0], tuple):
                if args.use_gt_image:
                    pred_rel = np.asarray(pred[1][0]) #[ann for ann in pred[0][0] if ann.score>0.2]
                    pred_det = np.asarray(pred[1][1])
                else:
                    pred_rel = np.asarray(pred[0][0]) #[ann for ann in pred[0][0] if ann.score>0.2]
                    pred_det = np.asarray(pred[0][1]) #[ann for ann in pred[0][1] if ann.score>0.2]
                res, acc = get_correct_matches(pred_det, pred_rel, gt_anns[1])
                #dict_acc[image_meta['image_id']] = acc
            if args.show_final_image:
                interim_folder = "nms0.6_car_det0.08"
                # show ground truth and predictions on original image
                #gt_anns = [ann.inverse_transform(image_meta) for ann in gt_anns]

                #annotation_painter = show.AnnotationPainter()
                # pred_rel = np.asarray(pred[0][0]) #[ann for ann in pred[0][0] if ann.score>0.2]
                # pred_det = np.asarray(pred[0][1]) #[ann for ann in pred[0][1] if ann.score>0.2]
                # res, acc = get_correct_matches(pred[0][1], pred[0][0], gt_anns[1])
                # dict_acc[image_meta['image_id']] = acc
                with open(image_meta['local_file_path'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')

                if len(pred_rel)>0:
                    fig_file = os.path.join('all-images', interim_folder, str(image_meta['file_name'])+'_corr_rel.jpg')
                    with show.image_canvas(cpu_image, fig_file=fig_file) as ax:
                        rel_painter.annotations(ax, pred_rel[res], metas=image_meta, gt="_corr", interim_folder=interim_folder+"/")

                # pred_rel = [ann for ann in pred_rel if ann.score>0.2]
                # pred_det = [ann for ann in pred_det if ann.score>0.2]

                fig_file = os.path.join('all-images', interim_folder, str(image_meta['file_name'])+'_rel.jpg')
                with show.image_canvas(cpu_image, fig_file=fig_file) as ax:
                    # if args.show_final_ground_truth:
                    #     rel_painter.annotations(ax, gt_anns[0][0])
                    rel_painter.annotations(ax, pred_rel, metas=image_meta, interim_folder=interim_folder+"/")

                if args.show_final_ground_truth:
                    fig_file = os.path.join('all-images', interim_folder, str(image_meta['file_name'])+'_gt_rel.jpg')
                    with show.image_canvas(cpu_image, fig_file=fig_file) as ax:
                        rel_painter.annotations(ax, gt_anns[0][0], metas=image_meta, gt="_gt", interim_folder=interim_folder+"/")
                fig_file = os.path.join('all-images', interim_folder, str(image_meta['file_name'])+'_det.jpg')

                with show.image_canvas(cpu_image, fig_file=fig_file) as ax:
                    # if args.show_final_ground_truth:
                    #     det_painter.annotations(ax, gt_anns[0][1])
                    det_painter.annotations(ax, pred_det)
                if args.show_final_ground_truth:
                    fig_file = os.path.join('all-images', interim_folder, str(image_meta['file_name'])+'_gt_det.jpg')
                    with show.image_canvas(cpu_image, fig_file=fig_file) as ax:
                        det_painter.annotations(ax, gt_anns[0][1])

    total_time = time.time() - total_start
    #dict_acc = {k: v for k, v in sorted(dict_acc.items(), key=lambda item: item[1])}
    #import pdb; pdb.set_trace()
    # processor.instance_scorer.write_data('instance_score_data.json')

    # model stats

    counted_ops = list(count_ops(model_cpu))
    local_checkpoint = network.local_checkpoint_path(network.Factory.checkpoint)

    file_size = os.path.getsize(local_checkpoint) if local_checkpoint else -1.0

    # write
    if args.run_metric:
        for metric in metrics:
            additional_data = {
                'args': sys.argv,
                'version': __version__,
                'dataset': args.dataset,
                'total_time': total_time,
                'checkpoint': network.Factory.checkpoint,
                'count_ops': counted_ops,
                'file_size': file_size,
                'n_images': n_images,
                'decoder_time': decoder_time,
                'nn_time': nn_time,
            }

            if args.write_predictions:
                metric.write_predictions(args.output, additional_data=additional_data)

            stats = dict(**metric.stats(), **additional_data)
            with open(args.output + '.stats.json', 'w') as f:
                json.dump(stats, f)

            LOG.info('stats:\n%s', json.dumps(stats, indent=4))
            LOG.info(
                'time per image: decoder = %.0fms, nn = %.0fms, total = %.0fms',
                1000 * stats['decoder_time'] / stats['n_images'],
                1000 * stats['nn_time'] / stats['n_images'],
                1000 * stats['total_time'] / stats['n_images'],
            )


def watch(args):
    assert args.output is None
    pattern = args.checkpoint
    evaluated_pattern = '{}*eval-{}.stats.json'.format(pattern, args.dataset)

    while True:
        # find checkpoints that have not been evaluated
        all_checkpoints = glob.glob(pattern)
        evaluated = glob.glob(evaluated_pattern)
        if args.skip_epoch0:
            all_checkpoints = [c for c in all_checkpoints
                               if not c.endswith('.epoch000')]
        checkpoints = [c for c in all_checkpoints
                       if not any(e.startswith(c) for e in evaluated)]
        LOG.info('%d checkpoints, %d evaluated, %d todo: %s',
                 len(all_checkpoints), len(evaluated), len(checkpoints), checkpoints)

        # evaluate all checkpoints
        for checkpoint in checkpoints:
            # reset
            args.output = None
            network.Factory.checkpoint = checkpoint

            evaluate(args)

        # wait before looking for more work
        time.sleep(args.watch)


def main():
    args = cli()

    if args.watch:
        watch(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
