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

    loader = datamodule.eval_loader()
    for batch_i, (image_tensors, anns_batch, meta_batch) in enumerate(loader):
        LOG.info('batch %d / %d, last loop: %.3fs, batches per second=%.1f',
                 batch_i, len(loader), time.time() - loop_start,
                 batch_i / max(1, (time.time() - total_start)))
        loop_start = time.time()

        pred_batch = processor.batch(model, image_tensors,
                                     device=args.device, gt_anns_batch=None)

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
                for i in range(len(gt_anns[0])):
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
            # if args.show_final_image:
            #     # show ground truth and predictions on original image
            #     gt_anns = [ann.inverse_transform(image_meta) for ann in gt_anns]
            #
            #     annotation_painter = show.AnnotationPainter()
            #     with open(image_meta['local_file_path'], 'rb') as f:
            #         cpu_image = PIL.Image.open(f).convert('RGB')
            #
            #     with show.image_canvas(cpu_image) as ax:
            #         if args.show_final_ground_truth:
            #             annotation_painter.annotations(ax, gt_anns, color='grey')
            #         annotation_painter.annotations(ax, pred)

    total_time = time.time() - total_start

    LOG.info(
        'time per image: decoder = %.0fms, nn = %.0fms, total = %.0fms',
        1000 * decoder_time / n_images,
        1000 * nn_time / n_images,
        1000 * total_time / n_images,
    )
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
