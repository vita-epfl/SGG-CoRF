"""Predict poses for given images."""

import argparse
import glob
import json
import logging
import os

import PIL
import cv2
import torch

import numpy as np

from . import datasets, decoder, logger, network, plugin, show, transforms, visualizer, __version__
from .plugins.raf.painters_updated import RelationPainter
from .plugins.centernet import VG, OBJ_CATEGORIES, REL_CATEGORIES, ToRafAnnotations

LOG = logging.getLogger(__name__)


# pylint: disable=too-many-statements
def cli():
    plugin.register()

    parser = argparse.ArgumentParser(
        prog='python3 -m openpifpaf.predict',
        usage='%(prog)s [options] images',
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--version', action='version',
                        version='OpenPifPaf {version}'.format(version=__version__))

    decoder.cli(parser)
    logger.cli(parser)
    network.Factory.cli(parser)
    show.cli(parser)
    visualizer.cli(parser)

    parser.add_argument('images', nargs='*',
                        help='input images')
    parser.add_argument('--glob',
                        help='glob expression for input images (for many images)')
    parser.add_argument('-o', '--image-output', default=None, nargs='?', const=True,
                        help='Whether to output an image, '
                             'with the option to specify the output path or directory')
    parser.add_argument('--json-output', default=None, nargs='?', const=True,
                        help='Whether to output a json file, '
                             'with the option to specify the output path or directory')
    parser.add_argument('--batch-size', default=1, type=int,
                        help='processing batch size')
    parser.add_argument('--long-edge', default=None, type=int,
                        help='rescale the long side of the image (aspect ratio maintained)')
    parser.add_argument('--loader-workers', default=None, type=int,
                        help='number of workers for data loading')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='disable CUDA')
    parser.add_argument('--dataset-matrix', default=None, type=str,
                        help='use dataset specific frequency priors')
    parser.add_argument('--show-final-image', default=False, action='store_true')
    parser.add_argument('--show-final-ground-truth', default=False, action='store_true')
    parser.add_argument('--use-gt-image', default=False, action='store_true')
    args = parser.parse_args()

    logger.configure(args, LOG)  # logger first

    # add args.device
    args.device = torch.device('cpu')
    args.pin_memory = False
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        args.pin_memory = True
    LOG.info('neural network device: %s (CUDA available: %s, count: %d)',
             args.device, torch.cuda.is_available(), torch.cuda.device_count())

    decoder.configure(args)
    network.Factory.configure(args)
    show.configure(args)
    visualizer.configure(args)

    if args.loader_workers is None:
        args.loader_workers = args.batch_size

    # glob
    if args.glob:
        args.images += glob.glob(args.glob)
    if not args.images:
        raise Exception("no image files given")

    return args


def processor_factory(args):
    # load model
    model_cpu, _ = network.Factory().factory()
    model = model_cpu.to(args.device)
    if not args.disable_cuda and torch.cuda.device_count() > 1:
        LOG.info('Using multiple GPUs: %d', torch.cuda.device_count())
        model = torch.nn.DataParallel(model)
        model.base_net = model_cpu.base_net
        model.head_nets = model_cpu.head_nets

    head_metas = [hn.meta for hn in model.head_nets]

    if not args.dataset_matrix is None:
        datamodule = datasets.factory(args.dataset_matrix)
        datamodule._get_fg_matrix()
        head_metas[-1].fg_matrix = datamodule.head_metas[1].fg_matrix
        head_metas[-1].smoothing_pred = datamodule.head_metas[1].smoothing_pred
    processor = decoder.factory(head_metas)

    return processor, model


def preprocess_factory(args, gt_included=False):
    rescale_t = None
    if args.long_edge:
        rescale_t = transforms.RescaleAbsolute(args.long_edge)

    pad_t = None
    if args.batch_size > 1 or args.long_edge:
        assert args.long_edge, '--long-edge must be provided for batch size > 1'
        pad_t = transforms.CenterPad(args.long_edge)
    else:
        #pad_t = transforms.CenterPadTight(128)
        #pad_t = transforms.CenterPadTight(32)
        pass
    if gt_included:
        return transforms.Compose([
            transforms.NormalizeAnnotations(),
            rescale_t,
            pad_t,
            transforms.EVAL_TRANSFORM,
            transforms.ToAnnotations([
                ToRafAnnotations(OBJ_CATEGORIES, REL_CATEGORIES),
            ]),
        ])
    return transforms.Compose([
        transforms.NormalizeAnnotations(),
        rescale_t,
        pad_t,
        transforms.EVAL_TRANSFORM,
    ])


def out_name(arg, in_name, default_extension):
    """Determine an output name from args, input name and extension.

    arg can be:
    - none: return none (e.g. show image but don't store it)
    - True: activate this output and determine a default name
    - string:
        - not a directory: use this as the output file name
        - is a directory: use directory name and input name to form an output
    """
    if arg is None:
        return None

    if arg is True:
        return in_name + default_extension

    if os.path.isdir(arg):
        return os.path.join(
            arg,
            os.path.basename(in_name)
        ) + default_extension

    return arg

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

class ImageList_CV(torch.utils.data.Dataset):
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                   dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, image_paths, preprocess=None):
        super().__init__()
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = cv2.imread(image_path)

        # height, width = image.shape[0], image.shape[1]
        #
        # inp_height = (height | 127) + 1
        # inp_width = (width | 127) + 1
        # c = np.array([width // 2, height // 2], dtype=np.float32)
        # s = np.array([inp_width, inp_height], dtype=np.float32)
        #
        # trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        # resized_image = cv2.resize(image, (width, height))
        # inp_image = cv2.warpAffine(
        #     resized_image, trans_input, (inp_width, inp_height),
        #     flags=cv2.INTER_LINEAR)
        #
        # inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)
        #
        # image = inp_image.transpose(2, 0, 1).reshape(3, inp_height, inp_width)

        image = PIL.Image.fromarray(image)
        width, height = image.size

        anns = []
        meta = {
            'dataset_index': index,
            'file_name': image_path,
            'local_file_path': image_path,
            'offset': np.array((0.0, 0.0)),
            'scale': np.array((1.0, 1.0)),
            'rotation': {'angle': 0.0, 'width': None, 'height': None},
            'valid_area': np.array((0.0, 0.0,  width - 1, height - 1)),
            'hflip': False,
            'width_height': np.array((width, height)),
        }

        # meta['offset'] = - affine_transform(meta['offset'], trans_input)
        # meta['valid_area'][:2] = np.maximum(0.0, affine_transform(meta['valid_area'][:2], trans_input))
        # meta['valid_area'][2:] = np.minimum(affine_transform(meta['valid_area'][2:], trans_input), [inp_width - 1, inp_height - 1])

        if self.preprocess:
            image, anns, meta = self.preprocess(image, anns, meta)

        return image, anns, meta

    def __len__(self):
        return len(self.image_paths)

def main():
    args = cli()

    processor, model = processor_factory(args)
    #preprocess = preprocess_factory(args)

    # data
    data = ImageList_CV(args.images, preprocess=None)
    try:
        first_image = int(args.images[0])
        preprocess = preprocess_factory(args, gt_included=True)
        data = VG(
            data_dir="data/visual_genome/",
            preprocess=preprocess,
            split='test',
            use_512=True,
            image_ids=args.images
        )
    except:
        preprocess = preprocess_factory(args)
        data = ImageList_CV(args.images, preprocess=preprocess)

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers,
        collate_fn=datasets.collate_images_anns_meta)

    # visualizers
    rel_painter = RelationPainter(eval=True)
    det_painter = show.AnnotationPainter()

    for batch_i, (image_tensors_batch, anns_batch, meta_batch) in enumerate(data_loader):
        pred_batch = processor.batch(model, image_tensors_batch, device=args.device)

        # unbatch
        for pred, gt_anns, meta in zip(pred_batch, anns_batch, meta_batch):
            LOG.info('batch %d: %s', batch_i, meta['file_name'])
            pred_det = None
            if pred and (isinstance(pred[0], list) or isinstance(pred[0], tuple)):
                if isinstance(pred[0], list):
                    for i in range(len(pred)):
                        pred[i] = [ann.inverse_transform(meta) for ann in pred[i]]
                elif isinstance(pred[0], tuple):
                    for i in range(len(pred)):
                        pred[i] = ([ann.inverse_transform(meta) for ann in pred[i][0]], [ann.inverse_transform(meta) for ann in pred[i][1]])
                if len(gt_anns) > 0:
                    for i in range(len(gt_anns)):
                        gt_anns[i] = [ann.inverse_transform(meta) for ann in gt_anns[i]]

            # load the original image if necessary
            cpu_image = None
            if args.debug or args.show or args.image_output is not None:
                with open(meta['file_name'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')
            visualizer.Base.image(cpu_image)

            # json output
            if args.json_output is not None:
                json_out_name = out_name(
                    args.json_output, meta['file_name'], '.predictions.json')
                LOG.debug('json output = %s', json_out_name)
                with open(json_out_name, 'w') as f:
                    pred.sort(key=lambda x: x.score, reverse=True)
                    json.dump([ann.json_data() for ann in pred], f)
                json_out_name = out_name(
                    args.json_output, meta['file_name'], '.predictions_dets.json')
                if pred_det:
                    with open(json_out_name, 'w') as f:
                        pred_det.sort(key=lambda x: x.score, reverse=True)
                        json.dump([ann.json_data() for ann in pred_det], f)


            # image output
            if args.show or args.image_output is not None:
                ext = show.Canvas.out_file_extension
                image_out_name = out_name(
                    args.image_output, meta['file_name'], '.predictions.' + ext)
                LOG.debug('image output = %s', image_out_name)
                with show.image_canvas(cpu_image, image_out_name) as ax:
                    det_painter.annotations(ax, pred)

            if isinstance(pred[0], tuple):
                if args.use_gt_image:
                    pred_rel = np.asarray(pred[1][0]) #[ann for ann in pred[0][0] if ann.score>0.2]
                    pred_det = np.asarray(pred[1][1])
                else:
                    pred_rel = np.asarray(pred[0][0]) #[ann for ann in pred[0][0] if ann.score>0.2]
                    pred_det = np.asarray(pred[0][1]) #[ann for ann in pred[0][1] if ann.score>0.2]
            else:
                pred_rel = np.asarray(pred[0]) #[ann for ann in pred[0][0] if ann.score>0.2]
                pred_det = np.asarray(pred[1])

            if args.show_final_image:
                interim_folder = ""
                # show ground truth and predictions on original image
                #gt_anns = [ann.inverse_transform(meta) for ann in gt_anns]

                #annotation_painter = show.AnnotationPainter()
                # pred_rel = np.asarray(pred[0][0]) #[ann for ann in pred[0][0] if ann.score>0.2]
                # pred_det = np.asarray(pred[0][1]) #[ann for ann in pred[0][1] if ann.score>0.2]
                # res, acc = get_correct_matches(pred[0][1], pred[0][0], gt_anns[1])
                # dict_acc[meta['image_id']] = acc
                with open(meta['local_file_path'], 'rb') as f:
                    cpu_image = PIL.Image.open(f).convert('RGB')

                # if len(pred_rel)>0:
                #     fig_file = os.path.join('all-images', interim_folder, str(meta['file_name'])+'_corr_rel.jpg')
                #     with show.image_canvas(cpu_image, fig_file=fig_file) as ax:
                #         rel_painter.annotations(ax, pred_rel[res], metas=meta, gt="_corr", interim_folder=interim_folder+"/")

                # pred_rel = [ann for ann in pred_rel if ann.score>0.2]
                # pred_det = [ann for ann in pred_det if ann.score>0.2]

                fig_file = os.path.join('all-images', interim_folder, str(meta['file_name'])+'_rel.jpg')
                with show.image_canvas(cpu_image, fig_file=fig_file) as ax:
                    # if args.show_final_ground_truth:
                    #     rel_painter.annotations(ax, gt_anns[0][0])
                    rel_painter.annotations(ax, pred_rel, metas=meta, interim_folder=interim_folder+"/")

                if args.show_final_ground_truth:
                    fig_file = os.path.join('all-images', interim_folder, str(meta['file_name'])+'_gt_rel.jpg')
                    with show.image_canvas(cpu_image, fig_file=fig_file) as ax:
                        rel_painter.annotations(ax, gt_anns[0], metas=meta, gt="_gt", interim_folder=interim_folder+"/")
                fig_file = os.path.join('all-images', interim_folder, str(meta['file_name'])+'_det.jpg')

                with show.image_canvas(cpu_image, fig_file=fig_file) as ax:
                    # if args.show_final_ground_truth:
                    #     det_painter.annotations(ax, gt_anns[0][1])
                    det_painter.annotations(ax, pred_det)
                if args.show_final_ground_truth:
                    fig_file = os.path.join('all-images', interim_folder, str(meta['file_name'])+'_gt_det.jpg')
                    with show.image_canvas(cpu_image, fig_file=fig_file) as ax:
                        det_painter.annotations(ax, gt_anns[1])


if __name__ == '__main__':
    main()
