"""Predict poses for given images."""

import argparse
import glob
import json
import logging
import os

import PIL
import torch

from . import datasets, decoder, logger, network, plugin, show, transforms, visualizer, __version__
from ..transforms.toannotation import ToRafAnnotations

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
        head_metas[-1].fg_matrix = datamodule.head_metas[-1].fg_matrix
        head_metas[-1].smoothing_pred = datamodule.head_metas[-1].smoothing_pred
    processor = decoder.factory(head_metas)

    return processor, model


def preprocess_factory(args):
    rescale_t = None
    if args.long_edge:
        rescale_t = transforms.RescaleAbsolute(args.long_edge)

    pad_t = None
    if args.batch_size > 1 or args.long_edge:
        assert args.long_edge, '--long-edge must be provided for batch size > 1'
        pad_t = transforms.CenterPad(args.long_edge)
    else:
        pad_t = transforms.CenterPadTight(128)
        #pad_t = transforms.CenterPadTight(32)

    return transforms.Compose([
        transforms.NormalizeAnnotations(),
        rescale_t,
        pad_t,
        transforms.EVAL_TRANSFORM,
        openpifpaf.transforms.ToAnnotations([
            ToRafAnnotations(self.obj_categories, self.rel_categories),
        ]),
    ])


class Images_index():

    def __init__(self, images_chosen,data_dir=,*,
                 preprocess=None, split="train", num_im=-1, num_val_im=-1,
                 filter_duplicate_rels=True, filter_non_overlap=True, filter_empty_rels=True, use_512=False, eval_mode=False):
            super().__init__(data_dir,preprocess, split, num_im, num_val_im,
                         filter_duplicate_rels, filter_non_overlap, filter_empty_rels, use_512, eval_mode)
            self.images_chosen = np.asarray(images_chosen).astype(int)

    def __getitem__(self, index):
        index = self.image_ids.index(self.images_chosen[index])
        image = Image.fromarray(self._im_getter(index)); width, height = image.size

        image_id = self.image_ids[index]
        assert image_id == self.images_chosen[index]
        # get object bounding boxes, labels and relations
        obj_boxes = self.gt_boxes[index].copy()
        obj_labels = self.gt_classes[index].copy()
        obj_relation_triplets = self.relationships[index].copy()

        # if self.filter_duplicate_rels:
        #     # Filter out dupes!
        #     assert self.split == 'train'
        #     old_size = obj_relation_triplets.shape[0]
        #     all_rel_sets = defaultdict(list)
        #     for (o0, o1, r) in obj_relation_triplets:
        #         all_rel_sets[(o0, o1)].append(r)
        #     obj_relation_triplets = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
        #     obj_relation_triplets = np.array(obj_relation_triplets)

        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = obj_relation_triplets.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in obj_relation_triplets:
                all_rel_sets[(o0, o1, r)].append(1)
            obj_relation_triplets = [(k[0], k[1], k[2]) for k,v in all_rel_sets.items()]
            obj_relation_triplets = np.array(obj_relation_triplets)


        obj_relations = np.zeros((obj_boxes.shape[0], obj_boxes.shape[0]))

        for i in range(obj_relation_triplets.shape[0]):
            subj_id = obj_relation_triplets[i][0]
            obj_id = obj_relation_triplets[i][1]
            pred = obj_relation_triplets[i][2]
            obj_relations[subj_id, obj_id] = pred

        local_file_path = os.path.join(self.data_dir, "VG_100K", str(image_id)+".jpg")
        meta = {
            'dataset_index': index,
            'image_id': image_id,
            'file_name': image_id,
            'local_file_path': local_file_path,
        }

        anns = []
        dict_counter = {}
        for target in obj_relation_triplets:
            subj_id = target[0]
            obj_id = target[1]
            pred = target[2]
            x, y, x2, y2 = obj_boxes[subj_id]
            w, h = x2-x, y2-y

            if subj_id not in dict_counter:
                dict_counter[subj_id] = len(anns)
                anns.append({
                    'id': subj_id,
                    'detection_id': len(anns),
                    'image_id': image_id,
                    'category_id': int(obj_labels[subj_id]),
                    'bbox': [x, y, w, h],
                    "area": w*h,
                    "iscrowd": 0,
                    "keypoints":[x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2],
                    "segmentation":[],
                    'num_keypoints': 5,
                    'object_index': [len(anns) + 1] if obj_id not in dict_counter else [int(dict_counter[obj_id])],
                    'predicate': [pred-1],
                })
            else:
                if obj_id in dict_counter:
                    anns[dict_counter[subj_id]]['object_index'].append(dict_counter[obj_id])
                else:
                    anns[dict_counter[subj_id]]['object_index'].append(len(anns))
                anns[dict_counter[subj_id]]['predicate'].append(pred-1)

            x, y, x2, y2 = obj_boxes[obj_id]
            w, h = x2-x, y2-y

            if obj_id not in dict_counter:
                dict_counter[obj_id] = len(anns)
                anns.append({
                    'id': obj_id,
                    'detection_id': len(anns),
                    'image_id': image_id,
                    'category_id': int(obj_labels[obj_id]),
                    'bbox': [x, y, w, h],
                    "area": w*h,
                    "iscrowd": 0,
                    "keypoints":[x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2],
                    "segmentation":[],
                    'num_keypoints': 5,
                    'object_index': [],
                    'predicate': [],
                })
        for idx, det in enumerate(zip(obj_boxes, obj_labels)):
            if idx in dict_counter:
                continue
            x, y, x2, y2 = det[0]
            w, h = x2-x, y2-y
            dict_counter[idx] = len(anns)
            anns.append({
                    'id': idx,
                    'detection_id': len(anns),
                    'image_id': image_id,
                    'category_id': int(det[1]),
                    'bbox': [x, y, w, h],
                    "area": w*h,
                    "iscrowd": 0,
                    "keypoints":[x, y, 2, x+w, y, 2, x+w, y+h, 2, x, y+h, 2, x+w/2, y+h/2, 2],
                    "segmentation":[],
                    'num_keypoints': 5,
                    'object_index': [],
                    'predicate': [],
                    })

        assert len(anns) == len(obj_boxes)
        if self.eval_mode:
            anns_gt = copy.deepcopy(anns)
            image, anns, meta = self.preprocess(image, anns, meta)
        else:
            # preprocess image and annotations
            image, anns, meta = self.preprocess(image, anns, meta)

        # transform image

        # mask valid
        valid_area = meta['valid_area']
        utils.mask_valid_area(image, valid_area)

        # if there are not target transforms, done here
        LOG.debug(meta)
        if self.eval_mode:
            return image, (anns, anns_gt), meta
        return image, anns, meta

    def __len__(self):
        return len(self.images_chosen)

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


def main():
    args = cli()

    processor, model = processor_factory(args)
    preprocess = preprocess_factory(args)

    # data
    data = Images_index(args.images, preprocess=preprocess)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False,
        pin_memory=args.pin_memory, num_workers=args.loader_workers,
        collate_fn=datasets.collate_images_anns_meta)

    # visualizers
    annotation_painter = show.AnnotationPainter()

    for batch_i, (image_tensors_batch, _, meta_batch) in enumerate(data_loader):
        pred_batch = processor.batch(model, image_tensors_batch, device=args.device)

        # unbatch
        for pred, meta in zip(pred_batch, meta_batch):
            LOG.info('batch %d: %s', batch_i, meta['file_name'])

            if len(pred)>0 and isinstance(pred[0], list):
                pred, _ = pred
            pred = [ann.inverse_transform(meta) for ann in pred]

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

            # image output
            if args.show or args.image_output is not None:
                ext = show.Canvas.out_file_extension
                image_out_name = out_name(
                    args.image_output, meta['file_name'], '.predictions.' + ext)
                LOG.debug('image output = %s', image_out_name)
                with show.image_canvas(cpu_image, image_out_name) as ax:
                    annotation_painter.annotations(ax, pred)


if __name__ == '__main__':
    main()
