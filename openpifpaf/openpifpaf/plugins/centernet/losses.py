import argparse
import logging

import torch

from . import heads
from .losses_util import RegL1Loss, RegLoss, NormRegL1Loss, RegWeightedL1Loss, FocalLoss, GIOULoss, CIOULoss
LOG = logging.getLogger(__name__)

class CenterNetLoss(torch.nn.Module):
    prescale = 1.0
    crit_reg = None
    crit_wh = None

    def __init__(self, head_net: heads.CenterNetHead):
        super(CenterNetLoss, self).__init__()
        self.n_vectors = head_net.meta.n_vectors
        self.n_scales = head_net.meta.n_scales
        self.giou_loss = head_net.meta.giou_loss
        self.crit = FocalLoss()
        self.crit_giou = GIOULoss()
        text_iou = '{}.{}.giou'
        if self.ciou:
            self.crit_giou = CIOULoss()
            text_iou = '{}.{}.ciou'
        self.previous_losses = None
        LOG.debug('%s: n_vectors = %d, n_scales = %d',
                  head_net.meta.name, self.n_vectors, self.n_scales)

        self.field_names = (
            ['{}.{}.c'.format(head_net.meta.dataset, head_net.meta.name)]
            + ['{}.{}.vec{}'.format(head_net.meta.dataset, head_net.meta.name, i + 1)
               for i in range(self.n_vectors)]
            + ['{}.{}.scales{}'.format(head_net.meta.dataset, head_net.meta.name, i + 1)
               for i in range(self.n_scales)]
            + ([text_iou.format(head_net.meta.dataset, head_net.meta.name)] if (self.giou_loss or self.ciou) else [])
        )

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('CenterNet Loss')
        group.add_argument('--cn-loss-prescale', default=cls.prescale, type=float)
        group.add_argument('--cn-regression-loss', default='l1',
                           choices=['smoothl1', 'l1'],
                           help='type of regression loss')
        group.add_argument('--cn-wh-loss', default='same',
                          choices=['cat_spec', 'dense', 'norm', 'same'],
                          help='type of wh loss')
        group.add_argument('--cn-ciou-loss', default=False, action='store_true',
                          help='use ciou loss')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.prescale = args.loss_prescale

        if args.cn_regression_loss == 'smoothl1':
            cls.crit_reg = RegLoss()
        elif args.cn_regression_loss == 'l1':
            cls.crit_reg = RegL1Loss()
        elif args.cn_regression_loss is None:
            cls.crit_reg = None
        else:
            raise Exception('unknown regression loss type {}'.format(args.cn_regression_loss))

        if args.cn_wh_loss == 'same':
            cls.crit_wh = cls.crit_reg
        elif args.cn_wh_loss == 'norm':
            cls.crit_wh = NormRegL1Loss()
        elif args.cn_wh_loss == 'dense':
            cls.crit_wh = torch.nn.L1Loss(reduction='sum')
        elif args.cn_wh_loss == 'cat_spec':
            cls.crit_wh = RegWeightedL1Loss()
        else:
            raise Exception('unknown wh loss type {}'.format(args.cn_wh_loss))
        cls.ciou = args.cn_ciou_loss


    def forward(self, outputs, targets):
        hm_loss, wh_loss, off_loss = 0, 0, 0
        output_hm = outputs[:, :-4]
        output_reg = outputs[:, -4:-2]
        output_wh =  outputs[:, -2:]

        target_hm = targets[0]
        target_reg = targets[1]
        target_wh = targets[2]
        target_mask = targets[3]
        target_ind = targets[4]

        hm_loss = self.crit(output_hm, target_hm)/2.0

        wh_loss = self.crit_reg(
            output_wh, target_mask,
            target_ind, target_wh)/2.0

        off_loss = self.crit_reg(output_reg, target_mask,
                             target_ind, target_reg)/2.0

        all_losses = [hm_loss] + [wh_loss] + [off_loss]
        if self.giou_loss or self.ciou:
            giou_loss = self.crit_giou(
                output_wh, output_reg, target_mask,
                target_ind, target_wh, target_reg)/2.0
            all_losses += [giou_loss]

        if not all(torch.isfinite(l).item() if l is not None else True for l in all_losses):
            raise Exception('found a loss that is not finite: {}, prev: {}'
                            ''.format(all_losses, self.previous_losses))
        self.previous_losses = [float(l.item()) if l is not None else None for l in all_losses]

        return all_losses

class RafLoss(torch.nn.Module):
    prescale = 1.0
    crit_reg = None
    crit_wh = None

    def __init__(self, head_net: heads.CenterNetHead):
        super(RafLoss, self).__init__()
        self.n_vectors = head_net.meta.n_vectors
        self.n_scales = head_net.meta.n_scales

        self.crit = FocalLoss()
        self.previous_losses = None
        LOG.debug('%s: n_vectors = %d, n_scales = %d',
                  head_net.meta.name, self.n_vectors, self.n_scales)

        self.field_names = (
            ['{}.{}.c'.format(head_net.meta.dataset, head_net.meta.name)]
            + ['{}.{}.vec{}'.format(head_net.meta.dataset, head_net.meta.name, i + 1)
               for i in range(self.n_vectors)]
            + ['{}.{}.scales{}'.format(head_net.meta.dataset, head_net.meta.name, i + 1)
               for i in range(self.n_scales)]
        )

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('CenterNet Loss')
        group.add_argument('--rafcn-loss-prescale', default=cls.prescale, type=float)
        group.add_argument('--rafcn-regression-loss', default='l1',
                           choices=['smoothl1', 'l1'],
                           help='type of regression loss')
        group.add_argument('--rafcn-wh-loss', default='same',
                          choices=['cat_spec', 'dense', 'norm', 'same'],
                          help='type of wh loss')
    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.prescale = args.loss_prescale

        if args.rafcn_regression_loss == 'smoothl1':
            cls.crit_reg = RegLoss()
        elif args.rafcn_regression_loss == 'l1':
            cls.crit_reg = RegL1Loss()
        elif args.rafcn_regression_loss is None:
            cls.crit_reg = None
        else:
            raise Exception('unknown regression loss type {}'.format(args.rafcn_regression_loss))

        if args.rafcn_wh_loss == 'same':
            cls.crit_wh = cls.crit_reg
        elif args.rafcn_wh_loss == 'norm':
            cls.crit_wh = NormRegL1Loss()
        elif args.rafcn_wh_loss == 'dense':
            cls.crit_wh = torch.nn.L1Loss(reduction='sum')
        elif args.rafcn_wh_loss == 'cat_spec':
            cls.crit_wh = RegWeightedL1Loss()
        else:
            raise Exception('unknown wh loss type {}'.format(args.rafcn_wh_loss))


    def forward(self, outputs, targets):
        hm_loss, subj_loss, obj_loss, scale_loss = 0, 0, 0, 0
        output_hm = outputs[:, :-6]
        output_subj = outputs[:, -6:-4]
        output_obj =  outputs[:, -4:-2]
        output_scale =  outputs[:, -2:]

        target_hm = targets[0]
        target_subj = targets[1]
        target_obj = targets[2]
        target_scale = targets[3]
        target_mask = targets[4]
        target_ind = targets[5]

        hm_loss = self.crit(output_hm, target_hm)/2.0

        subj_loss = self.crit_reg(
            output_subj, target_mask,
            target_ind, target_subj)/2.0

        obj_loss = self.crit_reg(output_obj, target_mask,
                             target_ind, target_obj)/2.0

        scale_loss = self.crit_reg(output_scale, target_mask,
                             target_ind, target_scale)/2.0

        all_losses = [hm_loss] + [subj_loss] + [obj_loss] + [scale_loss]
        if not all(torch.isfinite(l).item() if l is not None else True for l in all_losses):
            raise Exception('found a loss that is not finite: {}, prev: {}'
                            ''.format(all_losses, self.previous_losses))
        self.previous_losses = [float(l.item()) if l is not None else None for l in all_losses]

        return all_losses
