import argparse
from openpifpaf.network.losses import CompositeLoss, components
from .refinement_heads import RefinedCompositeField3

import torch

import logging

LOG = logging.getLogger(__name__)

class SmoothL1:
    r_smooth = 1.0

    def __init__(self, *, scale_required=True):
        self.scale = None
        self.scale_required = scale_required

    def __call__(self, x1, x2, _, t1, t2, weight=None):
        """L1 loss.

        Loss for a single two-dimensional vector (x1, x2)
        true (t1, t2) vector.
        """

        d = torch.sqrt((x1 - t1)**2 + (x2 - t2)**2)
        smooth_regime = d < self.r_smooth

        smooth_loss = (0.5 *d[smooth_regime] ** 2)/self.r_smooth
        linear_loss = d[smooth_regime == 0] - (0.5 * self.r_smooth)
        losses = torch.cat((smooth_loss, linear_loss))

        if weight is not None:
            losses = losses * weight

        self.scale = None
        return torch.sum(losses)

class DCNCompositeLoss(torch.nn.Module):
    prescale = 1.0
    regression_loss = components.Laplace()

    def __init__(self, head_net: RefinedCompositeField3):
        super().__init__()
        self.n_offsets = head_net.meta.n_offsets

        self.n_vectors = head_net.meta.n_vectors
        self.n_scales = head_net.meta.n_scales

        LOG.debug('%s: n_vectors = %d, n_scales = %d',
                  head_net.meta.name, self.n_vectors, self.n_scales)

        self.confidence_loss = components.Bce()
        #self.regression_loss = regression_loss or components.laplace_loss
        self.offset_loss = SmoothL1(scale_required=False)
        self.scale_losses = torch.nn.ModuleList([
            components.Scale() for _ in range(self.n_scales)
        ])
        self.field_names = (
            ['{}.{}.c'.format(head_net.meta.dataset, head_net.meta.name)]
            + ['{}.{}.vec{}'.format(head_net.meta.dataset, head_net.meta.name, i + 1)
               for i in range(self.n_vectors)]
            + ['{}.{}.scales{}'.format(head_net.meta.dataset, head_net.meta.name, i + 1)
               for i in range(self.n_scales)]
            + ['{}.{}.offsets{}'.format(head_net.meta.dataset, head_net.meta.name, i + 1)
                 for i in range(self.n_offsets)]
        )
        self.weights = None
        self.bce_blackout = None
        self.previous_losses = None

    def _offset_losses(self, x_regs, t_regs, *, weight=None):
        assert x_regs.shape[1] == self.n_offsets * 2
        assert t_regs.shape[1] == self.n_offsets * 2
        batch_size = t_regs.shape[0]

        reg_losses = []
        for i in range(self.n_offsets):
            reg_masks = torch.isnan(t_regs[:, i * 2]).bitwise_not_()
            if not torch.any(reg_masks):
                reg_losses.append(None)
                continue

            loss = self.offset_loss(
                torch.masked_select(x_regs[:, i * 2 + 0], reg_masks),
                torch.masked_select(x_regs[:, i * 2 + 1], reg_masks),
                None,
                torch.masked_select(t_regs[:, i * 2 + 0], reg_masks),
                torch.masked_select(t_regs[:, i * 2 + 1], reg_masks),
            )
            if weight is not None:
                loss = loss * weight[:, :, 0][reg_masks]
            reg_losses.append(loss.sum() / batch_size)

        return reg_losses

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('DCNComposite Loss')
        group.add_argument('--dcn-loss-prescale', default=cls.prescale, type=float,
                           help='Laplace width b for scale lossin DCNComposite')
        group.add_argument('--dcn-regression-loss', default='laplace',
                           choices=['smoothl1', 'l1', 'laplace'],
                           help='type of regression loss')

    @classmethod
    def configure(cls, args: argparse.Namespace):
        cls.prescale = args.loss_prescale

        if args.regression_loss == 'smoothl1':
            cls.regression_loss = components.SmoothL1()
        elif args.regression_loss == 'l1':
            cls.regression_loss = staticmethod(components.l1_loss)
        elif args.regression_loss == 'laplace':
            cls.regression_loss = components.Laplace()
        elif args.regression_loss is None:
            cls.regression_loss = components.Laplace()
        else:
            raise Exception('unknown regression loss type {}'.format(args.regression_loss))

    def _confidence_loss(self, x_confidence, t_confidence):
        # TODO assumes one confidence
        x_confidence = x_confidence[:, :, 0]
        t_confidence = t_confidence[:, :, 0]

        bce_masks = torch.isnan(t_confidence).bitwise_not_()
        if not torch.any(bce_masks):
            return None

        batch_size = x_confidence.shape[0]
        LOG.debug('batch size = %d', batch_size)

        if self.bce_blackout:
            x_confidence = x_confidence[:, self.bce_blackout]
            bce_masks = bce_masks[:, self.bce_blackout]
            t_confidence = t_confidence[:, self.bce_blackout]

        LOG.debug('BCE: x = %s, target = %s, mask = %s',
                  x_confidence.shape, t_confidence.shape, bce_masks.shape)
        bce_target = torch.masked_select(t_confidence, bce_masks)
        x_confidence = torch.masked_select(x_confidence, bce_masks)
        ce_loss = self.confidence_loss(x_confidence, bce_target)

        if self.prescale != 1.0:
            ce_loss = ce_loss * self.prescale
        if self.weights is not None:
            weight = torch.ones_like(t_confidence, requires_grad=False)
            weight[:] = self.weights
            weight = torch.masked_select(weight, bce_masks)
            ce_loss = ce_loss * weight
        ce_loss = ce_loss.sum() / batch_size

        return ce_loss

    def _localization_loss(self, x_regs, t_regs, *, weight=None):
        assert x_regs.shape[2] == self.n_vectors * 3
        assert t_regs.shape[2] == self.n_vectors * 3
        batch_size = t_regs.shape[0]

        reg_losses = []
        if self.weights is not None:
            weight = torch.ones_like(t_regs[:, :, 0], requires_grad=False)
            weight[:] = self.weights
        for i in range(self.n_vectors):
            reg_masks = torch.isnan(t_regs[:, :, i * 2]).bitwise_not_()
            loss = self.regression_loss(
                torch.masked_select(x_regs[:, :, i * 2 + 0], reg_masks),
                torch.masked_select(x_regs[:, :, i * 2 + 1], reg_masks),
                torch.masked_select(x_regs[:, :, self.n_vectors * 2 + i], reg_masks),
                torch.masked_select(t_regs[:, :, i * 2 + 0], reg_masks),
                torch.masked_select(t_regs[:, :, i * 2 + 1], reg_masks),
                torch.masked_select(t_regs[:, :, self.n_vectors * 2 + i], reg_masks),
            )
            if self.prescale != 1.0:
                loss = loss * self.prescale
            if self.weights is not None:
                loss = loss * torch.masked_select(weight, reg_masks)
            reg_losses.append(loss.sum() / batch_size)

        return reg_losses

    def _scale_losses(self, x_scales, t_scales, *, weight=None):
        assert x_scales.shape[2] == t_scales.shape[2] == len(self.scale_losses)

        batch_size = x_scales.shape[0]
        losses = []
        if self.weights is not None:
            weight = torch.ones_like(t_scales[:, :, 0], requires_grad=False)
            weight[:] = self.weights
        for i, sl in enumerate(self.scale_losses):
            mask = torch.isnan(t_scales[:, :, i]).bitwise_not_()
            loss = sl(
                torch.masked_select(x_scales[:, :, i], mask),
                torch.masked_select(t_scales[:, :, i], mask),
            )
            if self.prescale != 1.0:
                loss = loss * self.prescale
            if self.weights is not None:
                loss = loss * torch.masked_select(weight, mask)
            losses.append(loss.sum() / batch_size)

        return losses
    def forward(self, *args):
        LOG.debug('loss for %s', self.field_names)
        x, t = args
        if self.n_offsets>0:
            x, x_offset = x
            t, t_offset = t
        if t is None:
            return [None for _ in range(1 + self.n_vectors + self.n_scales)]
        assert x.shape[2] == 1 + self.n_vectors * 3 + self.n_scales
        assert t.shape[2] == 1 + self.n_vectors * 3 + self.n_scales
        # x = x.double()
        x_confidence = x[:, :, 0:1]
        x_regs = x[:, :, 1:1 + self.n_vectors * 3]
        # if self.n_offsets>0:
        #     x_offset = x[:, 0, 1 + self.n_vectors * 3:1 + self.n_vectors * 3]
        #     x_scales = x[:, :, 1 + self.n_vectors * 3:]
        # else:
        x_scales = x[:, :, 1 + self.n_vectors * 3:]

        # t = t.double()
        t_confidence = t[:, :, 0:1]
        t_regs = t[:, :, 1:1 + self.n_vectors * 3]
        # if self.n_offsets>0:
        #     t_offset = t[:,0, 1 + self.n_vectors * 3:1 + self.n_vectors * 3]
        #     t_scales = t[:, :, 1 + self.n_vectors * 3:]
        # else:
        t_scales = t[:, :, 1 + self.n_vectors * 3:]


        ce_loss = self._confidence_loss(x_confidence, t_confidence)
        reg_losses = self._localization_loss(x_regs, t_regs)
        if self.n_offsets>0:
            off_losses = self._offset_losses(x_offset, t_offset)
        scale_losses = self._scale_losses(x_scales, t_scales)

        all_losses = [ce_loss] + reg_losses + scale_losses
        if self.n_offsets>0:
            all_losses += off_losses
        if not all(torch.isfinite(l).item() if l is not None else True for l in all_losses):
            raise Exception('found a loss that is not finite: {}, prev: {}'
                            ''.format(all_losses, self.previous_losses))
        self.previous_losses = [float(l.item()) if l is not None else None for l in all_losses]

        return all_losses
