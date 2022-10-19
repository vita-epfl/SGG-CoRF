import logging
import torch

from openpifpaf.network.neck_transformer import JointTransformer, SoloTransformer
LOG = logging.getLogger(__name__)

MODEL_MIGRATION = set()


class Shell(torch.nn.Module):
    def __init__(self, base_net, head_nets, *,
                 process_input=None, process_heads=None):
        super().__init__()

        self.base_net = base_net
        self.head_nets = None
        self.process_input = process_input
        self.process_heads = process_heads

        self.set_head_nets(head_nets)

        # Super mega-ugly hack
        if getattr(self.head_nets[1], 'joint_transformer', False):
            self.neck = JointTransformer(in_channels=self.base_net.out_features,
                                          out_channels=256, embed_dim=256, mlp_ratio=8.,
                                          num_heads=8, depth=6, drop_path_rate=0.0)
        elif getattr(self.head_nets[1], 'solo_transformer', False):
            self.neck = SoloTransformer(in_channels=self.base_net.out_features,
                                         out_channels=256, embed_dim=256, mlp_ratio=8.,
                                         num_heads=8, depth=6, drop_path_rate=0.0)
        else:
            self.neck = None

    def set_head_nets(self, head_nets):
        if not isinstance(head_nets, torch.nn.ModuleList):
            head_nets = torch.nn.ModuleList(head_nets)

        for hn_i, hn in enumerate(head_nets):
            hn.meta.head_index = hn_i
            hn.meta.base_stride = self.base_net.stride

        self.head_nets = head_nets

    def forward(self, *args):
        image_batch = args[0]

        if self.process_input is not None:
            image_batch = self.process_input(image_batch)

        x = self.base_net(image_batch)

        if getattr(self, 'neck', False) and self.neck is not None:
            x = self.neck(x)

        head_outputs = []
        if len(args) >= 2:
            head_mask = args[2]
            targets = args[1]
            for hn_idx, (hn, m) in enumerate(zip(self.head_nets, head_mask)):
                if m:
                    if hn.__class__.__name__ == "RafHead" and hn.meta.refine_hm:
                        if isinstance(x,list) and len(x) > 1:
                            head_outputs.append(hn((x[hn_idx%2], targets[hn_idx])))
                        else:
                            head_outputs.append(hn((x, targets)))
                    else:
                        if getattr(self, 'neck', False) and self.neck is not None:
                            head_outputs.append(hn(x[hn_idx]))
                        elif isinstance(x,list) and len(x) > 1:
                            head_outputs.append(hn(x[hn_idx%2]))
                        else:
                            head_outputs.append(hn(x, targets))
                else:
                    head_outputs.append(None)

            head_outputs = tuple(head_outputs)
        else:
            for hn_idx, hn in enumerate(self.head_nets):
                # if (len(self.head_nets) == 1) or len(x) != len(self.head_nets):
                #     head_outputs = tuple(hn(x) for hn in self.head_nets)
                # else:
                #     head_outputs = tuple(hn(x_input) for hn, x_input in zip(self.head_nets, x))
                if getattr(self, 'neck', False) and self.neck is not None:
                    head_outputs.append(hn(x[hn_idx]))
                elif isinstance(x,list) and len(x) > 1:
                    head_outputs.append(hn(x[hn_idx%2]))
                else:
                    head_outputs.append(hn(x))

        if self.process_heads is not None:
            head_outputs = self.process_heads(head_outputs)
        # if has_combined and self.base_net.training:
        #     return head_outputs, combined_hm_preds
        return head_outputs


class CrossTalk(torch.nn.Module):
    def __init__(self, strength=0.2):
        super().__init__()
        self.strength = strength

    def forward(self, *args):
        image_batch = args[0]
        if self.training and self.strength:
            rolled_images = torch.cat((image_batch[-1:], image_batch[:-1]))
            image_batch += rolled_images * self.cross_talk
        return image_batch


# pylint: disable=protected-access
def model_migration(net_cpu):
    model_defaults(net_cpu)

    if not hasattr(net_cpu, 'process_heads'):
        net_cpu.process_heads = None

    for m in net_cpu.modules():
        if not hasattr(m, '_non_persistent_buffers_set'):
            m._non_persistent_buffers_set = set()

    if not hasattr(net_cpu, 'head_nets') and hasattr(net_cpu, '_head_nets'):
        net_cpu.head_nets = net_cpu._head_nets

    for hn_i, hn in enumerate(net_cpu.head_nets):
        if not hn.meta.base_stride:
            hn.meta.base_stride = net_cpu.base_net.stride
        if hn.meta.head_index is None:
            hn.meta.head_index = hn_i
        if hn.meta.name == 'cif' and 'score_weights' not in vars(hn.meta):
            hn.meta.score_weights = [3.0] * 3 + [1.0] * (hn.meta.n_fields - 3)

    for mm in MODEL_MIGRATION:
        mm(net_cpu)


def model_defaults(net_cpu):
    return
    import pdb; pdb.set_trace()
    for m in net_cpu.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            # avoid numerical instabilities
            # (only seen sometimes when training with GPU)
            # Variances in pretrained models can be as low as 1e-17.
            # m.running_var.clamp_(min=1e-8)
            # m.eps = 1e-3  # tf default is 0.001
            # m.eps = 1e-5  # pytorch default

            # This epsilon only appears inside a sqrt in the denominator,
            # i.e. the effective epsilon for division is much bigger than the
            # given eps.
            # See equation here:
            # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
            m.eps = 1e-4

            # smaller step size for running std and mean update
            m.momentum = 0.01  # tf default is 0.99
            # m.momentum = 0.1  # pytorch default

        elif isinstance(m, (torch.nn.GroupNorm, torch.nn.LayerNorm)):
            m.eps = 1e-4

        elif isinstance(m, (torch.nn.InstanceNorm1d, torch.nn.InstanceNorm2d)):
            m.eps = 1e-4
            m.momentum = 0.01
