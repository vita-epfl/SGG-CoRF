import torch
import torch.nn as nn


class HeadNet(nn.Module):
    def __init__(self, transformer, raf_head):
        super().__init__()
        self.transformer = transformer
        self.raf_head = raf_head

    def forward(self, x):
        x = self.transformer(x)
        x = self.raf_head(x)
        return x

    def get_last_selfattention(self, x):
        attn = self.transformer.get_last_selfattention(x)
        return attn


class Joiner(nn.Module):

    def __init__(self, base_net, head_net):
        super().__init__()
        self.base_net = base_net
        self.head_net = head_net

    def forward(self, x):
        x = self.base_net(x)
        x = self.head_net(x)
        return x

    def get_last_selfattention(self, x):
        x = self.base_net(x)
        attn = self.head_net.get_last_selfattention(x)
        return attn
