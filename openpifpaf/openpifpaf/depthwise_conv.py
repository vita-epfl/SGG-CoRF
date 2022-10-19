from torch import nn

class DepthwiseSeparableConv2d(nn.Module):

    def __init__(self, channels, padding):
        super().__init__()
        self.depthwise_conv = nn.Conv2d(channels, channels, 3, padding=padding, groups=channels, bias=False)
        self.pointwise_conv = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x
