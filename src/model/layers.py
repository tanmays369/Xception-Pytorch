import torch.nn as nn

class SeperableConv(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, pad=0):
        super(SeperableConv, self).__init__()
        self.add_module("depthwise", nn.Conv2d(in_ch, in_ch, k, s, pad, groups=in_ch, bias=False))
        self.add_module("pointwise", nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False))
        self.add_module("bn", nn.BatchNorm2d(out_ch))

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, pad=0, bias=False, use_relu=True):
        super(ConvBlock, self).__init__()
        self.add_module("conv", nn.Conv2d(in_ch, out_ch, k, s, pad, bias=bias))
        self.add_module("bn", nn.BatchNorm2d(out_ch))
        if use_relu:
            self.add_module("relu", nn.ReLU6(inplace=True))

class MiddleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, s):
        super(MiddleBlock, self).__init__()
        self.add_module("middle1", SeperableConv(in_ch, 728, k=3, s=s, pad=1))
        self.add_module("middle2", SeperableConv(728, out_ch, k=3, s=1, pad=1))
        self.add_module("relu", nn.ReLU6(inplace=True))

class ExitBlock(nn.Module):
    def __init__(self, in_ch, out_ch, s):
        super(ExitBlock, self).__init__()
        self.add_module("exit1", SeperableConv(in_ch, out_ch, k=3, s=s, pad=1))
        self.add_module("exit2", SeperableConv(out_ch, 2 * out_ch, k=3, s=1, pad=1))
        self.add_module("relu", nn.ReLU6(inplace=True))
