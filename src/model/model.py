import torch.nn as nn
from layers import ConvBlock, MiddleBlock, ExitBlock

class Xception(nn.Sequential):
    def __init__(self, version, config, classify=True):
        super(Xception, self).__init__()

        in_ch = config["in_ch"]
        channels = config["channels"]
        strides = config["strides"]
        n_classes = config["n_classes"]

        self.add_module('entry_flow', ConvBlock(in_ch, channels[0], k=3, s=2, pad=1))

        middle_blocks = [
            MiddleBlock(in_ch=channels[i-1], out_ch=channels[i], stride=strides[i-1])
            for i in range(1, len(channels))
        ]
        self.add_module('middle_flow', nn.Sequential(*middle_blocks))

        self.add_module('exit_flow', ExitBlock(channels[-2], channels[-1], strides[-1]))

        if classify:
            self.add_module(
                name='fc',
                module=nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(channels[-1], n_classes)
                )
            )

    def forward(self, inputs):
        return super(Xception, self).forward(inputs)

config_71 = {
    "in_ch": 3,
    "n_classes": 10, 
    "channels": [32, 64, 128, 256, 728, 728, 728],
    "strides": [1, 2, 1, 2, 1, 2,1],
}

config_56 = {
    "in_ch": 3,
    "n_classes": 10,
    "channels": [32, 64, 128, 256, 728],
    "strides": [1, 2, 1, 2]
}

model_71 = Xception("71", config_71, classify=True)
model_56 = Xception("56", config_56, classify=True)

print(model_71(torch.randn(1, 3, 224, 224)).shape)
print(model_56(torch.randn(1, 3, 224, 224)).shape)
