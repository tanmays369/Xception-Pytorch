from torch import nn
from layers import ConvBlock, MiddleBlock, ExitBlock

class Xception(nn.Sequential):
    def __init__(self, config, classify=True):
        super(Xception, self).__init__()

        in_ch= config["in_ch"]
        channels= config["channels"]
        strides= config["strides"]
        n_classes= config["n_classes"]

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
        return super(Xception, self).__init__(inputs)

config = {
    "in_ch":3,
    "n_classes":10,
    "channels":[32,64,128,256,728],
    "strides":[1,2,1,2]
}

model= Xception(config, classify=True)
print(model(torch.randn(1,3,224,224)).shape)
