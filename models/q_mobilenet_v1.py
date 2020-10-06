import math
import numpy as np
import torch.nn as nn


from .quant_ops import (
    SwitchBN2d,
    QConv2d,
    QLinear
)
from utils.config import FLAGS


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inp, outp, stride):
        super(DepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]

        layers = [
            QConv2d(
                inp, inp, 3, stride, 1, groups=inp, bias=False, pact_fp=getattr(FLAGS, 'pact_fp', False)),
            SwitchBN2d(inp, affine=not getattr(FLAGS, 'stats_sharing', False)),
            nn.ReLU(inplace=True),

            QConv2d(inp, outp, 1, 1, 0, bias=False, pact_fp=getattr(FLAGS, 'pact_fp', False)),
            SwitchBN2d(outp, affine=not getattr(FLAGS, 'stats_sharing', False)),
            nn.ReLU(inplace=True),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        return self.body(x)


class Model(nn.Module):
    def __init__(self, num_classes=1000):
        super(Model, self).__init__()

        # setting of inverted residual blocks
        self.block_setting = [
            # c, n, s
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2],
        ]

        # head
        channels = 32
        first_stride = 2
        self.head = nn.Sequential(
                        QConv2d(
                            3, channels, 3,
                            first_stride, 1, bias=False,
                            bitw_min=8, bita_min=8, weight_only=True),
                        SwitchBN2d(channels, affine=not getattr(FLAGS, 'stats_sharing', False)),
                        nn.ReLU(inplace=True),
                    )

        # body
        for idx, [c, n, s] in enumerate(self.block_setting):
            outp = c
            for i in range(n):
                if i == 0:
                    setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                        DepthwiseSeparableConv(channels, outp, s))
                else:
                    setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                        DepthwiseSeparableConv(channels, outp, 1))
                channels = outp

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # classifier
        self.classifier = nn.Sequential(
            QLinear(
                outp,
                num_classes,
                bitw_min=8,
                pact_fp=getattr(FLAGS, 'pact_fp', False)
            )
        )
        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.head(x)
        for idx, [_, n, _] in enumerate(self.block_setting):
            for i in range(n):
                x = getattr(self, 'stage_{}_layer_{}'.format(idx, i))(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, SwitchBN2d):
                if not m.affine:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
