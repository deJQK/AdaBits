import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair


from .quant_ops import (
    SwitchBN2d,
    QConv2d,
    QLinear,
)

from utils.config import FLAGS


class Block(nn.Module):
    def __init__(self, inp, outp, stride):
        super(Block, self).__init__()
        assert stride in [1, 2]

        l1 = QConv2d(inp, outp, 3, stride, 1, bias=False)
        l2 = QConv2d(outp, outp, 3, 1, 1, bias=False)
        layers = [
            l1,
            SwitchBN2d(outp, affine=not getattr(FLAGS, 'stats_sharing', False)),
            nn.ReLU(inplace=True),

            l2,
            SwitchBN2d(outp, affine=not getattr(FLAGS, 'stats_sharing', False)),
        ]
        self.body = nn.Sequential(*layers)

        self.residual_connection = stride == 1 and inp == outp
        if not self.residual_connection:
            self.shortcut = nn.Sequential(
                QConv2d(inp, outp, 1, stride=stride, bias=False),
                SwitchBN2d(outp, affine=not getattr(FLAGS, 'stats_sharing', False)),
            )
        self.post_relu = nn.ReLU(inplace=True)


    def forward(self, x):
        res = self.body(x)
        if self.residual_connection:
            res += x
        else:
            res += self.shortcut(x)
        return self.post_relu(res)


class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()

        # head
        channels = 16
        l_head = QConv2d(3, channels, 3, 1, 1, bias=False,
                         bitw_min=max(FLAGS.bits_list), bita_min=8, weight_only=True)
        self.head = nn.Sequential(
                        l_head,
                        SwitchBN2d(channels, affine=not getattr(FLAGS, 'stats_sharing', False)),
                        nn.ReLU(inplace=True),
                    )

        # setting of inverted residual blocks
        self.block_setting_dict = {
            # : [stage1, stage2, stage3, stage4]
            
            20: [3, 3, 3],
            56: [9, 9, 9],
            110: [18, 18, 18]
        }
        self.block_setting = self.block_setting_dict[FLAGS.depth]

        feats = [16, 32, 64]

        # body
        for idx, n in enumerate(self.block_setting):
            outp = feats[idx]
            for i in range(n):
                if i == 0 and idx != 0:
                    layer = Block(channels, outp, 2)
                else:
                    layer = Block(channels, outp, 1)
                setattr(self, 'stage_{}_layer_{}'.format(idx, i), layer)
                channels = outp

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # classifier
        self.classifier = nn.Sequential(
            QLinear(
                outp,
                num_classes,
                bitw_min=max(FLAGS.bits_list)
            )
        )

        if FLAGS.reset_parameters:
            self.reset_parameters()

    def forward(self, x):
        x = self.head(x)
        for idx, n in enumerate(self.block_setting):
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
