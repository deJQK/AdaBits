import math
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.utils import _pair


from utils.config import FLAGS


class q_k(Function):
    """
        This is the quantization module.
        The input and output should be all on the interval [0, 1].
        bit is only defined on positive integer values.
    """
    @staticmethod
    def forward(ctx, input, bit, scheme='novel'):
        assert bit > 0
        assert torch.all(input >= 0) and torch.all(input <= 1)
        if scheme == 'original':
            a = (1 << bit) - 1
            res = torch.round(a * input)
            res.div_(a)
        elif scheme == 'modified':
            a = 1 << bit
            res = torch.floor(a * input)
            res.clamp_(max=a - 1).div_(a)
        else:
            raise NotImplementedError
        assert torch.all(res >= 0) and torch.all(res <= 1)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def round_width(width, wm=None,
                width_divisor=getattr(FLAGS, 'width_divisor', 1),
                min_width=getattr(FLAGS, 'min_width', 1)):
    if not wm:
        return width
    width *= wm
    if min_width is None:
        min_width = width_divisor
    new_width = max(min_width, int(width + width_divisor / 2) // width_divisor * width_divisor)
    if new_width < 0.9 * width:
        new_width += width_divisor
    return int(new_width)


class SwitchBN2d(nn.Module):
    def __init__(self, num_features,
                 eps=1e-05, momentum=0.1,
                 affine=True):
        super(SwitchBN2d, self).__init__()
        self.num_features = num_features
        bns = []
        for i in range(len(FLAGS.bits_list)):
            bns.append(nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine))
        self.bn = nn.ModuleList(bns)
        self.bits = FLAGS.bits_list[-1]
        self.ignore_model_profiling = True
        if affine:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        else:
            self.weight = nn.Parameter(torch.Tensor(self.num_features))
            self.bias = nn.Parameter(torch.Tensor(self.num_features))
        self.affine = affine

    def forward(self, input):
        if getattr(FLAGS, 'switchbn', True):
            if self.bits in FLAGS.bits_list:
                idx = FLAGS.bits_list.index(self.bits)
            else:
                idx = 0
            y = self.bn[idx](input)
            if not self.affine:
                y = self.weight[None, :, None, None] * y + self.bias[None, :, None, None]
        else:
            y = self.bn[0](input)
        return y


class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True,
                 same_padding=False,
                 bitw_min=None, bita_min=None,
                 pact_fp=False,
                 double_side=False,
                 weight_only=False):
        super(QConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride=stride,
            padding=padding if not same_padding else 0,
            dilation=dilation,
            groups=groups, bias=bias)
        self.bits = FLAGS.bits_list[-1]
        self.same_padding = same_padding
        self.bitw_min = bitw_min
        self.bita_min = bita_min
        self.pact_fp = pact_fp
        self.double_side = double_side
        self.weight_only = weight_only or getattr(FLAGS, 'weight_only', False)
        self.quant = q_k.apply
        if getattr(FLAGS, 'switch_alpha', False):
            self.alpha = nn.Parameter(torch.ones(len(FLAGS.bits_list)) * 8.0)
        else:
            self.alpha = nn.Parameter(torch.tensor(8.0))

    def forward(self, input, bitw, bita):
        if self.same_padding:
            ih, iw = input.size()[-2:]
            kh, kw = self.weight.size()[-2:]
            sh, sw = self.stride
            oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
            pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
            pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
            if pad_h > 0 or pad_w > 0:
                input = nn.functional.pad(input, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        bitw, bita = self.get_bits()

        weight_quant_scheme = getattr(FLAGS, 'weight_quant_scheme', 'modified')
        act_quant_scheme = getattr(FLAGS, 'act_quant_scheme', 'original')

        if bitw == 0:
            return nn.Identity()(input)

        if bitw < 32:
            weight = torch.tanh(self.weight) / torch.max(torch.abs(torch.tanh(self.weight)))
            weight.add_(1.0)
            weight.div_(2.0)
            weight = self.quant(weight, bitw, weight_quant_scheme)
            weight.mul_(2.0)
            weight.sub_(1.0)
            if getattr(FLAGS, 'rescale_conv', False):
                rescale_type = getattr(FLAGS, 'rescale_type', 'constant')
                if rescale_type == 'stddev':
                    weight_scale = torch.std(self.weight.detach())
                elif rescale_type == 'constant':
                    weight_scale = 1.0 / (self.out_channels * self.kernel_size[0] * self.kernel_size[1]) ** 0.5
                else:
                    raise NotImplementedError
                weight_scale = weight_scale / torch.std(weight.detach())
                weight.mul_(weight_scale)
        else:
            # no quantizing but only clamping
            if getattr(FLAGS, 'clamp', True):
                weight = torch.tanh(self.weight) / torch.max(torch.abs(torch.tanh(self.weight)))
            else:
                weight = self.weight * 1.0
            if getattr(FLAGS, 'rescale_conv', False):
                rescale_type = getattr(FLAGS, 'rescale_type', 'constant')
                if rescale_type == 'stddev':
                    weight_scale = torch.std(self.weight.detach())
                elif rescale_type == 'constant':
                    weight_scale = 1.0 / (self.out_channels * self.kernel_size[0] * self.kernel_size[1]) ** 0.5
                else:
                    raise NotImplementedError
                weight_scale = weight_scale / torch.std(weight.detach())
                weight.mul_(weight_scale)

        if (bita < 32 and not self.weight_only) or self.pact_fp:
            if getattr(FLAGS, 'switch_alpha', False):
                if bita in FLAGS.bits_list:
                    idx_alpha = FLAGS.bits_list.index(bita)
                else:
                    idx_alpha = 0
                alpha = torch.abs(self.alpha[idx_alpha])
            else:
                alpha = torch.abs(self.alpha)
            if self.double_side:
                input_val = torch.where(input_val > -alpha, input, -alpha)
            else:
                input_val = torch.relu(input)
            input_val = torch.where(input_val < alpha, input_val, alpha)
            if bita < 32 and not self.weight_only:
                input_val.div_(alpha)
                if self.double_side:
                    input_val.add_(1.0)
                    input_val.div_(2.0)
                input_val = self.quant(input_val, bita, act_quant_scheme)
                if self.double_side:
                    input_val.mul_(2.0)
                    input_val.sub_(1.0)
                input_val.mul_(alpha)
        else:
            input_val = input

        y = nn.functional.conv2d(
            input_val, weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups)
        return y

    def get_bits(self):
        bitw, bita = _pair(self.bits)
        if self.bitw_min is not None:
            bitw = max(bitw, self.bitw_min)
        if self.bita_min is not None:
            bita = max(bita, self.bita_min)
        if self.weight_only:
            bita = 32
        return bitw, bita


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, bitw_min=None, bita_min=None, pact_fp=False, weight_only=False):
        super(QLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.bits = FLAGS.bits_list[-1]
        self.bitw_min = bitw_min
        self.bita_min = bita_min
        self.pact_fp = pact_fp
        self.weight_only = weight_only or getattr(FLAGS, 'weight_only', False)
        self.quant = q_k.apply
        if getattr(FLAGS, 'switch_alpha', False):
            self.alpha = nn.Parameter(torch.ones(len(FLAGS.bits_list)) * 10.0)
        else:
            self.alpha = nn.Parameter(torch.tensor(10.0))

    def forward(self, input, bitw, bita):
        bitw, bita = self.get_bits()

        weight_quant_scheme = getattr(FLAGS, 'weight_quant_scheme', 'modified')
        act_quant_scheme = getattr(FLAGS, 'act_quant_scheme', 'original')

        if bitw == 0:
            return nn.Identity()(input)

        if bitw < 32:
            weight = torch.tanh(self.weight) / torch.max(torch.abs(torch.tanh(self.weight)))
            weight.add_(1.0)
            weight.div_(2.0)
            weight = self.quant(weight, bitw, weight_quant_scheme)
            weight.mul_(2.0)
            weight.sub_(1.0)
            if getattr(FLAGS, 'rescale', True):
                rescale_type = getattr(FLAGS, 'rescale_type', 'constant')
                if rescale_type == 'stddev':
                    weight_scale = torch.std(self.weight.detach())
                elif rescale_type == 'constant':
                    weight_scale = 1.0 / (self.out_features) ** 0.5
                else:
                    raise NotImplementedError
                weight_scale = weight_scale / torch.std(weight.detach())
                if self.training:
                    weight.mul_(weight_scale)
        else:
            if getattr(FLAGS, 'clamp', True):
                weight = torch.tanh(self.weight) / torch.max(torch.abs(torch.tanh(self.weight)))
            else:
                weight = self.weight * 1.0
            if getattr(FLAGS, 'rescale', True):
                rescale_type = getattr(FLAGS, 'rescale_type', 'constant')
                if rescale_type == 'stddev':
                    weight_scale = torch.std(self.weight.detach())
                elif rescale_type == 'constant':
                    weight_scale = 1.0 / (self.out_features) ** 0.5
                else:
                    raise NotImplementedError
                weight_scale = weight_scale / torch.std(weight.detach())
                if self.training:
                   weight.mul_(weight_scale)
        bias = self.bias
        if bias is not None and getattr(FLAGS, 'rescale', True) and not self.training:
                bias.div_(weight_scale)

        if (bita < 32 and not self.weight_only) or self.pact_fp:
            if getattr(FLAGS, 'switch_alpha', False):
                if bita in FLAGS.bits_list:
                    idx_alpha = FLAGS.bits_list.index(bita)
                else:
                    idx_alpha = 0
                alpha = torch.abs(self.alpha[idx_alpha])
            else:
                alpha = torch.abs(self.alpha)
            input_val = torch.relu(input)
            input_val = torch.where(input_val < alpha, input_val, alpha)
            if bita < 32 and not self.weight_only:
                input_val.div_(alpha)
                input_val = self.quant(input_val, bita, act_quant_scheme)
                input_val.mul_(alpha)
        else:
            input_val = input
        return nn.functional.linear(input_val, weight, bias)

    def get_bits(self):
        bitw, bita = _pair(self.bits)
        if self.bitw_min is not None:
            bitw = max(bitw, self.bitw_min)
        if self.bita_min is not None:
            bita = max(bita, self.bita_min)
        if self.weight_only:
            bita = 32
        return bitw, bita
