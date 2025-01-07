import torch
import torch.nn as nn
from torch.autograd.function import Function, InplaceFunction

try:
    from .Qconfig import qconfig
    from .QFunction import *
    from .utils import *
except:
    from Qconfig import qconfig
    from utils import *
    from QFunction import *

import os
from copy import deepcopy

import matplotlib.pyplot as plt


class QIdentity(nn.Identity):
    def __init__(self):
        super().__init__()

    def forward(self, input, scale):
        input = QuantIdentity.apply(input, scale)

        return input


class QuantIdentity(Function):
    @staticmethod
    def forward(ctx, input, scale):
        return input, scale

    @staticmethod
    def backward(ctx, grad_output, Gscale):
        import IPython

        IPython.embed()
        return grad_output, Gscale


if __name__ == "__main__":
    Sum = torch.load("tensor/QAct_nan_epoch16.pt")
    Qinput, Binput, input, args, layer_type, name = (
        Sum["Qinput"],
        Sum["Binput"],
        Sum["input"],
        Sum["args"],
        Sum["layer_type"],
        Sum["name"],
    )
    if_nan, if_inf = check_nan_inf(input, True, False)
    print(if_nan)

    Q = block_quant(Binput, True, 8, stochastic=False, epsilon=1e-8)


