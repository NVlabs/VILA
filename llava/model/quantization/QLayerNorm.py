import torch
import torch.nn as nn
from torch.autograd.function import Function, InplaceFunction

try:
    from .QAct import QAct_FPin, QAct_FPout
    from .Qconfig import qconfig
    from .QFunction import *
    from .utils import *

except:
    from Qconfig import qconfig
    from utils import *
    from QFunction import *
    from .QAct import QAct_FPin, QAct_FPout

import os
from copy import deepcopy

import matplotlib.pyplot as plt


class QLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, args=None, layer_type=""):
        super().__init__()
        self.args = deepcopy(args)
        self.layer_type = layer_type
        assert layer_type != "", "layer_type is not defined"
        assert layer_type in qconfig.qlayernorm_config, f"{layer_type} not in qlayernorm_config"

        self.apply_quantize = list_has_common_element(args.qchoice, qconfig.qlayernorm_config[layer_type])

        self.fbit = self.args.fabit if self.args.fabit else self.Ubit
        self.bbit = self.args.babit if self.args.babit else self.Ubit

        quantize_flag = format_string_with_condition(
            layer_type,
            {"apply": self.apply_quantize},
            self.args.symm,
            self.fbit,
            self.bbit,
            {"row": self.args.row_blocksize, "col": self.args.col_blocksize},
        )
        print(quantize_flag)

        self.ln_in = QAct_FPout(args, layer_type=layer_type + "_in")
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps)
        self.ln_out = QAct_FPin(args, layer_type=layer_type + "_out")

    def forward(self, Qinput, Iscale):
        # input shape is (Batch Size, Sequence Length, Hidden Size)
        input = self.ln_in(Qinput, Iscale)
        output_fp = self.layer_norm(input)
        # import IPython
        # IPython.embed()
        output, scale = self.ln_out(output_fp)
        return output, scale


if __name__ == "__main__":
    Sum = torch.load("tensor/QAct_nan_epoch16.pt")


