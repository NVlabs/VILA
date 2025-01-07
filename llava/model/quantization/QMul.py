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


class QMul(nn.Module):
    def __init__(self, args=None, layer_type=""):
        super().__init__()
        self.args = deepcopy(args)
        self.layer_type = layer_type
        assert layer_type != "", "layer_type is not defined"
        assert layer_type in qconfig.qmul_config, f"{layer_type} not in qgelu_config"

        self.apply_quantize = list_has_common_element(args.qchoice, qconfig.qmul_config[layer_type])

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

        self.Mul_in1 = QAct_FPout(args, layer_type=layer_type + "_in1")
        self.Mul_in2 = QAct_FPout(args, layer_type=layer_type + "_in2")
        self.Mul_out = QAct_FPin(args, layer_type=layer_type + "_out")

    def forward(self, Qinput1, Qinput2, Iscale1, Iscale2):
        # input shape is (Batch Size, Sequence Length, Hidden Size)
        input1 = self.Mul_in1(Qinput1, Iscale1)
        input2 = self.Mul_in2(Qinput2, Iscale2)
        output_fp = input1 * input2
        Qoutput, Oscale = self.Mul_out(output_fp)
        return Qoutput, Oscale


if __name__ == "__main__":
    Sum = torch.load("tensor/QAct_nan_epoch16.pt")


