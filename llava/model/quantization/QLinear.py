import os
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function, InplaceFunction
from torch.cuda import amp

from .Qconfig import qconfig
from .QFunction import *
from .utils import *


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, args=None, layer_type=""):
        super().__init__(in_features, out_features, bias)
        self.args = deepcopy(args)
        self.layer_type = layer_type
        assert layer_type != "", "layer_type is not defined"
        assert layer_type in qconfig.qlinear_config.keys(), f"{layer_type} not in qlinear_config"

        self.apply_quantize = list_has_common_element(args.qchoice, qconfig.qlinear_config[layer_type])
        self.apply_quantize_fw, self.apply_quantize_fo, self.apply_quantize_bw, self.apply_quantize_ba = (
            self.apply_quantize,
            self.apply_quantize,
            self.apply_quantize,
            self.apply_quantize,
        )

        self.refine_rowcol_blocksize()

        self.fbit = self.args.fwbit if self.args.fwbit else self.Ubit
        self.bbit = self.args.bwbit if self.args.bwbit else self.Ubit
        quantize_flag = format_string_with_condition(
            layer_type,
            {
                "apply-fw": self.apply_quantize_fw,
                "apply-fo": self.apply_quantize_fo,
                "apply-bw": self.apply_quantize_bw,
                "apply-ba": self.apply_quantize_ba,
            },
            self.args.symm,
            self.fbit,
            self.bbit,
            {
                "row-fa": self.args.row_blocksize_fa,
                "col-fa": self.args.col_blocksize_fa,
                "row-fw": self.args.row_blocksize_fw,
                "col-fw": self.args.col_blocksize_fw,
                "row-fo": self.args.row_blocksize_fo,
                "col-fo": self.args.col_blocksize_fo,
                "row-ba": self.args.row_blocksize_ba,
                "col-ba": self.args.col_blocksize_ba,
                "row-bw": self.args.row_blocksize_bw,
                "col-bw": self.args.col_blocksize_bw,
                "row-bo": self.args.row_blocksize_bo,
                "col-bo": self.args.col_blocksize_bo,
            },
        )
        if quant_get_local_rank() == 0:
            print(quantize_flag)

    def refine_rowcol_blocksize(self):
        self.args.row_blocksize_fa, self.args.col_blocksize_fa = self.args.row_blocksize, self.args.col_blocksize
        self.args.row_blocksize_fw, self.args.col_blocksize_fw = self.args.row_blocksize, self.args.col_blocksize
        self.args.row_blocksize_fo, self.args.col_blocksize_fo = self.args.row_blocksize, self.args.col_blocksize
        self.args.row_blocksize_ba, self.args.col_blocksize_ba = self.args.row_blocksize, self.args.col_blocksize
        self.args.row_blocksize_bw, self.args.col_blocksize_bw = self.args.row_blocksize, self.args.col_blocksize
        self.args.row_blocksize_bo, self.args.col_blocksize_bo = self.args.row_blocksize, self.args.col_blocksize

        if self.args.refine_attn_blocksize:
            if self.layer_type in ["attn_q", "attn_k", "attn_v"]:
                self.apply_quantize_fo = False
                self.args.row_blocksize_ba, self.args.col_blocksize_ba = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )
            if self.layer_type in ["attn_proj"]:
                self.apply_quantize_ba = False
                self.args.row_blocksize_fo, self.args.col_blocksize_fo = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )

        if self.args.refine_mlp_blocksize:
            if self.layer_type in ["mlp_gate", "mlp_up", "mlp_down"]:
                self.args.row_blocksize_fo, self.args.col_blocksize_fo = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )
                self.args.row_blocksize_ba, self.args.col_blocksize_ba = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )

    def forward(self, Qinput, Iscale):
        if self.training:
            output = QuantLinear.apply(
                Qinput,
                Iscale,
                self.weight,
                self.bias,
                self.args,
                self.layer_name,
                self.apply_quantize_fw,
                self.apply_quantize_fo,
                self.apply_quantize_bw,
                self.apply_quantize_ba,
            )
            return output
        else:
            output = F.linear(Qinput, self.weight, self.bias)
            return output, None


# class QuantLinear(Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias, args, layer_type):
#         ctx.saved = input, weight, bias, args, layer_type
#         return F.linear(input, weight, bias)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         input, weight, bias, args, layer_type = ctx.saved
#
#         C_in = input.shape[-1]
#         C_out = grad_output.shape[-1]
#
#         grad_output_flatten = grad_output.reshape(-1, C_out)
#         input_flatten = input.reshape(-1, C_in)
#
#         if grad_output_flatten.dtype == input_flatten.dtype:
#             grad_weight = grad_output_flatten.t().mm(input_flatten)
#         else:
#             grad_weight = grad_output_flatten.float().t().mm(input_flatten)
#
#         if grad_output_flatten.dtype == weight.dtype:
#             grad_input = grad_output_flatten.mm(weight)
#         else:
#             grad_input = grad_output_flatten.float().mm(weight)
#
#         if bias is not None:
#             grad_bias = grad_output_flatten.sum(0)
#         else:
#             grad_bias = None
#
#         grad_input_transform = grad_input.reshape(input.size())
#
#         return grad_input_transform, grad_weight, grad_bias, None, None

# B%% = block_cut(%%, args.row_blocksize, args.col_blocksize)
# RQ%%, Q%%, Wscale = block_quant(B%%, args.symm, args.fwbit, stochastic=False, epsilon=args.epsilon)
# Q%% = block_reshape(Q%%, %%, args.row_blocksize, args.col_blocksize)
# RQ%% = block_reshape(RQ%%, %%, args.row_blocksize, args.col_blocksize)


class QuantLinear(Function):
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(
        ctx,
        Qinput,
        Iscale,
        weight,
        bias,
        args,
        layer_name,
        apply_quantize_fw=True,
        apply_quantize_fo=True,
        apply_quantize_bw=True,
        apply_quantize_ba=True,
    ):

        # shrink Iscale to let the size of gradient the same as forward
        ideal_scale_num = Qinput.numel() / (args.min_blockunit_row * args.min_blockunit_col)
        actual_scale_num = calculate_scale_num(Qinput, args.row_blocksize_fa, args.col_blocksize_fa)
        # actual_scale_num = Qinput.numel() / (args.row_blocksize_fa * args.col_blocksize_fa)
        assert Iscale.shape[0] == ideal_scale_num
        Iscale = Iscale[: int(actual_scale_num), :, :]

        Binput = block_cut(Qinput, args.row_blocksize_fa, args.col_blocksize_fa)
        RQinput = Binput * Iscale
        RQinput = block_reshape(RQinput, Qinput, args.row_blocksize_fa, args.col_blocksize_fa)

        Bweight = block_cut(weight, args.row_blocksize_fw, args.col_blocksize_fw)
        RQweight, Qweight, Wscale = block_quant(
            Bweight,
            args.symm,
            args.fwbit,
            stochastic=False,
            epsilon=args.epsilon,
            apply_quantize=apply_quantize_fw,
            layer_name=layer_name + "WeightQuant",
        )
        Qweight = block_reshape(Qweight, weight, args.row_blocksize_fw, args.col_blocksize_fw)
        RQweight = block_reshape(RQweight, weight, args.row_blocksize_fw, args.col_blocksize_fw)

        if args.draw_distribution_forward:
            save_tensor(weight, Qweight, RQweight, fb="forward", aw="Weight", layer_name=layer_name)

        ctx.saved = Qinput, Iscale, Qweight, Wscale, bias, args, layer_name
        ctx.apply_quantize = apply_quantize_fw, apply_quantize_fo, apply_quantize_bw, apply_quantize_ba
        fc_output = F.linear(RQinput, RQweight, bias)

        Bfc_output = block_cut(fc_output, args.row_blocksize_fo, args.col_blocksize_fo)
        RQfc_output, Qfc_output, Oscale = block_quant(
            Bfc_output,
            args.symm,
            args.fabit,
            stochastic=False,
            epsilon=args.epsilon,
            apply_quantize=apply_quantize_fo,
            layer_name=layer_name + "LinearOutput",
        )
        RQfc_output = block_reshape(RQfc_output, fc_output, args.row_blocksize_fo, args.col_blocksize_fo)
        Qfc_output = block_reshape(Qfc_output, fc_output, args.row_blocksize_fo, args.col_blocksize_fo)

        if args.draw_distribution_forward:
            save_tensor(fc_output, Qfc_output, RQfc_output, fb="forward", aw="Output", layer_name=layer_name)

        # enlarge Oscale to let the size of gradient the same as forward
        ideal_scale_num = Qfc_output.numel() / (args.min_blockunit_row * args.min_blockunit_col)
        actual_scale_num = calculate_scale_num(Qfc_output, args.row_blocksize_fo, args.col_blocksize_fo)
        # actual_scale_num = Qfc_output.numel() / (args.row_blocksize_fo * args.col_blocksize_fo)
        assert Oscale.shape[0] == actual_scale_num
        Oscale = torch.nn.functional.pad(Oscale, (0, 0, 0, 0, 0, int(ideal_scale_num - actual_scale_num)))

        return Qfc_output, Oscale

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, Qgrad_output, Gscale):
        Qinput, Iscale, Qweight, Wscale, bias, args, layer_name = ctx.saved
        apply_quantize_fw, apply_quantize_fo, apply_quantize_bw, apply_quantize_ba = ctx.apply_quantize

        # shrink Gscale to let the size of gradient the same as forward
        ideal_scale_num = Qgrad_output.numel() / (args.min_blockunit_row * args.min_blockunit_col)
        actual_scale_num = calculate_scale_num(Qgrad_output, args.row_blocksize_bo, args.col_blocksize_bo)
        # actual_scale_num = Qgrad_output.numel() / (args.row_blocksize_bo * args.col_blocksize_bo)
        assert Gscale.shape[0] == ideal_scale_num
        Gscale = Gscale[: int(actual_scale_num), :, :]

        Bgrad_output = block_cut(Qgrad_output, args.row_blocksize_bo, args.col_blocksize_bo)
        RQgrad_output = Bgrad_output * Gscale
        grad_output = block_reshape(RQgrad_output, Qgrad_output, args.row_blocksize_bo, args.col_blocksize_bo)

        if args.draw_distribution_backward:
            save_tensor(
                grad_output, Qgrad_output, RQgrad_output, fb="backward in", aw="Activation", layer_name=layer_name
            )

        C_in = Qinput.shape[-1]
        C_out = Qgrad_output.shape[-1]

        Binput = block_cut(Qinput, args.row_blocksize_fa, args.col_blocksize_fa)
        input = Binput * Iscale
        input = block_reshape(input, Qinput, args.row_blocksize_fa, args.col_blocksize_fa)

        grad_output_flatten = grad_output.reshape(-1, C_out)
        input_flatten = input.reshape(-1, C_in)

        if grad_output_flatten.dtype == input_flatten.dtype:
            grad_weight = grad_output_flatten.t().mm(input_flatten)
        else:
            grad_weight = grad_output_flatten.float().t().mm(input_flatten)

        Bgrad_weight = block_cut(grad_weight, args.row_blocksize_bw, args.col_blocksize_bw)
        RQgrad_weight, Qgrad_weight, GWscale = block_quant(
            Bgrad_weight,
            args.symm,
            args.bwbit,
            stochastic=True,
            epsilon=args.epsilon,
            apply_quantize=apply_quantize_bw,
            layer_name=layer_name + "WeightGradient",
        )
        Qgrad_weight = block_reshape(Qgrad_weight, grad_weight, args.row_blocksize_bw, args.col_blocksize_bw)
        RQgrad_weight = block_reshape(RQgrad_weight, grad_weight, args.row_blocksize_bw, args.col_blocksize_bw)

        if args.draw_distribution_backward:
            save_tensor(grad_weight, Qgrad_weight, RQgrad_weight, fb="backward", aw="Weight", layer_name=layer_name)

        # Calculate Weight Gradient
        Bweight = block_cut(Qweight, args.row_blocksize_fw, args.col_blocksize_fw)
        weight = Bweight * Wscale
        weight = block_reshape(weight, Qweight, args.row_blocksize_fw, args.col_blocksize_fw)

        if grad_output_flatten.dtype == Qweight.dtype:
            grad_input = grad_output_flatten.mm(weight)
        else:
            grad_input = grad_output_flatten.float().mm(weight)

        Bgrad_input = block_cut(grad_input, args.row_blocksize_ba, args.col_blocksize_ba)
        RQgrad_input, Qgrad_input, GIscale = block_quant(
            Bgrad_input,
            args.symm,
            args.babit,
            stochastic=True,
            epsilon=args.epsilon,
            apply_quantize=apply_quantize_ba,
            layer_name=layer_name + "ActivationGradient",
        )
        Qgrad_input = block_reshape(Qgrad_input, grad_input, args.row_blocksize_ba, args.col_blocksize_ba)
        RQgrad_input = block_reshape(RQgrad_input, grad_input, args.row_blocksize_ba, args.col_blocksize_ba)

        if args.draw_distribution_backward:
            save_tensor(
                grad_input, Qgrad_input, RQgrad_input, fb="backward out", aw="Activation out", layer_name=layer_name
            )

        # enlarge Qgrad_input to let the size of gradient the same as forward
        ideal_scale_num = Qgrad_input.numel() / (args.min_blockunit_row * args.min_blockunit_col)
        actual_scale_num = calculate_scale_num(Qgrad_input, args.row_blocksize_ba, args.col_blocksize_ba)
        # actual_scale_num = Qgrad_input.numel() / (args.row_blocksize_ba * args.col_blocksize_ba)
        assert GIscale.shape[0] == actual_scale_num
        GIscale = torch.nn.functional.pad(GIscale, (0, 0, 0, 0, 0, int(ideal_scale_num - actual_scale_num)))

        Qgrad_input_transform = Qgrad_input.reshape(Qinput.size())

        if bias is not None:
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None

        return Qgrad_input_transform, GIscale, RQgrad_weight, grad_bias, None, None, None, None, None, None


