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


class QAct_FPout(nn.Identity):
    def __init__(self, args, normalize_before=False, layer_type=""):
        super().__init__()
        self.args = deepcopy(args)
        self.normalize_before = normalize_before
        self.layer_type = layer_type
        assert layer_type != "", "layer_type is not defined"
        assert layer_type in qconfig.qact_config, f"{layer_type} not in qact_config"

        self.apply_quantize = list_has_common_element(args.qchoice, qconfig.qact_config[layer_type])
        self.apply_quantize_f, self.apply_quantize_b = self.apply_quantize, self.apply_quantize

        self.refine_rowcol_blocksize()

        self.fbit = self.args.fabit if self.args.fabit else self.Ubit
        self.bbit = self.args.babit if self.args.babit else self.Ubit

        quantize_flag = format_string_with_condition(
            layer_type,
            {"apply-f": self.apply_quantize_f, "apply-b": self.apply_quantize_b},
            self.args.symm,
            self.fbit,
            self.bbit,
            {
                "row-f": self.args.row_blocksize_f,
                "col-f": self.args.col_blocksize_f,
                "row-b": self.args.row_blocksize_b,
                "col-b": self.args.col_blocksize_b,
            },
        )
        if quant_get_local_rank() == 0:
            print(quantize_flag)

    def refine_rowcol_blocksize(self):
        self.args.row_blocksize_f, self.args.col_blocksize_f = self.args.row_blocksize, self.args.col_blocksize
        self.args.row_blocksize_b, self.args.col_blocksize_b = self.args.row_blocksize, self.args.col_blocksize
        if self.args.refine_residual_fp:
            if self.layer_type in ["add_attn_in_re", "add_mlp_in_re"]:
                self.apply_quantize_f, self.apply_quantize_b = False, False

        if self.args.refine_ln_blocksize:
            if self.layer_type in ["ln_attn_in"]:
                if self.args.refine_ln_pertoken:
                    self.args.row_blocksize_f, self.args.col_blocksize_f = (
                        1,
                        self.args.refine_row_blocksize * self.args.refine_col_blocksize,
                    )
                    self.args.row_blocksize_b, self.args.col_blocksize_b = (
                        1,
                        self.args.refine_row_blocksize * self.args.refine_col_blocksize,
                    )
                else:
                    self.args.row_blocksize_f, self.args.col_blocksize_f = (
                        self.args.refine_row_blocksize,
                        self.args.refine_col_blocksize,
                    )
                    self.args.row_blocksize_b, self.args.col_blocksize_b = (
                        self.args.refine_row_blocksize,
                        self.args.refine_col_blocksize,
                    )

                assert not (
                    self.args.refine_ln_blocksize_but_only_forward and self.args.refine_ln_blocksize_but_only_backward
                )  # This will not happen at the same time
                if self.args.refine_ln_blocksize_but_only_forward:
                    self.apply_quantize_f, self.apply_quantize_b = True, False
                if self.args.refine_ln_blocksize_but_only_backward:
                    self.apply_quantize_f, self.apply_quantize_b = False, True

            if self.layer_type in [
                "ln_mlp_in",
            ]:
                if self.args.refine_ln_pertoken:
                    self.args.row_blocksize_f, self.args.col_blocksize_f = (
                        1,
                        self.args.refine_row_blocksize * self.args.refine_col_blocksize,
                    )
                    self.args.row_blocksize_b, self.args.col_blocksize_b = (
                        1,
                        self.args.refine_row_blocksize * self.args.refine_col_blocksize,
                    )
                else:
                    self.args.row_blocksize_f, self.args.col_blocksize_f = (
                        self.args.refine_row_blocksize,
                        self.args.refine_col_blocksize,
                    )
                    self.args.row_blocksize_b, self.args.col_blocksize_b = (
                        self.args.refine_row_blocksize,
                        self.args.refine_col_blocksize,
                    )

                assert not (
                    self.args.refine_ln_blocksize_but_only_forward and self.args.refine_ln_blocksize_but_only_backward
                )  # This will not happen at the same time
                if self.args.refine_ln_blocksize_but_only_forward:
                    self.apply_quantize_f, self.apply_quantize_b = True, False
                if self.args.refine_ln_blocksize_but_only_backward:
                    self.apply_quantize_f, self.apply_quantize_b = False, True

        if self.args.refine_attn_blocksize:
            if self.layer_type in ["ln_attn_in"]:
                if self.args.refine_ln_pertoken:
                    self.args.row_blocksize_f, self.args.col_blocksize_f = (
                        1,
                        self.args.refine_row_blocksize * self.args.refine_col_blocksize,
                    )
                    self.args.row_blocksize_b, self.args.col_blocksize_b = (
                        1,
                        self.args.refine_row_blocksize * self.args.refine_col_blocksize,
                    )
                else:
                    self.args.row_blocksize_f, self.args.col_blocksize_f = (
                        self.args.refine_row_blocksize,
                        self.args.refine_col_blocksize,
                    )
                    self.args.row_blocksize_b, self.args.col_blocksize_b = (
                        self.args.refine_row_blocksize,
                        self.args.refine_col_blocksize,
                    )

            if self.layer_type in ["attn_qkv_sum"]:
                self.args.row_blocksize_b, self.args.col_blocksize_b = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )
            if self.layer_type in ["add_attn_in_fx"]:
                self.args.row_blocksize_f, self.args.col_blocksize_f = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )

        if self.args.refine_mlp_blocksize:
            if self.layer_type in [
                "ln_mlp_in",
            ]:
                if self.args.refine_ln_pertoken:
                    self.args.row_blocksize_f, self.args.col_blocksize_f = (
                        1,
                        self.args.refine_row_blocksize * self.args.refine_col_blocksize,
                    )
                    self.args.row_blocksize_b, self.args.col_blocksize_b = (
                        1,
                        self.args.refine_row_blocksize * self.args.refine_col_blocksize,
                    )
                else:
                    self.args.row_blocksize_f, self.args.col_blocksize_f = (
                        self.args.refine_row_blocksize,
                        self.args.refine_col_blocksize,
                    )
                    self.args.row_blocksize_b, self.args.col_blocksize_b = (
                        self.args.refine_row_blocksize,
                        self.args.refine_col_blocksize,
                    )

            if self.layer_type in ["mlp_act_sum"]:
                self.args.row_blocksize_b, self.args.col_blocksize_b = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )
            if self.layer_type in ["mlp_act_in"]:
                self.args.row_blocksize_f, self.args.col_blocksize_f = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )
            if self.layer_type in [
                "mul_act_in1",
            ]:
                self.args.row_blocksize_f, self.args.col_blocksize_f = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )
                self.args.row_blocksize_b, self.args.col_blocksize_b = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )
            if self.layer_type in [
                "mul_act_in2",
            ]:
                self.args.row_blocksize_f, self.args.col_blocksize_f = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )
            if self.layer_type in ["add_mlp_in_fx"]:
                self.args.row_blocksize_f, self.args.col_blocksize_f = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )

    def forward(self, Qinput, Iscale):
        # input shape is (Batch Size, Sequence Length, Hidden Size)
        if self.training:
            return QuantAct_FPout.apply(
                Qinput, Iscale, self.args, self.layer_name, self.apply_quantize_f, self.apply_quantize_b
            )
        else:
            return Qinput


class QuantAct_FPout(Function):
    @staticmethod
    def forward(ctx, Qinput, Iscale, args, layer_name, apply_quantize_f=True, apply_quantize_b=True):
        ctx.saved = args, layer_name, apply_quantize_f, apply_quantize_b

        # shrink Iscale to let the size of gradient the same as forward
        ideal_scale_num = Qinput.numel() / (args.min_blockunit_row * args.min_blockunit_col)
        actual_scale_num = calculate_scale_num(Qinput, args.row_blocksize_f, args.col_blocksize_f)
        # actual_scale_num = Qinput.numel() / (args.row_blocksize_f * args.col_blocksize_f)
        assert Iscale.shape[0] == ideal_scale_num
        Iscale = Iscale[: int(actual_scale_num), :, :]

        Binput = block_cut(Qinput, args.row_blocksize_f, args.col_blocksize_f)
        input = Binput * Iscale
        input = block_reshape(input, Qinput, args.row_blocksize_f, args.col_blocksize_f)

        if args.draw_distribution_forward:
            save_tensor(input, None, None, fb="forward", aw="Activation", layer_name=layer_name)

        return input

    @staticmethod
    def backward(ctx, grad_output):
        args, layer_name, apply_quantize_f, apply_quantize_b = ctx.saved

        Bgrad_output = block_cut(grad_output, args.row_blocksize_b, args.col_blocksize_b)
        RQgrad_output, Qgrad_output, Gscale = block_quant(
            Bgrad_output,
            args.symm,
            args.babit,
            stochastic=True,
            epsilon=args.epsilon,
            apply_quantize=apply_quantize_b,
            layer_name=layer_name,
        )
        Qgrad_output = block_reshape(Qgrad_output, grad_output, args.row_blocksize_b, args.col_blocksize_b)

        if args.draw_distribution_backward:
            save_tensor(grad_output, RQgrad_output, Qgrad_output, fb="backward", aw="Activation", layer_name=layer_name)

        # enlarge grad_output to let the size of gradient the same as forward
        ideal_scale_num = grad_output.numel() / (args.min_blockunit_row * args.min_blockunit_col)
        actual_scale_num = calculate_scale_num(grad_output, args.row_blocksize_b, args.col_blocksize_b)
        # actual_scale_num = grad_output.numel() / (args.row_blocksize_b * args.col_blocksize_b)
        assert Gscale.shape[0] == actual_scale_num
        Gscale = torch.nn.functional.pad(Gscale, (0, 0, 0, 0, 0, int(ideal_scale_num - actual_scale_num)))

        return Qgrad_output, Gscale, None, None, None, None


class QAct_FPin(nn.Identity):
    def __init__(self, args, normalize_before=False, layer_type=""):
        super().__init__()
        self.args = deepcopy(args)
        self.normalize_before = normalize_before
        self.layer_type = layer_type
        assert layer_type != "", "layer_type is not defined"
        assert layer_type in qconfig.qact_config, f"{layer_type} not in qact_config"

        self.apply_quantize = list_has_common_element(args.qchoice, qconfig.qact_config[layer_type])
        self.apply_quantize_f, self.apply_quantize_b = self.apply_quantize, self.apply_quantize

        self.refine_rowcol_blocksize()

        self.fbit = self.args.fabit if self.args.fabit else self.Ubit
        self.bbit = self.args.babit if self.args.babit else self.Ubit

        quantize_flag = format_string_with_condition(
            layer_type,
            {"apply-f": self.apply_quantize_f, "apply-b": self.apply_quantize_b},
            self.args.symm,
            self.fbit,
            self.bbit,
            {
                "row-f": self.args.row_blocksize_f,
                "col-f": self.args.col_blocksize_f,
                "row-b": self.args.row_blocksize_b,
                "col-b": self.args.col_blocksize_b,
            },
        )
        if quant_get_local_rank() == 0:
            print(quantize_flag)

    def refine_rowcol_blocksize(self):
        self.args.row_blocksize_f, self.args.col_blocksize_f = self.args.row_blocksize, self.args.col_blocksize
        self.args.row_blocksize_b, self.args.col_blocksize_b = self.args.row_blocksize, self.args.col_blocksize

        if self.args.refine_residual_fp:
            if self.layer_type in ["re_attn_out_re", "re_mlp_out_re"]:
                self.apply_quantize_f, self.apply_quantize_b = False, False

        if self.args.refine_ln_blocksize:
            if self.layer_type in ["re_attn_out_fx"]:
                if self.args.refine_ln_pertoken:
                    self.args.row_blocksize_f, self.args.col_blocksize_f = (
                        1,
                        self.args.refine_row_blocksize * self.args.refine_col_blocksize,
                    )
                    self.args.row_blocksize_b, self.args.col_blocksize_b = (
                        1,
                        self.args.refine_row_blocksize * self.args.refine_col_blocksize,
                    )
                else:
                    self.args.row_blocksize_f, self.args.col_blocksize_f = (
                        self.args.refine_row_blocksize,
                        self.args.refine_col_blocksize,
                    )
                    self.args.row_blocksize_b, self.args.col_blocksize_b = (
                        self.args.refine_row_blocksize,
                        self.args.refine_col_blocksize,
                    )

                assert not (
                    self.args.refine_ln_blocksize_but_only_forward and self.args.refine_ln_blocksize_but_only_backward
                )  # This will not happen at the same time
                if self.args.refine_ln_blocksize_but_only_forward:
                    self.apply_quantize_f, self.apply_quantize_b = True, False
                if self.args.refine_ln_blocksize_but_only_backward:
                    self.apply_quantize_f, self.apply_quantize_b = False, True

            if self.layer_type in ["re_mlp_out_fx"]:
                if self.args.refine_ln_pertoken:
                    self.args.row_blocksize_f, self.args.col_blocksize_f = (
                        1,
                        self.args.refine_row_blocksize * self.args.refine_col_blocksize,
                    )
                    self.args.row_blocksize_b, self.args.col_blocksize_b = (
                        1,
                        self.args.refine_row_blocksize * self.args.refine_col_blocksize,
                    )
                else:
                    self.args.row_blocksize_f, self.args.col_blocksize_f = (
                        self.args.refine_row_blocksize,
                        self.args.refine_col_blocksize,
                    )
                    self.args.row_blocksize_b, self.args.col_blocksize_b = (
                        self.args.refine_row_blocksize,
                        self.args.refine_col_blocksize,
                    )

                assert not (
                    self.args.refine_ln_blocksize_but_only_forward and self.args.refine_ln_blocksize_but_only_backward
                )  # This will not happen at the same time
                if self.args.refine_ln_blocksize_but_only_forward:
                    self.apply_quantize_f, self.apply_quantize_b = True, False
                if self.args.refine_ln_blocksize_but_only_backward:
                    self.apply_quantize_f, self.apply_quantize_b = False, True

        if self.args.refine_attn_blocksize:
            if self.layer_type in ["re_attn_out_fx"]:
                if self.args.refine_ln_pertoken:
                    self.args.row_blocksize_f, self.args.col_blocksize_f = (
                        1,
                        self.args.refine_row_blocksize * self.args.refine_col_blocksize,
                    )
                    self.args.row_blocksize_b, self.args.col_blocksize_b = (
                        1,
                        self.args.refine_row_blocksize * self.args.refine_col_blocksize,
                    )
                else:
                    self.args.row_blocksize_f, self.args.col_blocksize_f = (
                        self.args.refine_row_blocksize,
                        self.args.refine_col_blocksize,
                    )
                    self.args.row_blocksize_b, self.args.col_blocksize_b = (
                        self.args.refine_row_blocksize,
                        self.args.refine_col_blocksize,
                    )

            if self.layer_type in ["ln_attn_out"]:
                self.args.row_blocksize_b, self.args.col_blocksize_b = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )
            if self.layer_type in ["attn_q_in", "attn_k_in", "attn_v_in"]:
                self.args.row_blocksize_b, self.args.col_blocksize_b = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )

        if self.args.refine_mlp_blocksize:
            if self.layer_type in ["re_mlp_out_fx"]:
                if self.args.refine_ln_pertoken:
                    self.args.row_blocksize_f, self.args.col_blocksize_f = (
                        1,
                        self.args.refine_row_blocksize * self.args.refine_col_blocksize,
                    )
                    self.args.row_blocksize_b, self.args.col_blocksize_b = (
                        1,
                        self.args.refine_row_blocksize * self.args.refine_col_blocksize,
                    )
                else:
                    self.args.row_blocksize_f, self.args.col_blocksize_f = (
                        self.args.refine_row_blocksize,
                        self.args.refine_col_blocksize,
                    )
                    self.args.row_blocksize_b, self.args.col_blocksize_b = (
                        self.args.refine_row_blocksize,
                        self.args.refine_col_blocksize,
                    )

            if self.layer_type in ["ln_mlp_out"]:
                self.args.row_blocksize_b, self.args.col_blocksize_b = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )
            if self.layer_type in ["mlp_act_gate", "mlp_act_up", "mul_act_out"]:
                self.args.row_blocksize_b, self.args.col_blocksize_b = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )
            if self.layer_type in ["mlp_act_out"]:
                self.args.row_blocksize_f, self.args.col_blocksize_f = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )
                self.args.row_blocksize_b, self.args.col_blocksize_b = (
                    self.args.refine_row_blocksize,
                    self.args.refine_col_blocksize,
                )

    def forward(self, input):
        # input shape is (Batch Size, Sequence Length, Hidden Size)
        if self.training:
            return QuantAct_FPin.apply(input, self.args, self.layer_name, self.apply_quantize_f, self.apply_quantize_b)
        else:
            return input, None


class QuantAct_FPin(Function):
    @staticmethod
    def forward(ctx, input, args, layer_name, apply_quantize_f=True, apply_quantize_b=True):
        ctx.saved = args, layer_name, apply_quantize_f, apply_quantize_b

        Binput = block_cut(input, args.row_blocksize_f, args.col_blocksize_f)
        RQinput, Qinput, Iscale = block_quant(
            Binput,
            args.symm,
            args.fabit,
            stochastic=False,
            epsilon=args.epsilon,
            apply_quantize=apply_quantize_f,
            layer_name=layer_name,
        )
        Qinput = block_reshape(Qinput, input, args.row_blocksize_f, args.col_blocksize_f)
        RQinput = block_reshape(RQinput, input, args.row_blocksize_f, args.col_blocksize_f)

        if args.draw_distribution_forward:
            save_tensor(input, RQinput, Qinput, fb="forward", aw="Activation", layer_name=layer_name)

        # enlarge Iscale to let the size of gradient the same as forward
        ideal_scale_num = input.numel() / (args.min_blockunit_row * args.min_blockunit_col)
        actual_scale_num = calculate_scale_num(input, args.row_blocksize_f, args.col_blocksize_f)
        # actual_scale_num = input.numel() / (args.row_blocksize_f * args.col_blocksize_f)
        assert Iscale.shape[0] == actual_scale_num
        Iscale = torch.nn.functional.pad(Iscale, (0, 0, 0, 0, 0, int(ideal_scale_num - actual_scale_num)))

        return Qinput, Iscale

    @staticmethod
    def backward(ctx, Qgrad_output, Gscale):
        args, layer_name, apply_quantize_f, apply_quantize_b = ctx.saved

        # shrink Gscale to let the size of gradient the same as forward
        ideal_scale_num = Qgrad_output.numel() / (args.min_blockunit_row * args.min_blockunit_col)
        actual_scale_num = calculate_scale_num(Qgrad_output, args.row_blocksize_b, args.col_blocksize_b)
        # actual_scale_num = Qgrad_output.numel() / (args.row_blocksize_b * args.col_blocksize_b)
        assert Gscale.shape[0] == ideal_scale_num
        Gscale = Gscale[: int(actual_scale_num), :, :]

        Bgrad_output = block_cut(Qgrad_output, args.row_blocksize_b, args.col_blocksize_b)
        grad_output = Bgrad_output * Gscale
        grad_output = block_reshape(grad_output, Qgrad_output, args.row_blocksize_b, args.col_blocksize_b)

        if args.draw_distribution_backward:
            save_tensor(grad_output, None, None, fb="backward", aw="Activation", layer_name=layer_name)

        return grad_output, None, None, None, None


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


