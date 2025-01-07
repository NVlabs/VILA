import os
import time
from copy import deepcopy

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function, InplaceFunction
from torch.cuda import amp

from .language_model.configuration_quantize import QuantizationConfig
from .qfunction import block_cut, block_quant, block_reshape
from .qutils import quant_get_local_rank
from .realquantize.division_transpose import fp8_division_transpose
from .realquantize.linear import fp8_linear_backward, fp8_linear_forward
from .realquantize.quantize_and_transpose import fp8_quantize_and_transpose


class QLinearTE(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, args=None, layer_idx=0):
        super().__init__(in_features, out_features, bias, device)
        try:  # TODO: remove this try except (llama & qwen2)
            self.args = QuantizationConfig(**deepcopy(args))
        except:
            self.args = deepcopy(args)

        self.apply_quantize = min(self.weight.shape[0], self.weight.shape[1]) >= 3584

        if quant_get_local_rank() == 0:
            if self.apply_quantize:
                print(f"[qlinear debug] Apply QLinear, {layer_idx}")
            else:
                print(f"[qlinear debug] Don't QLinear, {layer_idx} since the weight is too small: {self.weight.shape}")
        self.layer_idx = layer_idx
        self.layer_name = None

    def forward(self, Input):
        # if torch.randn(1) < 0.01:
        #     print(Input.shape, self.weight.shape)
        if self.training and self.apply_quantize:
            # if False:
            output = QuantLinearTE.apply(Input, self.weight, self.bias, self.args, self.layer_name)
        else:
            output = F.linear(Input, self.weight, self.bias)

        return output


# if int(os.environ.get("LOCAL_RANK")) == 0:
#     import IPython
#     IPython.embed()
# else:
#     import time
#     time.sleep(1000)

# class QuantLinearTE(Function):
#     @staticmethod
#     def forward(ctx, input, weight, bias, args, layer_type):
#         ctx.saved = input, weight, bias, args, layer_type
#         return F.linear(input, weight, bias)

#     @staticmethod
#     def backward(ctx, grad_output):
#         input, weight, bias, args, layer_type = ctx.saved

#         C_in = input.shape[-1]
#         C_out = grad_output.shape[-1]

#         grad_output_flatten = grad_output.reshape(-1, C_out)
#         input_flatten = input.reshape(-1, C_in)

#         if grad_output_flatten.dtype == input_flatten.dtype:
#             grad_weight = grad_output_flatten.t().mm(input_flatten)
#         else:
#             grad_weight = grad_output_flatten.float().t().mm(input_flatten)

#         if grad_output_flatten.dtype == weight.dtype:
#             grad_input = grad_output_flatten.mm(weight)
#         else:
#             grad_input = grad_output_flatten.float().mm(weight)

#         if bias is not None:
#             grad_bias = grad_output_flatten.sum(0)
#         else:
#             grad_bias = None

#         grad_input_transform = grad_input.reshape(input.size())

#         return grad_input_transform, grad_weight, grad_bias, None, None


class QuantLinearTE(Function):
    @staticmethod
    @amp.custom_fwd(cast_inputs=torch.bfloat16)
    def forward(ctx, input, weight, bias, args, layer_name):

        time_bench = os.getenv("TIME_BENCH")

        if time_bench:
            start_1 = torch.cuda.Event(enable_timing=True)
            start_1.record()

        # Qinput, Iscale, Qinput_t = fp8_division_transpose(input, 16, args.fabit)
        Qinput, Iscale, Qinput_t = fp8_quantize_and_transpose(input, 16, args.fabit, transpose_output_2d=True)

        if time_bench:
            end_1 = torch.cuda.Event(enable_timing=True)
            end_1.record()
            start_2 = torch.cuda.Event(enable_timing=True)
            start_2.record()

        # Qweight, Wscale, Qweight_t = fp8_division_transpose(weight, 16, args.fwbit)
        Qweight, Wscale, Qweight_t = fp8_quantize_and_transpose(weight, 16, args.fwbit, transpose_output_2d=True)

        if time_bench:
            end_2 = torch.cuda.Event(enable_timing=True)
            end_2.record()
            start_3 = torch.cuda.Event(enable_timing=True)
            start_3.record()

        ctx.saved = Qinput_t, Iscale, Qweight_t, Wscale, bias, args, layer_name
        fc_output = fp8_linear_forward(Qinput, Iscale, Qweight, Wscale, False, 0, bias)

        if time_bench:
            end_3 = torch.cuda.Event(enable_timing=True)
            end_3.record()
            start_4 = torch.cuda.Event(enable_timing=True)
            start_4.record()

            output = F.linear(input, weight, bias)

            end_4 = torch.cuda.Event(enable_timing=True)
            end_4.record()

            torch.cuda.synchronize()
            if quant_get_local_rank() == 0:
                print(
                    f"[Forward] Part 1: {start_1.elapsed_time(end_1):.6f} ms | Part 2: {start_2.elapsed_time(end_2):.6f} ms | Part 3: {start_3.elapsed_time(end_3):.6f} ms | "
                    f"FP8: {start_1.elapsed_time(end_3):.6f} | BF16: {start_4.elapsed_time(end_4):.6f} | Input shape: {input.shape} | Weight shape: {weight.shape}"
                )

        return fc_output

    @staticmethod
    @amp.custom_bwd
    def backward(ctx, grad_output):
        Qinput_t, Iscale, Qweight_t, Wscale, bias, args, layer_name = ctx.saved

        time_bench = os.getenv("TIME_BENCH")
        if time_bench:
            start_1 = torch.cuda.Event(enable_timing=True)
            start_1.record()

        # Qgrad_output, Gscale, Qgrad_output_t = fp8_division_transpose(grad_output, 16, args.bobit, stochastic=False)
        Qgrad_output, Gscale, Qgrad_output_t = fp8_quantize_and_transpose(
            grad_output, 16, args.bobit, stochastic=False, transpose_output_2d=True
        )

        if time_bench:
            end_1 = torch.cuda.Event(enable_timing=True)
            end_1.record()
            start_2 = torch.cuda.Event(enable_timing=True)
            start_2.record()

        grad_input, grad_weight = fp8_linear_backward(
            Qinput_t,
            Iscale,
            Qgrad_output,
            Gscale,
            Qgrad_output_t,
            Qweight_t,
            Wscale,
            16,
            bias,
            stochastic=False,
            dgrad_quantize=False,
        )

        if time_bench:
            end_2 = torch.cuda.Event(enable_timing=True)
            end_2.record()
            start_3 = torch.cuda.Event(enable_timing=True)
            start_3.record()

        if bias is not None:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(0)
        else:
            grad_bias = None

        if time_bench:
            end_3 = torch.cuda.Event(enable_timing=True)
            end_3.record()

            # ========== BF16 ==========
            C_in = Qinput_t.shape[0]
            C_out = grad_output.shape[-1]
            grad_output_flatten = grad_output.reshape(-1, C_out)
            input_flatten = Qinput_t.t().reshape(-1, C_in).to(torch.bfloat16)
            weight = Qweight_t.t().to(torch.bfloat16)

            start_4 = torch.cuda.Event(enable_timing=True)
            start_4.record()

            if grad_output_flatten.dtype == input_flatten.dtype:
                _grad_weight = grad_output_flatten.t().mm(input_flatten)
            else:
                _grad_weight = grad_output_flatten.float().t().mm(input_flatten)

            if grad_output_flatten.dtype == weight.dtype:
                _grad_input = grad_output_flatten.mm(weight)
            else:
                _grad_input = grad_output_flatten.float().mm(weight)

            end_4 = torch.cuda.Event(enable_timing=True)
            end_4.record()

            torch.cuda.synchronize()
            if quant_get_local_rank() == 0:
                print(
                    f"[Backward] Part 1: {start_1.elapsed_time(end_1):.6f} ms | Part 2: {start_2.elapsed_time(end_2):.6f} ms | Part 3: {start_3.elapsed_time(end_3):.6f} ms | "
                    f"FP8: {start_1.elapsed_time(end_3):.6f} | BF16: {start_4.elapsed_time(end_4):.6f} | Input shape: {Qinput_t.shape} | Weight shape: {weight.shape}"
                )

        return grad_input, grad_weight, grad_bias, None, None


