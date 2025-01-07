# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Adapted from
[MosaiclML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git)
"""

from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import Callable, Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple, cast

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from olmo.aliases import PathOrStr
from olmo.beam_search import BeamSearch, Constraint, FinalSequenceScorer, Sampler
from olmo.config import (
    ActivationCheckpointingStrategy,
    ActivationType,
    BlockType,
    CheckpointType,
    FSDPWrapStrategy,
    InitFnType,
    LayerNormType,
    ModelConfig,
    QuantActivationConfig,
    ShardedCheckpointerType,
    TrainConfig,
)
from olmo.exceptions import OLMoConfigurationError
from olmo.initialization import init_normal
from olmo.model import (
    Activation,
    BufferCache,
    Dropout,
    LayerNorm,
    LayerNormBase,
    OLMo,
    OLMoBlock,
    OLMoBlockGroup,
    OLMoGenerateOutput,
    OLMoOutput,
    RMSLayerNorm,
    RotaryEmbedding,
    _non_meta_init_device,
    activation_checkpoint_function,
    alibi_attention_bias,
    causal_attention_bias,
    get_causal_attention_bias,
    should_checkpoint_block,
)
from olmo.torch_util import ensure_finite_, get_cumulative_document_lengths
from torch import einsum

from ..real_quantization import (
    Coat_quantize_bgn,
    Coat_quantize_end,
    fp8_add_Ifp_Ifp_Ofp_Og16,
    fp8_add_Ifp_Ifp_Ofp_Opt,
    fp8_division,
    fp8_division_transpose,
    fp8_gelu_backward,
    fp8_gelu_forward,
    fp8_layernorm_noparam_backward,
    fp8_layernorm_noparam_forward,
    fp8_linear_backward,
    fp8_linear_forward,
    fp8_mul_backward,
    fp8_mul_forward,
    fp8_quantize,
    fp8_quantize_pertensor,
    fp8_quantize_pertensor_transpose,
    fp8_rmsnorm_backward,
    fp8_rmsnorm_forward,
    fp8_silu_backward,
    fp8_silu_forward,
    fp8_transpose,
)
from ._fp8_weightcache import FP8CacheWeightModule
from ._fp8manager import FP8Manager

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")

__all__ = [
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm",
    "RotaryEmbedding",
    "Activation",
    "GELU",
    "ReLU",
    "SwiGLU",
    "OLMoBlock",
    "OLMoSequentialBlock",
    "OLMo",
    "OLMoOutput",
    "OLMoGenerateOutput",
]


log = logging.getLogger(__name__)


class CoatOLMoBeforeAttentionResidual(FP8CacheWeightModule):
    """
    This is a typical transformer attention module that contains (1) Residual (2) LayerNorm / RMSNorm (3) 1 * Linear layers
    """

    def __init__(self, config: ModelConfig, qargs: QuantActivationConfig, layer_id, fused_dims: tuple):
        super().__init__(config, qargs, layer_id)

        self.qargs = qargs
        self.fwobits = {
            "fabit": self.qargs.fabit,
            "fwbit": self.qargs.fwbit,
            "fobit": self.qargs.fobit,
            "babit": self.qargs.babit,
            "bwbit": self.qargs.bwbit,
            "bobit": self.qargs.bobit,
        }
        self.ln_normalized_shape = config.d_model
        self.att_proj = nn.Linear(config.d_model, sum(fused_dims), bias=config.include_bias, device=config.init_device)

        self.attn_norm = LayerNorm.build(config)

    def forward(self, re_x, x, s):
        if self.training:
            if self.qargs.weight_memory_efficient:
                # Prepare
                with torch.no_grad():
                    weight1_s = self.prepare_weight(self.att_proj.weight, "att_proj", FP8Manager.is_first_microbatch)
                return _CoatOLMoBeforeAttentionResidual.apply(
                    re_x,
                    x,
                    s,
                    self.att_proj.weight,
                    None,
                    None,
                    weight1_s,
                    self.qargs.group_size,
                    self.fwobits,
                    self.layer_id,
                    self.config,
                    self.qargs,
                )
            else:
                # Prepare
                with torch.no_grad():
                    weight1, weight1_t, weight1_s = self.prepare_weight(
                        self.att_proj.weight, "att_proj", FP8Manager.is_first_microbatch
                    )
                return _CoatOLMoBeforeAttentionResidual.apply(
                    re_x,
                    x,
                    s,
                    self.att_proj.weight,
                    weight1,
                    weight1_t,
                    weight1_s,
                    self.qargs.group_size,
                    self.fwobits,
                    self.layer_id,
                    self.config,
                    self.qargs,
                )
        else:
            return re_x, self.att_proj(self.attn_norm(re_x))


class _CoatOLMoBeforeAttentionResidual(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        re_x,
        in_x,
        in_s,
        weight1_origin,
        weight1,
        weight1_t,
        weight1_s,
        group_size,
        fwobits,
        layer_id,
        config,
        qargs,
        eps=1e-5,
    ):
        # for autograd
        if fwobits["fabit"] == "E4M3":
            # in_x = in_x.to(torch.float8_e4m3fn)
            in_x = in_x.view(torch.float8_e4m3fn)
        else:
            raise ValueError("fabit should be E4M3")

        # LayerNorm
        ln_x, ln_s, ln_x_t, ln_utils = fp8_layernorm_noparam_forward(
            in_x, in_s, group_size, eps, transpose_output_2d=True
        )

        # Linear Layer QKV Projection
        if qargs.weight_memory_efficient:
            assert weight1 is None  # memory efficient
            weight1, weight1_s = fp8_division(weight1_origin, qargs.group_size, fwobits["fwbit"], weight1_s)
        fc1_x = fp8_linear_forward(ln_x, ln_s, weight1, weight1_s, False, group_size)

        # ==================== save for backward ====================
        ctx.save_for_backward(in_x, in_s, ln_x_t, ln_s)
        if qargs.weight_memory_efficient:
            assert weight1_t is None
            ctx.weight = weight1_origin, weight1_s
        else:
            ctx.weight = weight1_t, weight1_s
        ctx.group_size = group_size
        ctx.ln_utils = ln_utils
        ctx.utils = fwobits, layer_id, config, qargs

        return re_x, fc1_x

    @staticmethod
    def backward(ctx, fp_grad, flash_g):
        in_x, in_s, ln_x_t, ln_s = ctx.saved_tensors
        weight1_t, weight1_s = ctx.weight
        group_size = ctx.group_size
        mean, rstd, num_warps = ctx.ln_utils
        fwobits, layer_id, config, qargs = ctx.utils

        # ==================== Begin backward ====================
        # Quantize the RoPE and FlashAttention Output. grad_input and grad_weight requires different data layout.
        flash_g, flash_gs, flash_g_t = fp8_quantize_pertensor_transpose(
            flash_g, group_size, fwobits["babit"], transpose_output_2d=True, stochastic=False
        )

        # Linear Layer QKV Projection
        if qargs.weight_memory_efficient:
            weight1_t, weight1_s = fp8_division_transpose(
                weight1_t, qargs.group_size, fwobits["fwbit"], weight1_s, only_transposed=True
            )
        fc1_g, att_proj_wg = fp8_linear_backward(
            ln_x_t, ln_s, flash_g, flash_gs, flash_g_t, weight1_t, weight1_s, group_size
        )

        # LayerNorm
        in_g = fp8_layernorm_noparam_backward(in_x, in_s, fc1_g, group_size, mean, rstd, num_warps)

        # Add the gradient together, and prepare the input of the next layer.
        re_g, (in_g, in_sg, in_sg_g16) = fp8_add_Ifp_Ifp_Ofp_Opt(
            fp_grad, in_g, group_size, fwobits["babit"], stochastic=False
        )

        # for autograd. forward's data type should be the same of backward tensor. this will not change the actual binary representation.
        in_g = in_g.view(torch.float8_e4m3fn)

        # Although the next operator is a linear layer in MLPResidual module, we return in_sg_g16 to make the size compatible with the forward. Otherwise it will not pass autograd.
        return re_g, in_g, in_sg_g16, att_proj_wg, None, None, None, None, None, None, None, None, None


class CoatOLMoAfterAttentionResidual(FP8CacheWeightModule):
    """
    This is a typical transformer attention module that contains (1) Residual (2) 1 * Linear layers
    """

    def __init__(self, config: ModelConfig, qargs: QuantActivationConfig, layer_id):
        super().__init__(config, qargs, layer_id)

        self.qargs = qargs
        self.fwobits = {
            "fabit": self.qargs.fabit,
            "fwbit": self.qargs.fwbit,
            "fobit": self.qargs.fobit,
            "babit": self.qargs.babit,
            "bwbit": self.qargs.bwbit,
            "bobit": self.qargs.bobit,
        }
        self.attn_out = nn.Linear(config.d_model, config.d_model, bias=config.include_bias, device=config.init_device)

    def forward(self, re_x, in_x):
        if self.training:
            if self.qargs.weight_memory_efficient:
                # prepare for the weight
                with torch.no_grad():
                    weight2_s = self.prepare_weight(self.attn_out.weight, "attn_out", FP8Manager.is_first_microbatch)

                return _CoatOLMoAfterAttentionResidual.apply(
                    re_x,
                    in_x,
                    self.attn_out.weight,
                    None,
                    None,
                    weight2_s,
                    self.qargs.group_size,
                    self.fwobits,
                    self.layer_id,
                    self.config,
                    self.qargs,
                )
            else:
                # prepare for the weight
                with torch.no_grad():
                    weight2, weight2_t, weight2_s = self.prepare_weight(
                        self.attn_out.weight, "attn_out", FP8Manager.is_first_microbatch
                    )

                return _CoatOLMoAfterAttentionResidual.apply(
                    re_x,
                    in_x,
                    self.attn_out.weight,
                    weight2,
                    weight2_t,
                    weight2_s,
                    self.qargs.group_size,
                    self.fwobits,
                    self.layer_id,
                    self.config,
                    self.qargs,
                )
        else:
            return re_x + self.attn_out(in_x), None, None


class _CoatOLMoAfterAttentionResidual(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, re_x, flash_x, weight2_origin, weight2, weight2_t, weight2_s, group_size, fwobits, layer_id, config, qargs
    ):
        # Quantize the FlashAttention Output
        flash_qx, flash_s, _ = fp8_quantize_pertensor(
            flash_x, group_size, fwobits["fabit"]
        )  # Modified to make it memory efficient

        # # Attention Projection Linear Layer
        if qargs.weight_memory_efficient:
            assert weight2 is None  # memory efficient
            weight2, weight2_s = fp8_division(weight2_origin, qargs.group_size, fwobits["fwbit"], weight2_s)
        fc2_x = fp8_linear_forward(flash_qx, flash_s, weight2, weight2_s, False, group_size)  #

        # import IPython
        # IPython.embed()
        # Add the activations together
        fp_x, (out_x, out_s) = fp8_add_Ifp_Ifp_Ofp_Og16(re_x, fc2_x, flash_qx.dtype, group_size)

        # ==================== save for backward ====================
        ctx.save_for_backward(flash_x, flash_s)
        if qargs.weight_memory_efficient:
            assert weight2_t is None
            ctx.weight = weight2_origin, weight2_s
        else:
            ctx.weight = weight2_t, weight2_s
        ctx.group_size = group_size
        ctx.fwobits = fwobits
        ctx.utils = fwobits, layer_id, config, qargs

        # For autograd
        out_x = out_x.view(torch.float8_e4m3fn)

        return fp_x, out_x, out_s

    @staticmethod
    def backward(ctx, fp_grad, out_g, out_gs):
        flash_x, flash_s = ctx.saved_tensors
        weight2_t, weight2_s = ctx.weight
        group_size = ctx.group_size
        fwobits = ctx.fwobits
        fwobits, layer_id, config, qargs = ctx.utils

        # for autograd
        if fwobits["babit"] == "E5M2":
            # out_g = out_g.to(torch.float8_e5m2)
            out_g = out_g.view(torch.float8_e5m2)
        else:
            raise ValueError("babit should be E5M2")
        out_gs_max = out_gs.max()

        # ==================== Begin backward ====================
        # Output Projection
        out_g_t = fp8_transpose(out_g, transpose_output_2d=True)

        # We do not save an extra flash_x to save the memory usage
        flash_x_t, flash_s = fp8_division_transpose(
            flash_x, group_size, fwobits["fabit"], flash_s, stochastic=False, only_transposed=True
        )

        if qargs.weight_memory_efficient:
            weight2_t, weight2_s = fp8_division_transpose(
                weight2_t, qargs.group_size, fwobits["fwbit"], weight2_s, only_transposed=True
            )
        fc2_g, attn_out_wg = fp8_linear_backward(
            flash_x_t, flash_s, out_g, out_gs_max, out_g_t, weight2_t, weight2_s, group_size
        )

        return fp_grad, fc2_g, attn_out_wg, None, None, None, None, None, None, None, None


class CoatOLMoMLPResidual(FP8CacheWeightModule):
    """
    This is a typical transformer attention module that contains (1) Residual (2) LayerNorm / RMSNorm (3) 2 / 3 * Linear layers
    (4) GELU / Silu Activation
    """

    def __init__(self, config: ModelConfig, qargs: QuantActivationConfig, layer_id, hidden_size: int):
        super().__init__(config, qargs, layer_id)

        self.qargs = qargs
        self.fwobits = {
            "fabit": self.qargs.fabit,
            "fwbit": self.qargs.fwbit,
            "fobit": self.qargs.fobit,
            "babit": self.qargs.babit,
            "bwbit": self.qargs.bwbit,
            "bobit": self.qargs.bobit,
        }
        self.ln_normalized_shape = config.d_model
        self.act_output_multiplier = 0.5 if config.activation_type == ActivationType.swiglu else 1
        self.ff_proj = nn.Linear(config.d_model, hidden_size, bias=config.include_bias, device=config.init_device)
        self.ff_out = nn.Linear(
            int(self.act_output_multiplier * hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.training = True

        # below is only used when training = False
        self.ff_norm = LayerNorm.build(config)
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * hidden_size) % 1 == 0

    def forward(self, re_x, x, s):
        if self.training:
            if self.qargs.weight_memory_efficient:  # prepare for the weight
                with torch.no_grad():
                    weight1_s = self.prepare_weight(self.ff_proj.weight, "ff_proj", FP8Manager.is_first_microbatch)
                    weight2_s = self.prepare_weight(self.ff_out.weight, "ff_out", FP8Manager.is_first_microbatch)

                return _CoatOLMoMLPResidual.apply(
                    re_x,
                    x,
                    s,
                    self.ff_proj.weight,
                    None,
                    None,
                    weight1_s,
                    self.ff_out.weight,
                    None,
                    None,
                    weight2_s,
                    self.qargs.group_size,
                    self.fwobits,
                    self.layer_id,
                    self.config,
                    self.qargs,
                )
            else:
                # prepare for the weight
                with torch.no_grad():
                    weight1, weight1_t, weight1_s = self.prepare_weight(
                        self.ff_proj.weight, "ff_proj", FP8Manager.is_first_microbatch
                    )
                    weight2, weight2_t, weight2_s = self.prepare_weight(
                        self.ff_out.weight, "ff_out", FP8Manager.is_first_microbatch
                    )

                return _CoatOLMoMLPResidual.apply(
                    re_x,
                    x,
                    s,
                    self.ff_proj.weight,
                    weight1,
                    weight1_t,
                    weight1_s,
                    self.ff_out.weight,
                    weight2,
                    weight2_t,
                    weight2_s,
                    self.qargs.group_size,
                    self.fwobits,
                    self.layer_id,
                    self.config,
                    self.qargs,
                )
        else:
            og_x = re_x
            re_x = self.ff_norm(re_x)
            re_x = self.ff_proj(re_x)
            re_x = self.act(re_x)
            re_x = self.ff_out(re_x)
            re_x = og_x + re_x
            return re_x, None, None


class _CoatOLMoMLPResidual(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        re_x,
        in_x,
        in_s,
        weight1_origin,
        weight1,
        weight1_t,
        weight1_s,
        weight2_origin,
        weight2,
        weight2_t,
        weight2_s,
        group_size,
        fwobits,
        layer_id,
        config,
        qargs,
        eps=1e-5,
    ):
        # For autograd
        if fwobits["fabit"] == "E4M3":
            # in_x = in_x.to(torch.float8_e4m3fn)
            in_x = in_x.view(torch.float8_e4m3fn)
        else:
            raise ValueError("fabit should be E4M3")

        # LayerNorm
        ln_x, ln_s, ln_x_t, ln_utils = fp8_layernorm_noparam_forward(
            in_x, in_s, group_size, eps, transpose_output_2d=True
        )

        # Linear Layer of Up Projection and Gate Projection. They are fused as one linear layer.
        if qargs.weight_memory_efficient:
            assert weight1 is None  # memory efficient
            weight1, weight1_s = fp8_division(weight1_origin, qargs.group_size, fwobits["fwbit"], weight1_s)
        fc1_x, fc1_s = fp8_linear_forward(ln_x, ln_s, weight1, weight1_s, True, group_size)

        # NOTE: Becareful of the order
        up_x, gate_x = fc1_x.chunk(2, dim=-1)
        up_s, gate_s = fc1_s.chunk(2, dim=-1)

        # silu Activation
        silu_x, silu_s = fp8_silu_forward(gate_x, gate_s, group_size)

        # Element-wise Multiplication
        mul_x, mul_s, mul_x_t = fp8_mul_forward(silu_x, silu_s, up_x, up_s, group_size, transpose_output_2d=True)

        # Output Projection
        if weight2 is None:  # memory efficient
            weight2, weight2_s = fp8_division(weight2_origin, qargs.group_size, fwobits["fwbit"], weight2_s)
        fc2_x = fp8_linear_forward(mul_x, mul_s, weight2, weight2_s, False, group_size)

        # Add the activation together
        fp_x, (out_x, out_s) = fp8_add_Ifp_Ifp_Ofp_Og16(re_x, fc2_x, mul_x.dtype, group_size)

        # ==================== save for backward ====================
        ctx.save_for_backward(in_x, in_s, ln_x_t, ln_s, gate_x, gate_s, up_x, up_s, silu_x, silu_s, mul_x_t, mul_s)

        ctx.weight = (weight1_t, weight1_s, weight2_t, weight2_s)
        if (
            qargs.weight_memory_efficient
        ):  # Weight_1/2_origin will not be saved twice, so it will be more memory efficient.
            assert weight1_t is None
            ctx.weight = (weight1_origin, weight1_s, weight2_origin, weight2_s)
        else:  # Weight1/2_t is different from the origin weight, so saving it will consumes additional memory footprint.
            ctx.weight = (weight1_t, weight1_s, weight2_t, weight2_s)

        ctx.group_size = group_size
        ctx.ln_utils = ln_utils
        ctx.utils = fwobits, layer_id, config, qargs

        out_x = out_x.view(torch.float8_e4m3fn)

        return fp_x, out_x, out_s

    @staticmethod
    def backward(ctx, fp_grad, out_g, out_gs):
        fwobits, layer_id, config, qargs = ctx.utils

        in_x, in_s, ln_x_t, ln_s, gate_x, gate_s, up_x, up_s, silu_x, silu_s, mul_x_t, mul_s = ctx.saved_tensors

        (weight1_t, weight1_s, weight2_t, weight2_s) = ctx.weight
        group_size = ctx.group_size
        mean, rstd, num_warps = ctx.ln_utils
        fwobits, layer_id, config, qargs = ctx.utils

        # For autograd
        if fwobits["babit"] == "E5M2":
            # out_g = out_g.to(torch.float8_e5m2)
            out_g = out_g.view(torch.float8_e5m2)
        else:
            raise ValueError("babit should be E5M2")
        out_gs_max = out_gs.max()

        # ==================== Begin backward ====================
        # Output Projection
        out_gs = out_gs.max()
        out_g_t = fp8_transpose(out_g, transpose_output_2d=True)

        if qargs.weight_memory_efficient:
            weight2_t, weight2_s = fp8_division_transpose(
                weight2_t, qargs.group_size, fwobits["fwbit"], weight2_s, only_transposed=True
            )
        fc2_g, weight2_grad = fp8_linear_backward(
            mul_x_t, mul_s, out_g, out_gs_max, out_g_t, weight2_t, weight2_s, group_size
        )

        # [MEM TEST]
        del out_g, out_g_t, weight2_t

        # Element-wise Multiplication, 1 means gate, 2 means up
        mul_g1, (mul_g2, mul_gs2) = fp8_mul_backward(silu_x, silu_s, up_x, up_s, fc2_g, group_size, fwobits["babit"])

        # Silu activation
        silu_g, silu_gs = fp8_silu_backward(gate_x, gate_s, mul_g1, group_size, fwobits["babit"])

        # Prepare the input of Linear Layer. NOTE: Becareful of the order
        gateup_g = torch.cat([mul_g2, silu_g], dim=-1)
        gateup_gs = torch.cat([mul_gs2, silu_gs])
        gateup_gs = torch.max(gateup_gs)

        gateup_g, gateup_gs, gateup_g_t = fp8_division_transpose(
            gateup_g, group_size, fwobits["babit"], gateup_gs, stochastic=False
        )

        # Linear Layer of Up and Gate Projection
        if qargs.weight_memory_efficient:
            weight1_t, weight1_s = fp8_division_transpose(
                weight1_t, group_size, fwobits["fwbit"], weight1_s, only_transposed=True
            )
        fc1_g, weight1_grad = fp8_linear_backward(
            ln_x_t, ln_s, gateup_g, gateup_gs, gateup_g_t, weight1_t, weight1_s, group_size
        )

        # layerNorm
        in_g = fp8_layernorm_noparam_backward(in_x, in_s, fc1_g, group_size, mean, rstd, num_warps)

        # Add the gradient together
        re_g, (in_g, in_sg, in_sg_g16) = fp8_add_Ifp_Ifp_Ofp_Opt(
            fp_grad, in_g, group_size, fwobits["babit"], stochastic=False
        )

        in_g = in_g.view(torch.float8_e4m3fn)

        return (
            re_g,
            in_g,
            in_sg_g16,
            weight1_grad,
            None,
            None,
            None,
            weight2_grad,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class CoatOLMoBlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: ModelConfig, qargs: QuantActivationConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.qargs = qargs
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn: Callable | None = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

        # Layer norms.
        self.k_norm: LayerNormBase | None = None
        self.q_norm: LayerNormBase | None = None
        if config.attention_layer_norm:
            assert config.effective_n_kv_heads is not None
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)

        # Make sure QKV clip coefficient is positive, otherwise it's not well-defined.
        if config.clip_qkv is not None:
            assert config.clip_qkv > 0

        # Activation function.
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        if not self.qargs.use_quantize_model:
            # Attention output projection.
            self.attn_out = nn.Linear(
                config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
            )

            # Feed-forward output projection.
            self.ff_out = nn.Linear(
                int(self.act.output_multiplier * self.hidden_size),
                config.d_model,
                bias=config.include_bias,
                device=config.init_device,
            )
            self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config, self.__cache)

        self.flash_attn_func = None
        self.flash_attn_varlen_func = None
        if config.flash_attention:
            try:
                from flash_attn import flash_attn_func, flash_attn_varlen_func  # type: ignore

                self.flash_attn_func = flash_attn_func
                self.flash_attn_varlen_func = flash_attn_varlen_func
            except ModuleNotFoundError:
                pass

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()

        if not self.qargs.use_quantize_model:
            if self.config.init_fn == InitFnType.normal:
                attn_out_std = ff_out_std = self.config.init_std
                cutoff_factor = self.config.init_cutoff_factor

            elif self.config.init_fn == InitFnType.mitchell:
                attn_out_std = 1 / (math.sqrt(2 * self.config.d_model * (self.layer_id + 1)))
                ff_out_std = 1 / (math.sqrt(2 * self.ff_out.in_features * (self.layer_id + 1)))
                cutoff_factor = self.config.init_cutoff_factor or 3.0

            elif self.config.init_fn == InitFnType.full_megatron:
                attn_out_std = ff_out_std = self.config.init_std / math.sqrt(2.0 * self.config.n_layers)
                cutoff_factor = self.config.init_cutoff_factor or 3.0

            else:
                raise NotImplementedError(self.config.init_fn)

            init_normal(self.attn_out, std=attn_out_std, init_cutoff_factor=cutoff_factor)
            init_normal(self.ff_out, std=ff_out_std, init_cutoff_factor=cutoff_factor)

    def set_activation_checkpointing(
        self, strategy: ActivationCheckpointingStrategy | None, checkpoint_func: Callable | None = None
    ):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = checkpoint_func or activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        max_doc_len: int | None = None,
        cu_doc_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        if max_doc_len is not None and cu_doc_lens is not None:
            assert self.flash_attn_varlen_func is not None, "flash-attn is required for document masking"
            assert attn_mask is None, "attn-mask is currently not supported with document masking"
            B, T, D = q.size(0), q.size(2), q.size(3)
            r = self.flash_attn_varlen_func(
                q.transpose(1, 2).view(B * T, -1, D),
                k.transpose(1, 2).view(B * T, -1, D),
                v.transpose(1, 2).view(B * T, -1, D),
                cu_doc_lens,
                cu_doc_lens,
                max_doc_len,
                max_doc_len,
                dropout_p=dropout_p,
                causal=is_causal,
            )
            return r.view(B, T, -1, D).transpose(1, 2)
        elif self.flash_attn_func is not None and attn_mask is None:
            r = self.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=is_causal
            )
            return r.transpose(1, 2)
        else:
            # torch's sdpa doesn't support GQA, so we're doing this
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        max_doc_len: int | None = None,
        cu_doc_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if self.config.rope:
            # Apply rotary embeddings.
            q, k = self.rotary_emb(q, k)

        if attention_bias is not None:
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            attention_bias = self._cast_attn_bias(attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype)

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=attention_bias is None,
            max_doc_len=max_doc_len,
            cu_doc_lens=cu_doc_lens,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection. NOTE: We move the attn output outside of this attention function
        return att, present

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.FloatTensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        max_doc_len: int | None = None,
        cu_doc_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: ModelConfig, qargs: QuantActivationConfig, cache: BufferCache) -> OLMoBlock:
        if config.block_type == BlockType.sequential:
            return CoatOLMoSequentialBlock(layer_id, config, qargs, cache)
        elif config.block_type == BlockType.llama:
            return CoatOLMoLlamaBlock(layer_id, config, qargs, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class CoatOLMoSequentialBlock(CoatOLMoBlock):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). To compute it as ``LN(MLP(x + LN(Attention(x))))``,
    use the flag `norm_after`.
    """

    def __init__(self, layer_id: int, config: ModelConfig, qargs: QuantActivationConfig, cache: BufferCache):
        super().__init__(layer_id, config, qargs, cache)
        # Attention input projection. Projects x -> (q, k, v)

        assert not self.config.norm_after, "COAT currently does not support PostNorm"

        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )

        if self.qargs.use_quantize_model:
            self.BeforeAttention = CoatOLMoBeforeAttentionResidual(config, qargs, self.layer_id, self.fused_dims)
            self.AfterAttention = CoatOLMoAfterAttentionResidual(config, qargs, self.layer_id)
            self.MLPResidual = CoatOLMoMLPResidual(config, qargs, self.layer_id, self.hidden_size)
        else:
            self.att_proj = nn.Linear(
                config.d_model, sum(self.fused_dims), bias=config.include_bias, device=config.init_device
            )
            # Feed-forward input projection.
            self.ff_proj = nn.Linear(
                config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
            )

        # Layer norms.
        self.attn_norm = LayerNorm.build(config, size=config.d_model)
        self.ff_norm = LayerNorm.build(config, size=config.d_model)

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.

        if self.qargs.use_quantize_model:  # The initialization appears here, not in CoatOLMoBlock's reset_parameters
            if self.config.init_fn == InitFnType.normal:
                attn_out_std = ff_out_std = self.config.init_std
                cutoff_factor = self.config.init_cutoff_factor

            elif self.config.init_fn == InitFnType.mitchell:
                attn_out_std = 1 / (math.sqrt(2 * self.config.d_model * (self.layer_id + 1)))
                ff_out_std = 1 / (math.sqrt(2 * self.MLPResidual.ff_out.in_features * (self.layer_id + 1)))
                cutoff_factor = self.config.init_cutoff_factor or 3.0

            elif self.config.init_fn == InitFnType.full_megatron:
                attn_out_std = ff_out_std = self.config.init_std / math.sqrt(2.0 * self.config.n_layers)
                cutoff_factor = self.config.init_cutoff_factor or 3.0

            else:
                raise NotImplementedError(self.config.init_fn)

            init_normal(self.AfterAttention.attn_out, std=attn_out_std, init_cutoff_factor=cutoff_factor)
            init_normal(self.MLPResidual.ff_out, std=ff_out_std, init_cutoff_factor=cutoff_factor)

        if self.config.init_fn == InitFnType.normal:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor
        elif self.config.init_fn == InitFnType.mitchell:
            std = 1 / math.sqrt(self.config.d_model)
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == InitFnType.full_megatron:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            raise NotImplementedError(self.config.init_fn)

        if not self.qargs.use_quantize_model:
            init_normal(self.att_proj, std, cutoff_factor)
            init_normal(self.ff_proj, std, cutoff_factor)
        else:
            init_normal(self.BeforeAttention.att_proj, std, cutoff_factor)
            init_normal(self.MLPResidual.ff_proj, std, cutoff_factor)

    def forward(
        self,
        x: torch.Tensor,
        qx: torch.Tensor,
        sx: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        max_doc_len: int | None = None,
        cu_doc_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)

        # import IPython
        # IPython.embed()

        if self.qargs.use_quantize_model:
            # if False:
            x, qkv = self.BeforeAttention(x, qx, sx)
        else:
            # apply norm before
            h = self.attn_norm(x)

            qkv = self.BeforeAttention.att_proj(h)

        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        q, k, v = qkv.split(self.fused_dims, dim=-1)

        # Get attention scores.
        att, cache = self.attention(
            q,
            k,
            v,
            attention_bias,
            layer_past=layer_past,
            use_cache=use_cache,
            max_doc_len=max_doc_len,
            cu_doc_lens=cu_doc_lens,
        )

        # import IPython
        # IPython.embed()
        if self.qargs.use_quantize_model:
            # if False:
            x, qx, sx = self.AfterAttention(x, att)
        else:
            att = self.AfterAttention.attn_out(att)

            # Add attention scores.
            # shape: (B, T, C)
            x = x + self.dropout(att)

        if self.qargs.use_quantize_model:
            # if False:
            x, qx, sx = self.MLPResidual(x, qx, sx)
        else:
            # Add feed-forward projection.
            # shape: (batch_size, seq_len, d_model)
            og_x = x

            x = self.ff_norm(x)

            x = self.MLPResidual.ff_proj(x)

            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
            else:
                x = self.act(x)
            x = self.MLPResidual.ff_out(x)

            x = self.dropout(x)
            x = og_x + x

        # import IPython
        # IPython.embed()

        return x, qx, sx, cache


class CoatOLMoLlamaBlock(OLMoBlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `OLMoSequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: ModelConfig, qargs: QuantActivationConfig, cache: BufferCache):
        super().__init__(layer_id, config, qargs, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        if config.multi_query_attention:
            q_proj_out_dim = config.d_model
            k_proj_out_dim = config.d_model // config.n_heads
            v_proj_out_dim = config.d_model // config.n_heads
        else:
            q_proj_out_dim = config.d_model
            k_proj_out_dim = config.d_model
            v_proj_out_dim = config.d_model
        self.q_proj = nn.Linear(config.d_model, q_proj_out_dim, bias=config.include_bias, device=config.init_device)
        self.k_proj = nn.Linear(config.d_model, k_proj_out_dim, bias=config.include_bias, device=config.init_device)
        self.v_proj = nn.Linear(config.d_model, v_proj_out_dim, bias=config.include_bias, device=config.init_device)

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device)

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.

        if self.config.init_fn == InitFnType.normal:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor
        elif self.config.init_fn == InitFnType.mitchell:
            std = 1 / math.sqrt(self.config.d_model)
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == InitFnType.full_megatron:
            std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            raise NotImplementedError(self.config.init_fn)

        init_normal(self.q_proj, std, cutoff_factor)
        init_normal(self.k_proj, std, cutoff_factor)
        init_normal(self.v_proj, std, cutoff_factor)
        init_normal(self.ff_proj, std, cutoff_factor)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        max_doc_len: int | None = None,
        cu_doc_lens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if max_doc_len is not None or cu_doc_lens is not None:
            raise NotImplementedError(f"attention document masking is not implemented for {self.__class__.__name__}")

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if is_causal:
            assert attn_mask is None

            query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None
            attn_bias = get_causal_attention_bias(self.__cache, key_len, q.device)[:, :, :query_len, :key_len]
        elif attn_mask is not None:
            attn_bias = attn_mask.to(q.dtype)
        else:
            attn_bias = torch.zeros_like(attn_weights)

        attn_weights += attn_bias
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(q.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout_p)
        return torch.matmul(attn_weights, v)

    def forward(
        self,
        x: torch.Tensor,
        qx: torch.Tensor,
        sx: torch.Tensor,
        attention_bias: torch.Tensor | None = None,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        max_doc_len: int | None = None,
        cu_doc_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        if self.config.clip_qkv is not None:
            q.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            k.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            v.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        # Get attention scores.
        att, cache = self.attention(
            q,
            k,
            v,
            attention_bias,
            layer_past=layer_past,
            use_cache=use_cache,
            max_doc_len=max_doc_len,
            cu_doc_lens=cu_doc_lens,
        )

        att = self.attn_out(att)  # NOTE: we move the attn_out outside the self.attention module

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


class CoatOLMoBlockGroup(nn.ModuleList):
    def __init__(self, config: ModelConfig, layer_offset: int, modules: Iterable[nn.Module] | None = None):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy: ActivationCheckpointingStrategy | None = None
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: torch.FloatTensor | None = None,
        layers_past: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        max_doc_len: int | None = None,
        cu_doc_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]] | None]:
        attn_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if use_cache else None
        for block_idx, block in enumerate(self):
            layer_past = None if layers_past is None else layers_past[block_idx]
            block_idx += self.layer_offset
            if should_checkpoint_block(self.activation_checkpointing_strategy, block_idx):
                # shape: (batch_size, seq_len, d_model)
                x, cache = self._activation_checkpoint_fn(  # type: ignore
                    block,
                    x,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                    max_doc_len=max_doc_len,
                    cu_doc_lens=cu_doc_lens,
                )
            else:
                # shape: (batch_size, seq_len, d_model)
                x, cache = block(
                    x,
                    attention_bias=attention_bias,
                    layer_past=layer_past,
                    use_cache=use_cache,
                    max_doc_len=max_doc_len,
                    cu_doc_lens=cu_doc_lens,
                )
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        return x, attn_key_values

    def reset_parameters(self):
        for block in self:
            block.reset_parameters()

    def set_activation_checkpointing(
        self, strategy: ActivationCheckpointingStrategy | None, checkpoint_func: Callable | None = None
    ):
        self.activation_checkpointing_strategy = strategy
        for block in self:
            block.set_activation_checkpointing(strategy, checkpoint_func=checkpoint_func)


class CoatOLMo(nn.Module):
    def __init__(self, config: ModelConfig, qargs: QuantActivationConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.qargs = qargs
        self.__cache = BufferCache()

        # Validate config.
        if self.config.alibi and self.config.flash_attention:
            raise OLMoConfigurationError("ALiBi is currently not supported with FlashAttention")

        if self.config.alibi and self.config.rope:
            raise OLMoConfigurationError("ALiBi and RoPE are mutually exclusive")

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise OLMoConfigurationError("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        self.activation_checkpointing_strategy: ActivationCheckpointingStrategy | None = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise OLMoConfigurationError("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.embedding_size or config.vocab_size, config.d_model, device=config.init_device),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )

        blocks = [CoatOLMoBlock.build(i, config, qargs, self.__cache) for i in range(config.n_layers)]
        if self.config.block_group_size > 1:
            block_groups = [
                CoatOLMoBlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
        if config.embedding_layer_norm:
            self.transformer.update({"emb_norm": LayerNorm.build(config)})

        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: int | None = None
        self.__num_bck_flops: int | None = None

        # Warm up cache.
        if self.config.alibi:
            get_causal_attention_bias(self.__cache, config.max_sequence_length, _non_meta_init_device(config))
            self.get_alibi_attention_bias(config.max_sequence_length, _non_meta_init_device(config))

        # Quantize
        self.quantize_input_before_block = Coat_quantize_bgn(qargs)
        self.quantize_output_after_block = Coat_quantize_end(qargs)

    set_activation_checkpointing = OLMo.set_activation_checkpointing
    device = OLMo.device
    reset_parameters = OLMo.reset_parameters
    get_alibi_attention_bias = OLMo.get_alibi_attention_bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: torch.FloatTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        attention_bias: torch.Tensor | None = None,
        past_key_values: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: bool | None = None,
        doc_lens: torch.Tensor | None = None,
        max_doc_lens: Sequence[int] | None = None,
    ) -> OLMoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        :param doc_lens: Document lengths to use in attention for intra-document masking.
            Shape `(batch_size, max_docs)`.
        :param max_doc_lens: Maximum document length for each instance in the batch.
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        max_doc_len: int | None = None
        cu_doc_lens: torch.Tensor | None = None
        if doc_lens is not None and max_doc_lens is not None:
            max_doc_len = max(max_doc_lens)
            cu_doc_lens = get_cumulative_document_lengths(doc_lens)

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

        # Apply embedding layer norm.
        if self.config.embedding_layer_norm:
            x = self.transformer.emb_norm(x)

        if not (self.config.alibi or self.config.rope):
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        # Merge attention mask with attention bias.
        if (
            attention_bias is not None
            or attention_mask is not None
            or self.config.alibi
            # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
            # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
            # scores correctly.
            or past_key_values is not None
        ):
            if attention_bias is None and self.config.alibi:
                attention_bias = get_causal_attention_bias(
                    self.__cache, past_length + seq_len, x.device
                ) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
            elif attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                # it can produce NaNs.
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if use_cache else None

        # decoder layers
        all_hidden_states = []

        # Prepare the input for COAT decoderlayer
        x, qx, sx = self.quantize_input_before_block(x)

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                if should_checkpoint_block(self.activation_checkpointing_strategy, block_idx):
                    # shape: (batch_size, seq_len, d_model)
                    x, qx, sx, cache = self._activation_checkpoint_fn(
                        block,
                        x,
                        qx,
                        sx,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        max_doc_len=max_doc_len,
                        cu_doc_lens=cu_doc_lens,
                    )
                else:
                    # shape: (batch_size, seq_len, d_model)
                    x, qx, sx, cache = block(
                        x,
                        qx,
                        sx,
                        attention_bias=attention_bias,
                        layer_past=layer_past,
                        use_cache=use_cache,
                        max_doc_len=max_doc_len,
                        cu_doc_lens=cu_doc_lens,
                    )

                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )
                x, cache = block_group(
                    x,
                    attention_bias=attention_bias,
                    layers_past=layers_past,
                    use_cache=use_cache,
                    max_doc_len=max_doc_len,
                    cu_doc_lens=cu_doc_lens,
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)

        # Summarize the output of the Decoder Layer
        x = self.quantize_output_after_block(x, qx, sx)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.transformer.ff_out(x)  # type: ignore
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        return OLMoOutput(
            logits=logits,
            attn_key_values=attn_key_values,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
        )

    def get_fsdp_wrap_policy(self, wrap_strategy: FSDPWrapStrategy | None = None):
        if wrap_strategy is None:
            return None

        # The 'recurse' mode for the wrap function does not behave like you'd expect.
        # Even if we return False, it may still recurse because PyTorch does what it wants,
        # not what you want. This causes issues when, for example, we want to wrap 'ff_out' (a linear layer)
        # but not other linear layers within a block.
        # So we have to explicitly tell PyTorch which linear layers to wrap, and we also just
        # return True in 'recurse' mode for simplicity.
        size_based_module_to_wrap = {self.transformer.wte}
        if hasattr(self.transformer, "ff_out"):
            size_based_module_to_wrap.add(self.transformer.ff_out)

        if wrap_strategy == FSDPWrapStrategy.by_block:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, CoatOLMoBlock)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_and_size:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, (CoatOLMoBlock,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, CoatOLMoBlockGroup)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group_and_size:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group_and_size' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, (CoatOLMoBlockGroup,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.size_based:
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            return size_based_auto_wrap_policy
        elif wrap_strategy in {
            FSDPWrapStrategy.one_in_two,
            FSDPWrapStrategy.one_in_three,
            FSDPWrapStrategy.one_in_four,
            FSDPWrapStrategy.one_in_five,
        }:
            c = {
                FSDPWrapStrategy.one_in_two: 2,
                FSDPWrapStrategy.one_in_three: 3,
                FSDPWrapStrategy.one_in_four: 4,
                FSDPWrapStrategy.one_in_five: 5,
            }[wrap_strategy]

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, CoatOLMoBlock) and module.layer_id % c == 0
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        else:
            raise NotImplementedError(wrap_strategy)

    num_params = OLMo.num_params

    @property
    def num_fwd_flops(self):
        if self.__num_fwd_flops:
            return self.__num_fwd_flops

        # embedding table is just a lookup in the forward pass
        n_params = self.num_params(include_embedding=False)
        # the number of parameters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # this gets us FLOPs / token
        params_flops_per_token = 2 * n_params
        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_token = self.config.n_layers * 2 * 2 * (self.config.d_model * self.config.max_sequence_length)
        self.__num_fwd_flops = params_flops_per_token + attn_flops_per_token
        return self.__num_fwd_flops

    @property
    def num_bck_flops(self):
        if self.__num_bck_flops:
            return self.__num_bck_flops

        n_params = self.num_params()
        params_flops_per_token = 4 * n_params
        attn_flops_per_token = self.config.n_layers * 8 * (self.config.d_model * self.config.max_sequence_length)
        self.__num_bck_flops = params_flops_per_token + attn_flops_per_token
        return self.__num_bck_flops

    generate = OLMo.generate

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: PathOrStr, device: str = "cpu", checkpoint_type: CheckpointType | None = None
    ) -> CoatOLMo:
        """
        Load an OLMo model from a checkpoint.
        """
        from olmo.util import resource_path

        # Guess checkpoint type.
        if checkpoint_type is None:
            try:
                if resource_path(checkpoint_dir, "model.pt").is_file():
                    checkpoint_type = CheckpointType.unsharded
                else:
                    checkpoint_type = CheckpointType.sharded
            except FileNotFoundError:
                checkpoint_type = CheckpointType.sharded

        # Load config.
        config_path = resource_path(checkpoint_dir, "config.yaml")
        model_config = ModelConfig.load(config_path, key="model", validate_paths=False)

        if checkpoint_type == CheckpointType.unsharded:
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            model = CoatOLMo(model_config)

            # Load state dict directly to target device.
            state_dict_path = resource_path(checkpoint_dir, "model.pt")
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(model._make_state_dict_compatible(state_dict)[0])
            model = model.to(torch.device(device))
        else:
            train_config = TrainConfig.load(config_path)
            if train_config.sharded_checkpointer == ShardedCheckpointerType.olmo_core:
                from olmo_core.distributed.checkpoint import load_model_and_optim_state  # type: ignore

                model_config.init_device = device
                model = CoatOLMo(model_config)
                load_model_and_optim_state(checkpoint_dir, model)
            else:
                # train_config.sharded_checkpointer == ShardedCheckpointerType.torch_new
                from olmo.checkpoint import load_model_state

                # Initialize model on target device. In this case the state dict is loaded in-place
                # so it's not necessary to start on CPU if the target device is a GPU.
                model_config.init_device = device
                model = CoatOLMo(model_config)

                # Load state dict in place.
                load_model_state(checkpoint_dir, model)

        return model.eval()

    def _make_state_dict_compatible(
        self, state_dict: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, set[str]]]:
        """
        Handles some cases where the state dict is valid yet may need to be transformed in order to
        be loaded.

        This modifies the state dict in-place and also returns it, along with a mapping of original key
        names to new key names in cases where the keys were simply renamed. That mapping can be used
        to make a corresponding optimizer state dict compatible as well.
        """
        import re
        from fnmatch import fnmatch

        new_keys_to_og_keys: dict[str, str] = {}

        # Remove "_fsdp_wrapped_module." prefix from all keys. We don't want this prefix when the model is
        # not wrapped in FSDP. And when the model is wrapped in FSDP, loading this state dict will still work
        # fine without the prefixes. This also simplifies the other steps below.
        for key in list(state_dict.keys()):
            state_dict[(new_key := key.replace("_fsdp_wrapped_module.", ""))] = state_dict.pop(key)
            new_keys_to_og_keys[new_key] = key

        # For backwards compatibility prior to fixing https://github.com/allenai/LLM/issues/222
        if self.config.block_type == BlockType.sequential:
            for key in list(state_dict.keys()):
                if fnmatch(key, "transformer.*.norm.weight"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.weight", "attn_norm.weight"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.weight", "ff_norm.weight"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]
                elif fnmatch(key, "transformer.*.norm.bias"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.bias", "attn_norm.bias"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.bias", "ff_norm.bias"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]

        # Realquantization will change the place the linear layers happen
        if self.qargs.use_quantize_model == "coat_real":
            for key in list(state_dict.keys()):
                if fnmatch(key, "transformer.blocks.*.att_proj.weight") and "BeforeAttention" not in key:
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("att_proj.weight", "BeforeAttention.att_proj.weight"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]
                elif fnmatch(key, "transformer.blocks.*.attn_out.weight") and "AfterAttention" not in key:
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("attn_out.weight", "AfterAttention.attn_out.weight"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]
                elif fnmatch(key, "transformer.blocks.*.ff_proj.weight") and "MLPResidual" not in key:
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("ff_proj.weight", "MLPResidual.ff_proj.weight"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]
                elif fnmatch(key, "transformer.blocks.*.ff_out.weight") and "MLPResidual" not in key:
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("ff_out.weight", "MLPResidual.ff_out.weight"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]

        # For loading a state dict that was saved with a different `block_group_size`.
        if "transformer.block_groups.0.0.attn_out.weight" in state_dict.keys():
            state_dict_block_group_size = len(
                [k for k in state_dict.keys() if fnmatch(k, "transformer.block_groups.0.*.attn_out.weight")]
            )
        else:
            state_dict_block_group_size = 1
        if self.config.block_group_size != state_dict_block_group_size:
            log.info(
                f"Regrouping state dict blocks from group size {state_dict_block_group_size} to "
                f"group size {self.config.block_group_size}"
            )
            # For simplicity we're first going to flatten out the block groups in the state dict (if necessary)
            # and then (re-)group them into the right block sizes.
            if state_dict_block_group_size > 1:
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.block_groups\.(\d+)\.(\d+)\..*", key)) is not None:
                        group_idx, group_block_idx = int(m.group(1)), int(m.group(2))
                        block_idx = (group_idx * state_dict_block_group_size) + group_block_idx
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"block_groups.{group_idx}.{group_block_idx}.", f"blocks.{block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

            if self.config.block_group_size > 1:
                # Group the state dict blocks into the right block size.
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.blocks\.(\d+)\..*", key)) is not None:
                        block_idx = int(m.group(1))
                        group_idx, group_block_idx = (
                            block_idx // self.config.block_group_size,
                            block_idx % self.config.block_group_size,
                        )
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"blocks.{block_idx}.", f"block_groups.{group_idx}.{group_block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

        og_keys_to_new: dict[str, set[str]] = defaultdict(set)
        for new_key, og_key in new_keys_to_og_keys.items():
            og_keys_to_new[og_key].add(new_key)

        return state_dict, og_keys_to_new


