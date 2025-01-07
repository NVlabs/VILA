# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch Qwen2 model."""

import math
import os
from dataclasses import asdict, dataclass, field
from fnmatch import fnmatch
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2FlashAttention2,
    Qwen2ForCausalLM,
    Qwen2MLP,
    Qwen2Model,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    Qwen2SdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

# FP8 related
from ..coat.activation.models._fp8_quantization_config import QuantizationConfig
from ..coat.activation.models._fp8_weightcache import FP8CacheWeightModule
from ..coat.activation.models._fp8manager import FP8Manager
from ..coat.activation.real_quantization import (
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
from ..qlinear_te import QLinearTE
from .configuration_quantize import QuantizationConfig

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)


class FP8ActivationResidualQwen2Config(Qwen2Config):
    model_type = "fp8activationresidual_qwen2"

    def __init__(
        self,
        coat_fp8_args=None,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            use_sliding_window,
            sliding_window,
            max_window_layers,
            attention_dropout,
            **kwargs,
        )

        self.coat_fp8_args = coat_fp8_args


class FP8ActivationResidualQwen2BeforeAttentionResidual(FP8CacheWeightModule):
    """
    This is a typical transformer attention module that contains (1) Residual (2) LayerNorm / RMSNorm (3) 1 * Linear layers
    """

    def __init__(
        self, config: FP8ActivationResidualQwen2Config, qargs: QuantizationConfig, layer_idx: Optional[int] = None
    ):
        super().__init__(config, qargs, layer_idx)

        self.qargs = qargs
        self.fwobits = {
            "fabit": self.qargs.fabit,
            "fwbit": self.qargs.fwbit,
            "fobit": self.qargs.fobit,
            "babit": self.qargs.babit,
            "bwbit": self.qargs.bwbit,
            "bobit": self.qargs.bobit,
        }

        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)

    def forward(self, x, s, rmsnorm_weight):
        if self.training:
            if self.qargs.weight_memory_efficient:
                # Prepare
                with torch.no_grad():
                    weight1_s = self.prepare_weight(self.q_proj.weight, "q_proj", FP8Manager.is_first_microbatch)
                    weight2_s = self.prepare_weight(self.k_proj.weight, "k_proj", FP8Manager.is_first_microbatch)
                    weight3_s = self.prepare_weight(self.v_proj.weight, "v_proj", FP8Manager.is_first_microbatch)
                return _FP8ActivationResidualQwen2BeforeAttentionResidual.apply(
                    x,
                    s,
                    self.q_proj.weight,
                    None,
                    None,
                    weight1_s,
                    self.q_proj.bias,
                    self.k_proj.weight,
                    None,
                    None,
                    weight2_s,
                    self.k_proj.bias,
                    self.v_proj.weight,
                    None,
                    None,
                    weight3_s,
                    self.v_proj.bias,
                    rmsnorm_weight,
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
                        self.q_proj.weight, "q_proj", FP8Manager.is_first_microbatch
                    )
                    weight2, weight2_t, weight2_s = self.prepare_weight(
                        self.k_proj.weight, "k_proj", FP8Manager.is_first_microbatch
                    )
                    weight3, weight3_t, weight3_s = self.prepare_weight(
                        self.v_proj.weight, "v_proj", FP8Manager.is_first_microbatch
                    )
                return _FP8ActivationResidualQwen2BeforeAttentionResidual.apply(
                    x,
                    s,
                    self.q_proj.weight,
                    weight1,
                    weight1_t,
                    weight1_s,
                    self.k_proj.weight,
                    weight2,
                    weight2_t,
                    weight2_s,
                    self.v_proj.weight,
                    weight3,
                    weight3_t,
                    weight3_s,
                    rmsnorm_weight,
                    self.qargs.group_size,
                    self.fwobits,
                    self.layer_id,
                    self.config,
                    self.qargs,
                )
        else:
            raise NotImplementedError("This should be implemented in the future")
            return re_x, self.att_proj(self.attn_norm(re_x))


class _FP8ActivationResidualQwen2BeforeAttentionResidual(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        in_x,
        in_s,
        weight1_origin,
        weight1,
        weight1_t,
        weight1_s,
        weight1_bias,
        weight2_origin,
        weight2,
        weight2_t,
        weight2_s,
        weight2_bias,
        weight3_origin,
        weight3,
        weight3_t,
        weight3_s,
        weight3_bias,
        rmsnorm_weight,
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
        ln_x, ln_s, ln_x_t, ln_utils = fp8_rmsnorm_forward(
            in_x, in_s, rmsnorm_weight, group_size, eps, transpose_output_2d=True
        )

        # Linear Layer QKV Projection
        if qargs.weight_memory_efficient:
            assert weight1 is None  # memory efficient
            weight1, weight1_s = fp8_division(weight1_origin, qargs.group_size, fwobits["fwbit"], weight1_s)
            weight2, weight2_s = fp8_division(weight2_origin, qargs.group_size, fwobits["fwbit"], weight2_s)
            weight3, weight3_s = fp8_division(weight3_origin, qargs.group_size, fwobits["fwbit"], weight3_s)

        fc1_x = fp8_linear_forward(ln_x, ln_s, weight1, weight1_s, False, group_size, bias=weight1_bias)  # query states
        fc2_x = fp8_linear_forward(ln_x, ln_s, weight2, weight2_s, False, group_size, bias=weight2_bias)  # key states
        fc3_x = fp8_linear_forward(ln_x, ln_s, weight3, weight3_s, False, group_size, bias=weight3_bias)  # value states

        # ==================== save for backward ====================
        ctx.save_for_backward(in_x, in_s, ln_x_t, ln_s)
        if qargs.weight_memory_efficient:
            assert weight1_t is None and weight2_t is None and weight3_t is None
            ctx.weight = weight1_origin, weight1_s, weight2_origin, weight2_s, weight3_origin, weight3_s
        else:
            ctx.weight = weight1_t, weight1_s, weight2_t, weight2_s, weight3_t, weight3_s
        ctx.bias = weight1_bias, weight2_bias, weight3_bias

        ctx.group_size = group_size
        ctx.ln_utils = ln_utils
        ctx.utils = fwobits, layer_id, config, qargs

        return in_x, in_s, fc1_x, fc2_x, fc3_x

    @staticmethod
    def backward(ctx, q_grad, s_grad, query_g, key_g, value_g):
        in_x, in_s, ln_x_t, ln_s = ctx.saved_tensors
        weight1_t, weight1_s, weight2_t, weight2_s, weight3_t, weight3_s = ctx.weight
        weight1_bias, weight2_bias, weight3_bias = ctx.bias

        group_size = ctx.group_size
        rms_weight, rstd, num_warps = ctx.ln_utils
        fwobits, layer_id, config, qargs = ctx.utils

        # ==================== Begin backward ====================
        # Gradient of Bias TODO: make this better
        if weight1_bias is not None and weight2_bias is not None and weight3_bias is not None:
            att_q_bg = query_g.reshape(-1, query_g.shape[-1]).sum(0)
            att_k_bg = key_g.reshape(-1, key_g.shape[-1]).sum(0)
            att_v_bg = value_g.reshape(-1, value_g.shape[-1]).sum(0)
        else:
            att_q_bg = None
            att_k_bg = None
            att_v_bg = None

        # Quantize the RoPE and FlashAttention Output. grad_input and grad_weight requires different data layout.
        query_g, query_gs, query_g_t = fp8_quantize_pertensor_transpose(
            query_g, group_size, fwobits["babit"], transpose_output_2d=True, stochastic=False
        )
        key_g, key_gs, key_g_t = fp8_quantize_pertensor_transpose(
            key_g, group_size, fwobits["babit"], transpose_output_2d=True, stochastic=False
        )
        value_g, value_gs, value_g_t = fp8_quantize_pertensor_transpose(
            value_g, group_size, fwobits["babit"], transpose_output_2d=True, stochastic=False
        )

        # Linear Layer QKV Projection
        if qargs.weight_memory_efficient:
            weight1_t, weight1_s = fp8_division_transpose(
                weight1_t, qargs.group_size, fwobits["fwbit"], weight1_s, only_transposed=True
            )
            weight2_t, weight2_s = fp8_division_transpose(
                weight2_t, qargs.group_size, fwobits["fwbit"], weight2_s, only_transposed=True
            )
            weight3_t, weight3_s = fp8_division_transpose(
                weight3_t, qargs.group_size, fwobits["fwbit"], weight3_s, only_transposed=True
            )

        fc1_g1, att_q_wg = fp8_linear_backward(
            ln_x_t, ln_s, query_g, query_gs, query_g_t, weight1_t, weight1_s, group_size
        )
        fc1_g2, att_k_wg = fp8_linear_backward(ln_x_t, ln_s, key_g, key_gs, key_g_t, weight2_t, weight2_s, group_size)
        fc1_g3, att_v_wg = fp8_linear_backward(
            ln_x_t, ln_s, value_g, value_gs, value_g_t, weight3_t, weight3_s, group_size
        )

        fc1_g = fc1_g1 + fc1_g2 + fc1_g3

        # LayerNorm
        in_g, rms_weight_grad = fp8_rmsnorm_backward(in_x, in_s, fc1_g, rms_weight, rstd, group_size, num_warps)

        # Add the gradient together, and prepare the input of the next layer.
        in_g, in_sg, in_sg_g16 = fp8_add_Ig16_Ifp_Opt(
            q_grad, s_grad, in_g, group_size, fwobits["babit"], stochastic=False
        )

        # for autograd. forward's data type should be the same of backward tensor. this will not change the actual binary representation.
        in_g = in_g.view(torch.float8_e4m3fn)

        # Although the next operator is a linear layer in MLPResidual module, we return in_sg_g16 to make the size compatible with the forward. Otherwise it will not pass autograd.
        return (
            in_g,
            in_sg_g16,
            att_q_wg,
            None,
            None,
            None,
            att_q_bg,
            att_k_wg,
            None,
            None,
            None,
            att_k_bg,
            att_v_wg,
            None,
            None,
            None,
            att_v_bg,
            rms_weight_grad,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FP8ActivationResidualQwen2AfterAttentionResidual(FP8CacheWeightModule):
    """
    This is a typical transformer attention module that contains (1) Residual (2) 1 * Linear layers
    """

    def __init__(self, config: FP8ActivationResidualQwen2Config, qargs: QuantizationConfig, layer_id):
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

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)

        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, re_qx, re_sx, in_x):
        if self.training:
            if self.qargs.weight_memory_efficient:
                # prepare for the weight
                with torch.no_grad():
                    weight4_s = self.prepare_weight(self.o_proj.weight, "o_proj", FP8Manager.is_first_microbatch)

                return _FP8ActivationResidualQwen2AfterAttentionResidual.apply(
                    re_qx,
                    re_sx,
                    in_x,
                    self.o_proj.weight,
                    None,
                    None,
                    weight4_s,
                    self.qargs.group_size,
                    self.fwobits,
                    self.layer_id,
                    self.config,
                    self.qargs,
                )
            else:
                # prepare for the weight
                with torch.no_grad():
                    weight4, weight4_t, weight4_s = self.prepare_weight(
                        self.o_proj.weight, "o_proj", FP8Manager.is_first_microbatch
                    )

                return _FP8ActivationResidualQwen2AfterAttentionResidual.apply(
                    re_qx,
                    re_sx,
                    in_x,
                    self.o_proj.weight,
                    weight4,
                    weight4_t,
                    weight4_s,
                    self.qargs.group_size,
                    self.fwobits,
                    self.layer_id,
                    self.config,
                    self.qargs,
                )
        else:
            return re_x + self.attn_out(in_x), None, None


class _FP8ActivationResidualQwen2AfterAttentionResidual(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        re_qx,
        re_sx,
        flash_x,
        weight4_origin,
        weight4,
        weight4_t,
        weight4_s,
        group_size,
        fwobits,
        layer_id,
        config,
        qargs,
    ):
        # Quantize the FlashAttention Output
        flash_qx, flash_s, _ = fp8_quantize_pertensor(
            flash_x, group_size, fwobits["fabit"]
        )  # Modified to make it memory efficient

        # # Attention Projection Linear Layer
        if qargs.weight_memory_efficient:
            assert weight4 is None  # memory efficient
            weight4, weight4_s = fp8_division(weight4_origin, qargs.group_size, fwobits["fwbit"], weight4_s)
        fc4_x = fp8_linear_forward(flash_qx, flash_s, weight4, weight4_s, False, group_size)  #

        # import IPython
        # IPython.embed()
        # Add the activations together
        fp_x, (out_x, out_s) = fp8_add_Ig16_Ifp_Ofp_Og16(re_qx, re_sx, fc4_x, flash_qx.dtype, group_size)

        # ==================== save for backward ====================
        ctx.save_for_backward(flash_x, flash_s)
        if qargs.weight_memory_efficient:
            assert weight4_t is None
            ctx.weight = weight4_origin, weight4_s
        else:
            ctx.weight = weight4_t, weight4_s
        ctx.group_size = group_size
        ctx.fwobits = fwobits
        ctx.utils = fwobits, layer_id, config, qargs

        # For autograd
        out_x = out_x.view(torch.float8_e4m3fn)

        return fp_x, out_x, out_s

    @staticmethod
    def backward(ctx, fp_grad, out_g, out_gs):
        flash_x, flash_s = ctx.saved_tensors
        weight4_t, weight4_s = ctx.weight
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
            weight4_t, weight4_s = fp8_division_transpose(
                weight4_t, qargs.group_size, fwobits["fwbit"], weight4_s, only_transposed=True
            )
        fc4_g, attn_out_wg = fp8_linear_backward(
            flash_x_t, flash_s, out_g, out_gs_max, out_g_t, weight4_t, weight4_s, group_size
        )

        return fp_grad, fc4_g, attn_out_wg, None, None, None, None, None, None, None, None


class FP8ActivationResidualQwen2MLPResidual(FP8CacheWeightModule):
    """
    This is a typical transformer attention module that contains (1) Residual (2) LayerNorm / RMSNorm (3) 2 / 3 * Linear layers
    (4) GELU / Silu Activation
    """

    def __init__(self, config: FP8ActivationResidualQwen2Config, qargs: QuantizationConfig, layer_id, hidden_size: int):
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

        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.training = True

        # below is only used when training = False
        assert config.hidden_act == "silu", "We only support silu activation currently"
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, re_x, x, s, rmsnorm_weight):
        if self.training:
            if self.qargs.weight_memory_efficient:  # prepare for the weight
                with torch.no_grad():
                    weight1_s = self.prepare_weight(self.gate_proj.weight, "gate_proj", FP8Manager.is_first_microbatch)
                    weight2_s = self.prepare_weight(self.up_proj.weight, "up_proj", FP8Manager.is_first_microbatch)
                    weight3_s = self.prepare_weight(self.down_proj.weight, "down_proj", FP8Manager.is_first_microbatch)

                return _FP8ActivationResidualQwen2MLPResidual.apply(
                    re_x,
                    x,
                    s,
                    self.gate_proj.weight,
                    None,
                    None,
                    weight1_s,
                    self.up_proj.weight,
                    None,
                    None,
                    weight2_s,
                    self.down_proj.weight,
                    None,
                    None,
                    weight3_s,
                    rmsnorm_weight,
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
                        self.gate_proj.weight, "gate_proj", FP8Manager.is_first_microbatch
                    )
                    weight2, weight2_t, weight2_s = self.prepare_weight(
                        self.up_proj.weight, "up_proj", FP8Manager.is_first_microbatch
                    )
                    weight3, weight3_t, weight3_s = self.prepare_weight(
                        self.down_proj.weight, "down_proj", FP8Manager.is_first_microbatch
                    )

                return _FP8ActivationResidualQwen2MLPResidual.apply(
                    re_x,
                    x,
                    s,
                    self.gate_proj.weight,
                    weight1,
                    weight1_t,
                    weight1_s,
                    self.up_proj.weight,
                    weight2,
                    weight2_t,
                    weight2_s,
                    self.down_proj.weight,
                    weight3,
                    weight3_t,
                    weight3_s,
                    rmsnorm_weight,
                    self.qargs.group_size,
                    self.fwobits,
                    self.layer_id,
                    self.config,
                    self.qargs,
                )
        else:
            raise NotImplementedError("Need TODO")
            og_x = re_x
            re_x = self.ff_norm(re_x)
            re_x = self.ff_proj(re_x)
            re_x = self.act(re_x)
            re_x = self.ff_out(re_x)
            re_x = og_x + re_x
            return re_x, None, None


class _FP8ActivationResidualQwen2MLPResidual(torch.autograd.Function):
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
        weight3_origin,
        weight3,
        weight3_t,
        weight3_s,
        rmsnorm_weight,
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
        ln_x, ln_s, ln_x_t, ln_utils = fp8_rmsnorm_forward(
            in_x, in_s, rmsnorm_weight, group_size, eps, transpose_output_2d=True
        )

        # Linear Layer of Up Projection and Gate Projection. They are fused as one linear layer.
        if qargs.weight_memory_efficient:
            assert weight1 is None and weight2 is None and weight3 is None  # memory efficient
            weight1, weight1_s = fp8_division(weight1_origin, qargs.group_size, fwobits["fwbit"], weight1_s)
            weight2, weight2_s = fp8_division(weight2_origin, qargs.group_size, fwobits["fwbit"], weight2_s)
            weight3, weight3_s = fp8_division(weight3_origin, qargs.group_size, fwobits["fwbit"], weight3_s)

        gate_x, gate_s = fp8_linear_forward(ln_x, ln_s, weight1, weight1_s, True, group_size)  # Gate Proj
        up_x, up_s = fp8_linear_forward(ln_x, ln_s, weight2, weight2_s, True, group_size)  # Up Proj

        # silu Activation
        silu_x, silu_s = fp8_silu_forward(gate_x, gate_s, group_size)

        # Element-wise Multiplication
        mul_x, mul_s, mul_x_t = fp8_mul_forward(silu_x, silu_s, up_x, up_s, group_size, transpose_output_2d=True)

        # Output Projection
        if weight3 is None:  # memory efficient
            weight3, weight3_s = fp8_division(weight3_origin, qargs.group_size, fwobits["fwbit"], weight3_s)
        fc3_x = fp8_linear_forward(mul_x, mul_s, weight3, weight3_s, False, group_size)

        # Add the activation together
        out_x, out_s = fp8_add_Ifp_Ifp_Og16(re_x, fc3_x, mul_x.dtype, group_size)

        # ==================== save for backward ====================
        ctx.save_for_backward(in_x, in_s, ln_x_t, ln_s, gate_x, gate_s, up_x, up_s, silu_x, silu_s, mul_x_t, mul_s)

        ctx.weight = (weight1_t, weight1_s, weight2_t, weight2_s)
        if (
            qargs.weight_memory_efficient
        ):  # Weight_1/2_origin will not be saved twice, so it will be more memory efficient.
            assert weight1_t is None and weight2_t is None and weight3_t is None
            ctx.weight = (weight1_origin, weight1_s, weight2_origin, weight2_s, weight3_origin, weight3_s)
        else:  # Weight1/2_t is different from the origin weight, so saving it will consumes additional memory footprint.
            ctx.weight = (weight1_t, weight1_s, weight2_t, weight2_s, weight3_t, weight3_s)

        ctx.group_size = group_size
        ctx.ln_utils = ln_utils
        ctx.utils = fwobits, layer_id, config, qargs

        out_x = out_x.view(torch.float8_e4m3fn)

        return out_x, out_s

    @staticmethod
    def backward(ctx, out_g, out_gs):
        fwobits, layer_id, config, qargs = ctx.utils

        in_x, in_s, ln_x_t, ln_s, gate_x, gate_s, up_x, up_s, silu_x, silu_s, mul_x_t, mul_s = ctx.saved_tensors

        (weight1_t, weight1_s, weight2_t, weight2_s, weight3_t, weight3_s) = ctx.weight
        group_size = ctx.group_size
        rms_weight, rstd, num_warps = ctx.ln_utils
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
            weight3_t, weight3_s = fp8_division_transpose(
                weight3_t, qargs.group_size, fwobits["fwbit"], weight3_s, only_transposed=True
            )
        fc3_g, weight3_grad = fp8_linear_backward(
            mul_x_t, mul_s, out_g, out_gs_max, out_g_t, weight3_t, weight3_s, group_size
        )

        # [MEM TEST]
        del out_g, out_g_t, weight3_t

        # Element-wise Multiplication, 1 means gate, 2 means up
        mul_g1, (mul_g2, mul_gs2, mul_g2_t) = fp8_mul_backward(
            silu_x, silu_s, up_x, up_s, fc3_g, group_size, fwobits["babit"], output_quantized_transpose=True
        )

        # Silu activation
        silu_g, silu_gs, silu_g_t = fp8_silu_backward(
            gate_x, gate_s, mul_g1, group_size, fwobits["babit"], output_quantized_transpose=True
        )

        # Linear Layer of Up and Gate Projection
        if qargs.weight_memory_efficient:
            weight1_t, weight1_s = fp8_division_transpose(
                weight1_t, group_size, fwobits["fwbit"], weight1_s, only_transposed=True
            )
            weight2_t, weight2_s = fp8_division_transpose(
                weight2_t, group_size, fwobits["fwbit"], weight2_s, only_transposed=True
            )

        # Gate Proj
        fc1_g, weight1_grad = fp8_linear_backward(
            ln_x_t, ln_s, silu_g, silu_gs, silu_g_t, weight1_t, weight1_s, group_size
        )
        fc2_g, weight2_grad = fp8_linear_backward(
            ln_x_t, ln_s, mul_g2, mul_gs2, mul_g2_t, weight2_t, weight2_s, group_size
        )

        fc_g = fc1_g + fc2_g

        # layerNorm
        in_g, rms_weight_grad = fp8_rmsnorm_backward(in_x, in_s, fc_g, rms_weight, rstd, group_size, num_warps)

        # Add the gradient together
        re_g, (in_g, in_sg, in_sg_g16) = fp8_add_Ifp_Ifp_Ofp_Opt(
            out_g, out_gs_max, in_g, group_size, fwobits["babit"], stochastic=False
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
            weight3_grad,
            None,
            None,
            None,
            rms_weight_grad,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class FP8ActivationResidualQwen2AttentionWithoutLinear(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.rotary_emb = Qwen2RotaryEmbedding(config=self.config)

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = query_states.size()

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class FP8ActivationResidualQwen2FlashAttention2WithoutLinear(FP8ActivationResidualQwen2AttentionWithoutLinear):
    """
    Qwen2 flash attention module, following Qwen2 attention module. This module inherits from `Qwen2Attention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ):
        bsz, q_len, _ = query_states.size()

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            kv_seq_len = key_states.shape[-2] + cache_position[0]
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        if (
            self.config.use_sliding_window
            and getattr(self.config, "sliding_window", None) is not None
            and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=sliding_window,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


class FP8ActivationResidualQwen2SdpaAttentionWithoutLinear(FP8ActivationResidualQwen2AttentionWithoutLinear):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                query_states=query_states,
                key_states=key_states,
                value_states=value_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = query_states.size()

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        return attn_output, None, past_key_value


FP8LINEARRESIDUALQWEN2_ATTENTION_CLASSES = {
    "eager": FP8ActivationResidualQwen2AttentionWithoutLinear,
    "flash_attention_2": FP8ActivationResidualQwen2FlashAttention2WithoutLinear,
    "sdpa": FP8ActivationResidualQwen2SdpaAttentionWithoutLinear,
}


class FP8ActivationResidualQwen2DecoderLayer(nn.Module):
    def __init__(self, config: FP8ActivationResidualQwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = FP8LINEARRESIDUALQWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.qargs = QuantizationConfig(**config.coat_fp8_args)
        self.BeforeAttention = FP8ActivationResidualQwen2BeforeAttentionResidual(config, self.qargs, layer_idx)
        self.AfterAttention = FP8ActivationResidualQwen2AfterAttentionResidual(config, self.qargs, layer_idx)
        self.MLPResidual = FP8ActivationResidualQwen2MLPResidual(config, self.qargs, layer_idx, self.hidden_size)

        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        quant_hidden_states: torch.Tensor,
        scale_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        # Coat: The residual, LayerNorm, and the Q/K/V Projection Linear Layer
        residual_quant, residual_scale, query_states, key_states, value_states = self.BeforeAttention(
            quant_hidden_states, scale_hidden_states, self.input_layernorm.weight
        )

        # Self Attention without any linear layer
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        # Coat: The Output Projection Linear Layer and Residual
        hidden_states, quant_hidden_states, scale_hidden_states = self.AfterAttention(
            residual_quant, residual_scale, hidden_states
        )

        # Residual Connection, LayerNorm, and the whole MLP module
        quant_hidden_states, scale_hidden_states = self.MLPResidual(
            hidden_states, quant_hidden_states, scale_hidden_states, self.post_attention_layernorm.weight
        )

        outputs = ((quant_hidden_states, scale_hidden_states),)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class FP8ActivationResidualQwen2PreTrainedModel(Qwen2PreTrainedModel):
    config_class = FP8ActivationResidualQwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FP8ActivationResidualQwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class FP8ActivationResidualQwen2Model(FP8ActivationResidualQwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: FP8ActivationResidualQwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [FP8ActivationResidualQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        # Quantize
        self.qargs = QuantizationConfig(**config.coat_fp8_args)
        self.quantize_input_before_block = Coat_quantize_bgn(self.qargs)
        self.quantize_output_after_block = Coat_quantize_end(self.qargs)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        # Prepare the input for Coat decoderlayer
        quant_hidden_states, scale_hidden_states = self.quantize_input_before_block(hidden_states)

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    quant_hidden_states,
                    scale_hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    quant_hidden_states,
                    scale_hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # Summarize the output of the Decoder Layer
        hidden_states = self.quantize_output_after_block(quant_hidden_states, scale_hidden_states)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    _update_causal_mask = Qwen2Model._update_causal_mask


class FP8ActivationResidualQwen2ForCausalLM(FP8ActivationResidualQwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = FP8ActivationResidualQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    forward = Qwen2ForCausalLM.forward

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    prepare_inputs_for_generation = Qwen2ForCausalLM.prepare_inputs_for_generation


AutoConfig.register("fp8activationresidual_qwen2", FP8ActivationResidualQwen2Config)
AutoModel.register(FP8ActivationResidualQwen2Config, FP8ActivationResidualQwen2Model)
AutoModelForCausalLM.register(FP8ActivationResidualQwen2Config, FP8ActivationResidualQwen2ForCausalLM)


def make_state_dict_compatible(state_dict: dict[str, torch.Tensor]):
    compatible_state_dict = {}

    for key, value in state_dict.items():
        if fnmatch(key, "*self_attn.q_proj*"):
            new_key = key.replace("self_attn.q_proj", "BeforeAttention.q_proj")
        elif fnmatch(key, "*self_attn.k_proj*"):
            new_key = key.replace("self_attn.k_proj", "BeforeAttention.k_proj")
        elif fnmatch(key, "*self_attn.v_proj*"):
            new_key = key.replace("self_attn.v_proj", "BeforeAttention.v_proj")
        elif fnmatch(key, "*self_attn.o_proj*"):
            new_key = key.replace("self_attn.o_proj", "AfterAttention.o_proj")

        elif fnmatch(key, "*mlp.gate_proj*"):
            new_key = key.replace("mlp.gate_proj", "MLPResidual.gate_proj")
        elif fnmatch(key, "*mlp.up_proj*"):
            new_key = key.replace("mlp.up_proj", "MLPResidual.up_proj")
        elif fnmatch(key, "*mlp.down_proj*"):
            new_key = key.replace("mlp.down_proj", "MLPResidual.down_proj")

        else:
            new_key = key

        compatible_state_dict[new_key] = value

    return compatible_state_dict


