# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch LLaMA model."""
import math
import os
import time
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaFlashAttention2,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaLinearScalingRotaryEmbedding,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from ..qlinear_te import QLinearTE

try:
    import transformer_engine.pytorch as te
except:
    pass
from ..quantization import QGELU, QAct_FPin, QAct_FPout, QAdd, QIdentity, QLayerNorm, QLinear, QMul

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "QMemLlamaConfig"


class QMemLlamaConfig(LlamaConfig):
    model_type = "qmemllama"


class QLlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, args=None, layer_type=None):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.qargs = args
        self.QAct_layernorm_in = QAct_FPout(args, layer_type=layer_type + "_in")
        self.QAct_layernorm_out = QAct_FPin(args, layer_type=layer_type + "_out")

    def forward(self, hidden_states, s):
        hidden_states = self.QAct_layernorm_in(hidden_states, s)

        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)

        hidden_states, s = self.QAct_layernorm_out(hidden_states)
        return hidden_states, s


ALL_LAYERNORM_LAYERS.append(QLlamaRMSNorm)


class QMemLlamaMLP(LlamaMLP):
    def __init__(self, config, layer_idx):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.gate_proj = QLinear(
            self.hidden_size, self.intermediate_size, bias=False, args=config, layer_type="mlp_gate"
        )
        self.up_proj = QLinear(self.hidden_size, self.intermediate_size, bias=False, args=config, layer_type="mlp_up")
        self.down_proj = QLinear(
            self.intermediate_size, self.hidden_size, bias=False, args=config, layer_type="mlp_down"
        )
        self.act_fn = ACT2FN[config.hidden_act]

        self.QAct_act_sum = QAct_FPout(config, layer_type="mlp_act_sum")
        self.QAct_act_gate = QAct_FPin(config, layer_type="mlp_act_gate")
        self.QAct_act_up = QAct_FPin(config, layer_type="mlp_act_up")

        self.QAct_act_in = QAct_FPout(config, layer_type="mlp_act_in")
        self.QAct_act_out = QAct_FPin(config, layer_type="mlp_act_out")

        self.QMul_act = QMul(config, layer_type="mul_act")

    def forward(self, x, s):
        if self.config.pretraining_tp > 1:
            raise ValueError("Currently Quantization is not implemented for tensor parallel for simplicity")
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)

            gate_proj = torch.cat([F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i]) for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            # down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
            x = self.QAct_act_sum(x, s)
            x_gate, s_gate = self.QAct_act_gate(x)
            x_up, s_up = self.QAct_act_up(x)
            x_gate, s_gate = self.gate_proj(x_gate, s_gate)
            x_gate = self.QAct_act_in(x_gate, s_gate)
            x_gate = self.act_fn(x_gate)
            x_gate, s_gate = self.QAct_act_out(x_gate)

            x_up, s_up = self.up_proj(x_up, s_up)
            x, s = self.QMul_act(x_gate, x_up, s_gate, s_up)
            down_proj, s = self.down_proj(x, s)
        return down_proj, s


class QMemLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: QMemLlamaConfig, layer_idx):
        super().__init__(config)
        self.layer_idx = layer_idx
        self.q_proj = QLinear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
            args=config,
            layer_type="attn_q",
        )
        self.k_proj = QLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            args=config,
            layer_type="attn_k",
        )
        self.v_proj = QLinear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            args=config,
            layer_type="attn_v",
        )
        self.o_proj = QLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            args=config,
            layer_type="attn_proj",
        )

        self.QAct_qkv_sum = QAct_FPout(config, layer_type="attn_qkv_sum")

        self.QAct_q_in = QAct_FPin(config, layer_type="attn_q_in")
        self.QAct_k_in = QAct_FPin(config, layer_type="attn_k_in")
        self.QAct_v_in = QAct_FPin(config, layer_type="attn_v_in")

        self.QAct_q_out = QAct_FPout(config, layer_type="attn_q_out")
        self.QAct_k_out = QAct_FPout(config, layer_type="attn_k_out")
        self.QAct_v_out = QAct_FPout(config, layer_type="attn_v_out")
        self.QAct_proj_in = QAct_FPin(config, layer_type="attn_proj_in")


class QMemLlamaFlashAttention2(QMemLlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        s: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if isinstance(past_key_value, StaticCache):
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
                "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
            )

        output_attentions = False

        bsz, q_len, _ = hidden_states.size()

        hidden_states = self.QAct_qkv_sum(hidden_states, s)

        q, sq = self.QAct_q_in(hidden_states)
        k, sk = self.QAct_k_in(hidden_states)
        v, sv = self.QAct_v_in(hidden_states)

        query_states, sq = self.q_proj(q, sq)
        key_states, sk = self.k_proj(k, sk)
        value_states, sv = self.v_proj(v, sv)

        query_states = self.QAct_q_out(query_states, sq)
        key_states = self.QAct_k_out(key_states, sk)
        value_states = self.QAct_v_out(value_states, sv)

        # Flash attention requires the input to have the shape
        # batch_size x seq_length x head_dim x hidden_dim
        # therefore we just need to keep the original shape
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
        # to be able to avoid many of these transpose/reshape/view.
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in the correct dtype just to be sure everything works as expected.
        # This might slowdown training & inference so it is recommended to not cast the LayerNorms
        # in fp32. (LlamaRMSNorm handles it correctly)

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

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            position_ids=position_ids,
            dropout=dropout_rate,
            sliding_window=getattr(self, "sliding_window", None),
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            is_causal=self.is_causal,
        )

        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        attn_output = attn_output.to(torch.float32)
        attn_output, s = self.QAct_proj_in(attn_output)
        attn_output, s = self.o_proj(attn_output, s)

        if not output_attentions:
            attn_weights = None

        return attn_output, s, attn_weights, past_key_value


class QMemLlamaSdpaAttention(QMemLlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LlamaAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        s: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        hidden_states = self.QAct_qkv_sum(hidden_states, s)

        q, sq = self.QAct_q_in(hidden_states)
        k, sk = self.QAct_k_in(hidden_states)
        v, sv = self.QAct_v_in(hidden_states)

        query_states, sq = self.q_proj(q, sq)
        key_states, sk = self.k_proj(k, sk)
        value_states, sv = self.v_proj(v, sv)

        query_states = self.QAct_q_out(query_states, sq)
        key_states = self.QAct_k_out(key_states, sk)
        value_states = self.QAct_v_out(value_states, sv)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
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
        attn_output = attn_output.view(bsz, q_len, -1)

        # attn_output = attn_output.to(torch.float32)
        attn_output, s = self.QAct_proj_in(attn_output)
        attn_output, s = self.o_proj(attn_output, s)

        return attn_output, s, None, past_key_value


QMemLLAMA_ATTENTION_CLASSES = {
    "eager": QMemLlamaAttention,
    "flash_attention_2": QMemLlamaFlashAttention2,
    "sdpa": QMemLlamaSdpaAttention,
}


class QMemLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: QMemLlamaConfig, layer_idx):
        super().__init__(config, layer_idx=layer_idx)
        self.hidden_size = config.hidden_size

        self.self_attn = QMemLLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = QMemLlamaMLP(config, layer_idx)
        self.input_layernorm = QLlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, args=config, layer_type="ln_attn"
        )
        self.post_attention_layernorm = QLlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, args=config, layer_type="ln_mlp"
        )

        self.QAdd_attn = QAdd(config, layer_type="add_attn")
        self.QAdd_mlp = QAdd(config, layer_type="add_mlp")

        self.QAct_reattnout_fx = QAct_FPin(config, layer_type="re_attn_out_fx")
        self.QAct_reattnout_re = QAct_FPin(config, layer_type="re_attn_out_re")

        self.QAct_remlpout_fx = QAct_FPin(config, layer_type="re_mlp_out_fx")
        self.QAct_remlpout_re = QAct_FPin(config, layer_type="re_mlp_out_re")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        residual, res = self.QAct_reattnout_re(hidden_states)
        hidden_states, s = self.QAct_reattnout_fx(hidden_states)

        hidden_states, s = self.input_layernorm(hidden_states, s)

        # Self Attention
        hidden_states, s, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            s=s,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = self.QAdd_attn(residual, hidden_states, res, s)

        # Fully Connected
        residual, res = self.QAct_remlpout_re(hidden_states)
        hidden_states, s = self.QAct_remlpout_fx(hidden_states)

        hidden_states, s = self.post_attention_layernorm(hidden_states, s)
        hidden_states, s = self.mlp(hidden_states, s)
        hidden_states = self.QAdd_mlp(residual, hidden_states, res, s)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class QMemLlamaPreTrainedModel(LlamaPreTrainedModel):
    config_class = QMemLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["QMemLlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear) or isinstance(module, QLinearTE):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class QMemLlamaModel(QMemLlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: QMemLlamaConfig
    """

    def __init__(self, config: QMemLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [QMemLlamaDecoderLayer(config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    _update_causal_mask = LlamaModel._update_causal_mask
    forward = LlamaModel.forward


class QMemLlamaForCausalLM(QMemLlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = QMemLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.forward_step_id = 0

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

    forward = LlamaForCausalLM.forward
    prepare_inputs_for_generation = LlamaForCausalLM.prepare_inputs_for_generation


class QMemLlamaForSequenceClassification(QMemLlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = QMemLlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    forward = LlamaForSequenceClassification.forward


AutoConfig.register("qmemllama", QMemLlamaConfig)
AutoModel.register(QMemLlamaConfig, QMemLlamaModel)
AutoModelForCausalLM.register(QMemLlamaConfig, QMemLlamaForCausalLM)


