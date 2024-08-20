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

from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import transformers
from einops import rearrange
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from torch import nn
from transformers import LlamaConfig
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import LlamaAttention, _get_unpad_data, apply_rotary_pos_emb

from llava.train.sequence_parallel.globals import get_pg_manager, get_ring_sp_pg, get_ring_type, get_ulysses_sp_pg

from .hybrid_attn import HybridAttention
from .ring import (
    ring_flash_attn_func,
    ring_flash_attn_qkvpacked_func,
    ring_flash_attn_varlen_func,
    ring_flash_attn_varlen_qkvpacked_func,
    stripe_flash_attn_func,
    stripe_flash_attn_qkvpacked_func,
    zigzag_ring_flash_attn_func,
    zigzag_ring_flash_attn_qkvpacked_func,
    zigzag_ring_flash_attn_varlen_func,
    zigzag_ring_flash_attn_varlen_qkvpacked_func,
)
from .ulysses_attn import UlyssesAttention


def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length, seqlens_in_batch=None):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask, seqlens_in_batch=seqlens_in_batch)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    num_query_heads = query_layer.shape[2]
    key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k)
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim), indices_k
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(
            query_layer.reshape(batch_size * kv_seq_len, num_query_heads, head_dim), indices_k
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        attention_mask = attention_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )


def flash_attn_varlen_func_helper(
    self,
    query_states,
    key_states,
    value_states,
    query_length,
    attention_mask=None,
    dropout_p=0.0,
    softmax_scale=None,
    seqlens_in_batch=None,
    causal=None,
):
    batch_size = query_states.shape[0]
    assert (
        attention_mask.shape[1] == query_states.shape[1]
    ), f"attention_mask.shape {attention_mask.shape}, query_states.shape {query_states.shape}"

    # overwrite query_length with the actual length of the sequence after seq parallel communciation
    query_length = attention_mask.shape[1]

    query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
        query_states, key_states, value_states, attention_mask, query_length, seqlens_in_batch=seqlens_in_batch
    )

    cu_seqlens_q, cu_seqlens_k = cu_seq_lens
    max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

    attn_output_unpad = flash_attn_varlen_func(
        query_states,
        key_states,
        value_states,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_in_batch_q,
        max_seqlen_k=max_seqlen_in_batch_k,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        causal=self.is_causal,
        # deterministic=True
    )

    attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    return attn_output


def hybrid_attn_varlen_func_helper(
    self,
    query_states,
    key_states,
    value_states,
    query_length,
    attention_mask=None,
    dropout_p=0.0,
    softmax_scale=None,
    seqlens_in_batch=None,
    causal=None,
    group=None,
):
    batch_size = query_states.shape[0]
    # assert (
    #     attention_mask.shape[1] == query_states.shape[1]
    # ), f"attention_mask.shape {attention_mask.shape}, query_states.shape {query_states.shape}"

    # overwrite query_length with the actual length of the sequence after seq parallel communication
    query_length = attention_mask.shape[1]

    # print("query_states", query_states.shape, query_states.device)
    # print("attn_mask", attention_mask.shape, attention_mask.device, attention_mask)

    query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
        query_states, key_states, value_states, attention_mask, query_length, seqlens_in_batch=None
    )

    # print("max_seq_lens", max_seq_lens)

    # print("after_upad_query_states", query_states.shape, query_states.device)
    # exit()

    cu_seq_lens = cu_seq_lens[0]

    # print("rank", dist.get_rank(), "cu_seq_lens", cu_seq_lens)
    # exit()

    ring_type = get_ring_type()
    if ring_type == "ring_varlen":
        attn_output_unpad = ring_flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seq_lens,
            max_seq_lens[0],
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=self.is_causal,
            group=group,
        )
    elif ring_type == "zigzag_ring_varlen":
        attn_output_unpad = zigzag_ring_flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seq_lens,
            max_seq_lens[0],
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=self.is_causal,
            group=group,
        )
    else:
        raise ValueError(f"Invalid ring_type: {ring_type}")

    # print(dist.get_rank(), "finish ring_flash_attn_varlen_func")

    attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    return attn_output


def __init__(self, config: LlamaConfig):
    nn.Module.__init__(self)
    self.config = config
    self.attention_dropout = config.attention_dropout
    self.hidden_size = config.hidden_size
    self.num_heads = config.num_attention_heads
    self.head_dim = self.hidden_size // self.num_heads
    self.num_key_value_heads = config.num_key_value_heads
    self.num_key_value_groups = self.num_heads // self.num_key_value_heads
    self.max_position_embeddings = config.max_position_embeddings
    self.rope_theta = config.rope_theta
    self.is_causal = True

    if (self.head_dim * self.num_heads) != self.hidden_size:
        raise ValueError(
            f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
            f" and `num_heads`: {self.num_heads})."
        )

    self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
    self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
    self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
    self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
    self._init_rope()

    # wrap two potential "local-attention" up with DeepSpeed Ulysses logic.
    self.ulysses_attn_varlen_func = UlyssesAttention(self.flash_attn_varlen_func_helper, get_ulysses_sp_pg())
    self.ulysses_attn_func = UlyssesAttention(flash_attn_func, get_ulysses_sp_pg())

    # Using Hybrid Sequence Parallelism
    self.ring_enabled = get_ring_sp_pg() is not None
    if self.ring_enabled:
        self.hybrid_attn_varlen_func = HybridAttention(attention_warper=self.hybrid_attn_varlen_func_helper)
        self.hybrid_attn_func = HybridAttention()


def _flash_attention_forward(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    dropout=0.0,
    softmax_scale=None,
    seqlens_in_batch=None,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        attention_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`int`, *optional*):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
    """
    # Contains at least one padding token in the sequence
    try:
        assert attention_mask is not None
    except AssertionError:
        print("attention_mask is None")

    if self.ring_enabled:
        if attention_mask is not None:
            attn_output = self.hybrid_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                query_length,
                attention_mask=attention_mask,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                seqlens_in_batch=seqlens_in_batch,
                # casual=self.is_causal,
            )
        else:
            attn_output = self.hybrid_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=self.is_causal,
            )
    else:
        if attention_mask is not None:
            attn_output = self.ulysses_attn_varlen_func(
                query_states,
                key_states,
                value_states,
                query_length,
                attention_mask=attention_mask,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                seqlens_in_batch=seqlens_in_batch,
                # casual=self.is_causal,
            )
        else:
            attn_output = self.ulysses_attn_func(
                query_states,
                key_states,
                value_states,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=self.is_causal,
            )

    return attn_output


def new_decoder_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    seqlens_in_batch: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


def new_llamamodel_forward(
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
    seqlens_in_batch: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    past_key_values_length = 0
    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None

    # embed positions
    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache,
                seqlens_in_batch,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                seqlens_in_batch=seqlens_in_batch,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def apply_hybrid_attn_monkey_patch_llama():
    # transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward = new_flash_attn_forward

    # transformers.models.llama.modeling_llama.LlamaDecoderLayer.forward = new_decoder_forward
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__ = __init__
    transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward = _flash_attention_forward
