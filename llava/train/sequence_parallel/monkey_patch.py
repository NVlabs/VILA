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

import inspect
import os
from typing import List, Optional, Tuple, Union

import torch
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import pad_input
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import _upad_input
from transformers.utils import is_flash_attn_greater_or_equal

from llava.model.utils.packing import _get_unpad_data
from llava.train.sequence_parallel.globals import get_ring_sp_pg, get_ring_type, get_ulysses_sp_pg
from llava.train.sequence_parallel.hybrid_attn import HybridAttention
from llava.train.sequence_parallel.ring import ring_flash_attn_varlen_func, zigzag_ring_flash_attn_varlen_func
from llava.train.sequence_parallel.ulysses_attn import UlyssesAttention


def _ulysses_attn_varlen_func(
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

    # overwrite query_length with the actual length of the sequence after SP communciation
    query_length = attention_mask.shape[1]

    query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
        query_states, key_states, value_states, attention_mask, query_length
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
        causal=True,
    )

    attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    return attn_output


def _hybrid_attn_varlen_func(
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

    # overwrite query_length with the actual length of the sequence after SP communciation
    query_length = attention_mask.shape[1]
    _get_unpad_data.seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)

    query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = _upad_input(
        query_states, key_states, value_states, attention_mask, query_length
    )

    cu_seq_lens = cu_seq_lens[0]

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
            causal=True,
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
            causal=True,
            group=group,
        )
    else:
        raise ValueError(f"Invalid ring_type: {ring_type}")

    attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)

    return attn_output


def _flash_attention_forward(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
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
        dropout (`float`):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
        use_top_left_mask (`bool`, defaults to `False`):
            flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference.
        softcap (`float`, *optional*):
            Softcap for the attention logits, used e.g. in gemma2.
        deterministic (`bool`, *optional*):
            Determines if the deterministic option introduced in flash_attn>=2.4.1 is enabled.
    """
    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__.
        causal = is_causal and query_length != 1

    # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
    use_sliding_windows = (
        "window_size" in list(inspect.signature(flash_attn_func).parameters)
        and sliding_window is not None
        and key_states.shape[1] > sliding_window
    )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    if is_flash_attn_greater_or_equal("2.4.1"):
        if deterministic is None:
            deterministic = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
        flash_kwargs["deterministic"] = deterministic

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    ring_enabled = get_ring_sp_pg() is not None

    if attention_mask is not None:
        if ring_enabled:
            attn_output = HybridAttention(attention_warper=_hybrid_attn_varlen_func)(
                query_states,
                key_states,
                value_states,
                query_length,
                attention_mask=attention_mask,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                seqlens_in_batch=_get_unpad_data.seqlens_in_batch,
            )
        else:
            attn_output = UlyssesAttention(_ulysses_attn_varlen_func, get_ulysses_sp_pg())(
                query_states,
                key_states,
                value_states,
                query_length,
                attention_mask=attention_mask,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                seqlens_in_batch=_get_unpad_data.seqlens_in_batch,
            )
    else:
        if ring_enabled:
            attn_output = HybridAttention()(
                query_states,
                key_states,
                value_states,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=is_causal,
            )
        else:
            attn_output = UlyssesAttention(flash_attn_func, get_ulysses_sp_pg())(
                query_states,
                key_states,
                value_states,
                query_length,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
            )
    return attn_output


def _update_causal_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    cache_position: torch.Tensor,
    past_key_values: Cache,
    output_attentions: bool,
):
    return attention_mask


