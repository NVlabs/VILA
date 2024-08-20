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

import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import transformers
from deepspeed import comm as dist
from einops import rearrange
from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
from torch import nn
from torch.profiler import ProfilerActivity, profile, record_function
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb

_SEQUENCE_PARALLEL_GROUP = None
_SEQUENCE_PARALLEL_SIZE = -1
_SEQUENCE_PARALLEL_RANK = -1

from deepspeed_distributed_attention import DistributedAttention
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from transformers_replace.models.llama.modeling_llama import LlamaAttention


# Modified from transformers_replace/models/llama/modeling_llama.py FlashAttention2
def __init__(self, config: LlamaConfig):
    super().__init__()
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
    self.distributed_flash_attn_varlen_func = DistributedAttention(flash_attn_varlen_func, _SEQUENCE_PARALLEL_GROUP)
    self.distributed_flash_attn_func = DistributedAttention(flash_attn_func, _SEQUENCE_PARALLEL_GROUP)


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
    if attention_mask is not None:
        # Shape: b, s, nh, hdim
        print(f"Input shape to _flash_attention_forward: {query_states.shape}")
        batch_size = query_states.shape[0]
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
            query_states, key_states, value_states, attention_mask, query_length, seqlens_in_batch=seqlens_in_batch
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        # Reshape it to s, b, hh, hdim to use DeepSpeed
        # DL: Alternatively, we can also modify the permuting dimensions of DeepSpeed backend.
        # But this may decrease the portability of the code.

        query_states = query_states.permute(1, 0, 2, 3).contiguous()
        key_states = key_states.permute(1, 0, 2, 3).contiguous()
        value_states = value_states.permute(1, 0, 2, 3).contiguous()

        attn_output_unpad = self.distributed_flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=self.is_causal,
        )

        # reshape it back to b, s, nh, hdim
        query_states = query_states.permute(1, 0, 2, 3).contiguous()
        key_states = key_states.permute(1, 0, 2, 3).contiguous()
        value_states = value_states.permute(1, 0, 2, 3).contiguous()
        attn_output_unpad = attn_output_unpad.permute(1, 0, 2, 3).contiguous()

        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    else:
        query_states = query_states.permute(1, 0, 2, 3).contiguous()
        key_states = key_states.permute(1, 0, 2, 3).contiguous()
        value_states = value_states.permute(1, 0, 2, 3).contiguous()

        attn_output = self.distributed_flash_attn_func(
            query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal
        )

        # reshape it back to b, s, nh, hdim
        query_states = query_states.permute(1, 0, 2, 3).contiguous()
        key_states = key_states.permute(1, 0, 2, 3).contiguous()
        value_states = value_states.permute(1, 0, 2, 3).contiguous()
        attn_output = attn_output.permute(1, 0, 2, 3).contiguous()

    return attn_output


def initialize_sequence_parallel(sequence_parallel_size):
    # first check torch distributed group init and set device accordingly;
    # (DL) TODO: Whether this can be skipped in DeepSpeed.
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(
                "torch distributed is already initialized, " "skipping initialization ...",
                flush=True,
            )
    else:
        if int(os.environ["RANK"]) == 0:
            print("Initializing Torch distributed.")
        dist.init_distributed(dist_backend="nccl", dist_init_required=True)
        local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
        global_world_size = dist.get_world_size()

        torch.cuda.set_device(dist.get_rank() % local_world_size)

    world_size = dist.get_world_size()
    num_sequence_parallel_groups = world_size // sequence_parallel_size

    global _SEQUENCE_PARALLEL_GROUP
    global _SEQUENCE_PARALLEL_SIZE
    global _SEQUENCE_PARALLEL_RANK
    for i in range(num_sequence_parallel_groups):
        ranks = range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
        group = dist.new_group(ranks)
        if rank in ranks:
            _SEQUENCE_PARALLEL_GROUP = group
            _SEQUENCE_PARALLEL_RANK = ranks.index(rank)
            _SEQUENCE_PARALLEL_SIZE = len(ranks)

    if dist.get_rank() == 0:
        print("************ Finish sequence pralell group Initialization. ***********")


def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_size():
    """Get the size of the sequence parallel group the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_SIZE


def get_sequence_parallel_rank():
    """Get the rank of this process in the sequence parallel group the caller rank belongs to."""
    return _SEQUENCE_PARALLEL_RANK


def replace_llama_attn_with_dpsp_attn():

    transformers.models.llama.modeling_llama.LlamaAttention.__init__ = __init__
    transformers.models.llama.modeling_llama.LlamaFlashAttention2._flash_attention_forward = _flash_attention_forward
