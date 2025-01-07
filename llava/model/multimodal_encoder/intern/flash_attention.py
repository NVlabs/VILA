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

# https://github.com/Dao-AILab/flash-attention/blob/v0.2.8/flash_attn/flash_attention.py
import torch
import torch.nn as nn
from einops import rearrange

try:  # v1
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
except:  # v2
    from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func as flash_attn_unpadded_qkvpacked_func

from flash_attn.bert_padding import pad_input, unpad_input


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0, device=None, dtype=None):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None, max_s=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert qkv.dtype in [torch.float16, torch.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, "b s ... -> (b s) ...")
                max_s = seqlen
                cu_seqlens = torch.arange(
                    0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32, device=qkv.device
                )
                output = flash_attn_unpadded_qkvpacked_func(
                    qkv,
                    cu_seqlens,
                    max_s,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                )
                output = rearrange(output, "(b s) ... -> b s ...", b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, "b s three h d -> b s (three h d)")
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, "nnz (three h d) -> nnz three h d", three=3, h=nheads)
                output_unpad = flash_attn_unpadded_qkvpacked_func(
                    x_unpad,
                    cu_seqlens,
                    max_s,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale,
                    causal=causal,
                )
                output = rearrange(
                    pad_input(rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, batch_size, seqlen),
                    "b s (h d) -> b s h d",
                    h=nheads,
                )
        else:
            assert max_s is not None
            output = flash_attn_unpadded_qkvpacked_func(
                qkv,
                cu_seqlens,
                max_s,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale,
                causal=causal,
            )

        return output, None


