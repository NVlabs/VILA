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

# This file is modified from https://github.com/feifeibear/long-context-attention
# Implementation refers to USP Paper: https://arxiv.org/abs/2405.07719

import copy
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module

from .all_to_all import SeqAllToAll4D, SeqAllToAll5D
from .globals import get_ring_sp_pg, get_ring_type, get_ulysses_sp_pg
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

RING_IMPL_DICT = {
    "ring": ring_flash_attn_func,
    "zigzag": zigzag_ring_flash_attn_func,
    "strip": stripe_flash_attn_func,
    "ring_varlen": ring_flash_attn_varlen_func,
    "zigzag_ring_varlen": zigzag_ring_flash_attn_varlen_func,
}

RING_IMPL_QKVPACKED_DICT = {
    "ring": ring_flash_attn_qkvpacked_func,
    "zigzag": zigzag_ring_flash_attn_qkvpacked_func,
    "strip": stripe_flash_attn_qkvpacked_func,
    "ring_varlen": ring_flash_attn_varlen_qkvpacked_func,
    "zigzag_varlen": zigzag_ring_flash_attn_varlen_qkvpacked_func,
}


class HybridAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        ulysses_pg (ProcessGroup): ulysses process group
        ring_pg (ProcessGroup): ring process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        use_pack_qkv: bool = False,
        attention_warper: Module = None,
    ) -> None:

        super().__init__()
        self.ring_pg = get_ring_sp_pg()
        self.ulysses_pg = get_ulysses_sp_pg()

        self.use_pack_qkv = use_pack_qkv
        assert (
            self.ulysses_pg is not None or self.ring_pg is not None
        ), f"use set_pg_manager() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        if attention_warper is None:
            self.ring_attn_fn = RING_IMPL_DICT[get_ring_type()]
        else:
            self.ring_attn_fn = attention_warper

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *args: Any,
        attention_mask=None,
        dropout_p=0.0,
        softmax_scale=None,
        seqlens_in_batch=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        # 3 X (bs, seq_len/N, head_cnt, head_size) -> 3 X (bs, seq_len, head_cnt/N, head_size)
        # scatter 2, gather 1
        if self.use_pack_qkv:

            # TODO (Qinghao): To support packed qkv
            raise NotImplementedError("Packed qkv is not supported yet.")
            # (3*bs, seq_len/N, head_cnt, head_size)
            qkv = torch.cat([query, key, value]).continous()
            # (3*bs, seq_len, head_cnt/N, head_size)
            qkv = SeqAllToAll4D.apply(self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx)
            qkv = torch.chunk(qkv, 3, dim=0)
            out = self.ring_attn_fn(
                qkv[0],
                qkv[1],
                qkv[2],
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                group=self.ring_pg,
            )

        query_layer = SeqAllToAll4D.apply(self.ulysses_pg, query, self.scatter_idx, self.gather_idx)
        key_layer = SeqAllToAll4D.apply(self.ulysses_pg, key, self.scatter_idx, self.gather_idx)
        value_layer = SeqAllToAll4D.apply(self.ulysses_pg, value, self.scatter_idx, self.gather_idx)

        if attention_mask is not None:
            new_attention_mask = torch.cat([attention_mask] * dist.get_world_size(self.ulysses_pg), dim=1)

            out = self.ring_attn_fn(
                query_layer,
                key_layer,
                value_layer,
                *args,
                attention_mask=new_attention_mask,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                seqlens_in_batch=seqlens_in_batch,
                causal=causal,
                group=self.ring_pg,
            )
        else:
            out = self.ring_attn_fn(
                query_layer,
                key_layer,
                value_layer,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                group=self.ring_pg,
            )

        if type(out) == tuple:
            context_layer, _, _ = out
        else:
            context_layer = out

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(self.ulysses_pg, context_layer, self.gather_idx, self.scatter_idx)

        # out e.g., [s/p::h]
        return output


# TODO (Qinghao): To be supported
class HybridAttentionQKVPacked(torch.nn.Module):
    """Initialization.

    Arguments:
        ulysses_pg (ProcessGroup): ulysses process group
        ring_pg (ProcessGroup): ring process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        scatter_idx: int = 3,
        gather_idx: int = 1,
        ring_impl_type: str = "zigzag",
    ) -> None:

        super().__init__()

        self.ring_pg = get_ring_sp_pg()
        self.ulysses_pg = get_ulysses_sp_pg()

        assert (
            self.ulysses_pg is not None or self.ring_pg is not None
        ), f"use set_pg_manager() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx

        self.ring_attn_fn = RING_IMPL_QKVPACKED_DICT[ring_impl_type]

    def forward(
        self,
        qkv,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        *args: Any,
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """

        # scatter 3, gather 1

        world_size = dist.get_world_size(self.ulysses_pg)

        if world_size > 1 and dist.is_initialized():
            qkv = SeqAllToAll5D.apply(self.ulysses_pg, qkv, self.scatter_idx, self.gather_idx)

        out = self.ring_attn_fn(
            qkv,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_attn_probs=return_attn_probs,
            group=self.ring_pg,
        )

        # print(f"out {out.shape}")

        if type(out) == tuple:
            out = out[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
        # scatter 1, gather 2

        if world_size > 1 and dist.is_initialized():
            out = SeqAllToAll4D.apply(self.ulysses_pg, out, self.gather_idx, self.scatter_idx - 1)
        # out e.g., [s/p::h]
        return out


# TODO (Qinghao): To be supported
class AsyncHybridAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        ulysses_pg (ProcessGroup): ulysses process group
        ring_pg (ProcessGroup): ring process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        scatter_idx: int = 2,
        gather_idx: int = 1,
        ring_impl_type: str = "zigzag",
    ) -> None:

        super().__init__()
        self.ring_pg = get_ring_sp_pg()
        self.ulysses_pg = get_ulysses_sp_pg()

        self.stream = torch.cuda.Stream()
        self._async_op = True

        assert (
            self.ulysses_pg is not None or self.ring_pg is not None
        ), f"use set_pg_manager() first. Now ulysses pg {self.ulysses_pg} and ring pg {self.ring_pg}"
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.ring_attn_fn = RING_IMPL_DICT[ring_impl_type]

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        alibi_slopes=None,
        deterministic=False,
        return_attn_probs=False,
        *args: Any,
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer (bs, seqlen/P, hc, hs)
            key (Tensor): key input to the layer (bs, seqlen/P, hc_kv, hs)
            value (Tensor): value input to the layer (bs, seqlen/P, hc_kv, hs)
            args: other args

        Returns:
            * output (Tensor): context output
        """

        # un*ud = hc

        ulysses_degree = dist.get_world_size(self.ulysses_pg)

        bs, shard_seqlen, hc, hs = query.shape
        bs, shard_seqlen, hc_kv, hs = key.shape
        seq_len = shard_seqlen * ulysses_degree
        un = hc // ulysses_degree
        un_kv = hc_kv // ulysses_degree

        assert un_kv == un, f"un_kv {un_kv} un {un}"

        qkv = torch.cat([query, key, value]).contiguous()
        # (3*bs, seqlen/P, hc, hs) -> (hc, seqlen/P, 3*bs, hs) -> (un, ud, seqlen/P, 3*bs, hs), where hc = un*ud
        qkv_list = torch.unbind(qkv.transpose(0, 2).contiguous().reshape(un, ulysses_degree, shard_seqlen, 3 * bs, hs))
        # 3xall-to-all output buffer
        qkv_trans_list = [
            torch.zeros(
                ulysses_degree,
                1,
                shard_seqlen,
                3 * bs,
                hs,
                dtype=query.dtype,
                device=query.device,
            )
            for i in range(len(qkv_list))
        ]
        # last all-to-all buffter
        context_layer_list = [
            torch.zeros(
                ulysses_degree,
                1,
                shard_seqlen,
                bs,
                hs,
                dtype=query.dtype,
                device=query.device,
            )
            for i in range(len(qkv_list))
        ]

        comm_handle_list = []

        # un * (ud, shard_seqlen, 3*bs, hs)
        for i, qkv in enumerate(qkv_list):
            with torch.cuda.stream(self.stream):
                ret = dist.all_to_all_single(
                    qkv_trans_list[i],
                    qkv,
                    group=self.ulysses_pg,
                    async_op=self._async_op,
                )
            comm_handle_list.append(ret)

        last_comm_handle_list = []
        for i, qkv_trans in enumerate(qkv_trans_list):
            if comm_handle_list[i] is not None:
                comm_handle_list[i].wait()
            qkv_trans = (
                qkv_trans.reshape(seq_len, 3 * bs, 1, hs).transpose(0, 1).contiguous().reshape(3 * bs, seq_len, 1, hs)
            )

            # qkv_trans = all_to_all_4D_async(qkv, qkv_trans_list[i], self.scatter_idx, self.gather_idx, self.ulysses_pg)
            qkv_trans = torch.chunk(qkv_trans, 3, dim=0)

            out = self.ring_attn_fn(
                qkv_trans[0],
                qkv_trans[1],
                qkv_trans[2],
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size=window_size,
                alibi_slopes=alibi_slopes,
                deterministic=deterministic,
                return_attn_probs=return_attn_probs,
                group=self.ring_pg,
            )

            if type(out) == tuple:
                context_layer, _, _ = out
            else:
                context_layer = out

            # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)
            # scatter 1, gather 2

            context_layer = (
                context_layer.reshape(bs, ulysses_degree, shard_seqlen, 1, hs)
                .transpose(0, 3)
                .transpose(0, 1)
                .contiguous()
                .reshape(ulysses_degree, 1, shard_seqlen, bs, hs)
            )
            with torch.cuda.stream(self.stream):
                ret = dist.all_to_all_single(
                    context_layer_list[i],
                    context_layer,
                    group=self.ulysses_pg,
                    async_op=self._async_op,
                )
            last_comm_handle_list.append(ret)

        # hc = un * P
        # un x (hc = P, seq_len/P, bs, hs) -> (bs, seq_len, hc = P, hs)
        for i, ret in enumerate(last_comm_handle_list):
            if ret is not None:
                ret.wait()
            context_layer_list[i] = (
                context_layer_list[i]
                .reshape(ulysses_degree, shard_seqlen, bs, hs)
                .transpose(0, 2)
                .contiguous()
                .reshape(bs, shard_seqlen, ulysses_degree, hs)
            )

        output = torch.cat(context_layer_list, dim=2)
        return output

    def backward(self, *args, **kwargs):
        raise RuntimeError("Backward computation is not allowed for AsyncHybridAttention.")


