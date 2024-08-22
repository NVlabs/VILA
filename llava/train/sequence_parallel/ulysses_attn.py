# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# This file is modified from https://github.com/feifeibear/long-context-attention
# Implementation refers to USP Paper: https://arxiv.org/abs/2405.07719
# This file is also partly modified from https://github.com/microsoft/DeepSpeed
# Implementation refers to Ulysses Paper: https://arxiv.org/abs/2309.14509

import copy
from typing import Any, Tuple

import deepspeed.comm as dist
import torch
import torch.distributed as torch_dist
from flash_attn import flash_attn_func
from torch import Tensor
from torch.nn import Module

from llava.train.sequence_parallel.globals import get_ulysses_seq_len, get_ulysses_sp_rank, get_ulysses_sp_size

from .all_to_all import SeqAllGather, SeqAllToAll4D, SeqAllToAll5D


class _ExpandKVFunction(torch.autograd.Function):
    """
    Copy the KV head repeat times to extend sequence parallel support for Ulysses.

    Args:
        kv: input kv.
        repeat_times: the repeat number of each head.
        num_head_dim: the dimension of head number.
    """

    @staticmethod
    def forward(ctx, k, v, repeat_times, num_head_dim):

        kv_shape = k.shape
        num_heads_kv = kv_shape[num_head_dim]

        ctx.num_head_dim = num_head_dim
        ctx.num_heads_kv = num_heads_kv

        # here we construct a repeat index to indicate which dim should copy
        repeat_index = [1] * k.ndim
        repeat_index[num_head_dim] = repeat_times

        # split the kv into head num splits
        k_splits = torch.chunk(k, chunks=num_heads_kv, dim=num_head_dim)
        v_splits = torch.chunk(v, chunks=num_heads_kv, dim=num_head_dim)
        k_repeats, v_repeats = [], []
        # for each split, we copy it to repeat_times copys.
        for split in k_splits:
            k_split_repeat = split.repeat(repeat_index)
            k_repeats.append(k_split_repeat)

        for split in v_splits:
            v_split_repeat = split.repeat(repeat_index)
            v_repeats.append(v_split_repeat)

        return torch.cat(k_repeats, dim=num_head_dim), torch.cat(v_repeats, dim=num_head_dim)

    @staticmethod
    def backward(ctx, grad_output_k, grad_output_v):
        """
        For backward, we sum the copy head inside a query group.
        """

        num_head_dim = ctx.num_head_dim
        num_heads_kv = ctx.num_heads_kv

        # we split the grad into query groups splits.
        grad_output_k_splits = torch.chunk(grad_output_k, chunks=num_heads_kv, dim=num_head_dim)
        grad_output_v_splits = torch.chunk(grad_output_v, chunks=num_heads_kv, dim=num_head_dim)

        grad_output_k_sums, grad_output_v_sums = [], []
        # for each split, we sum the head
        for grad_output_k_split in grad_output_k_splits:
            grad_output_k_sum = grad_output_k_split.sum(dim=num_head_dim, keepdim=True)
            grad_output_k_sums.append(grad_output_k_sum)

        for grad_output_v_split in grad_output_v_splits:
            grad_output_v_sum = grad_output_v_split.sum(dim=num_head_dim, keepdim=True)
            grad_output_v_sums.append(grad_output_v_sum)

        # then we concat the split sums on the num_head_dim dimension.
        grad_k = torch.cat(grad_output_k_sums, dim=num_head_dim)
        grad_v = torch.cat(grad_output_v_sums, dim=num_head_dim)

        return grad_k, grad_v, None, None


expandKV = _ExpandKVFunction.apply


class UlyssesAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        sequence_process_group (ProcessGroup): sequence parallel process group
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        local_attention: Module,
        sequence_process_group: dist.ProcessGroup = None,
        scatter_idx: int = 2,
        gather_idx: int = 1,
    ) -> None:

        super().__init__()
        self.local_attn = local_attention
        self.spg = sequence_process_group
        self.scatter_idx = scatter_idx
        self.gather_idx = gather_idx
        self.ulysses_degree = get_ulysses_sp_size()

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
        # TODO Merge three alltoall calls into one
        # TODO (Reza): change the api on the megatron-deepspeed side so that we only receive all data (q,k, and v) together!
        # in shape : e.g.,  [s/p:h:]
        # (bs, seq_len/N, head_cnt, head_size) -> (bs, seq_len, head_cnt/N, head_size)

        # KV Replication for GQA
        head_dim = 2
        num_head_kv = key.shape[head_dim]
        if self.ulysses_degree > num_head_kv:
            assert self.ulysses_degree % num_head_kv == 0, "Ulysses require num_head_kv to be dividable by sp degree."
            key, value = expandKV(key, value, self.ulysses_degree // num_head_kv, head_dim)

        # scatter 2, gather 1
        q = SeqAllToAll4D.apply(self.spg, query, self.scatter_idx, self.gather_idx)
        k = SeqAllToAll4D.apply(self.spg, key, self.scatter_idx, self.gather_idx)
        v = SeqAllToAll4D.apply(self.spg, value, self.scatter_idx, self.gather_idx)

        if attention_mask is not None:
            local_attention_mask = copy.deepcopy(attention_mask)
            shard_seqlen = local_attention_mask.size(1)
            ulysses_seq_len = get_ulysses_seq_len()
            max_global_length = max(ulysses_seq_len)
            global_attention_mask_list = []
            for i in range(get_ulysses_sp_size()):
                if i == get_ulysses_sp_rank():
                    global_attention_mask_list.append(
                        torch.cat(
                            [
                                local_attention_mask,
                                torch.zeros(
                                    (local_attention_mask.size(0), max_global_length - shard_seqlen),
                                    dtype=local_attention_mask.dtype,
                                    device=local_attention_mask.device,
                                ),
                            ],
                            dim=1,
                        )
                    )
                else:
                    global_attention_mask_list.append(
                        torch.zeros(
                            (local_attention_mask.size(0), max_global_length),
                            dtype=local_attention_mask.dtype,
                            device=local_attention_mask.device,
                        )
                    )

            global_attention_mask = torch.stack(global_attention_mask_list, dim=0)
            torch_dist.all_reduce(global_attention_mask, group=self.spg)
            torch_dist.barrier(group=self.spg)
            new_global_attention_mask_list = list(torch.unbind(global_attention_mask, dim=0))
            # Unpad the global attention mask list and concatenate them
            for i in range(len(new_global_attention_mask_list)):
                new_global_attention_mask_list[i] = new_global_attention_mask_list[i][:, : ulysses_seq_len[i]]
            global_attention_mask = torch.cat(new_global_attention_mask_list, dim=1)
            context_layer = self.local_attn(
                q,
                k,
                v,
                *args,
                attention_mask=global_attention_mask,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                seqlens_in_batch=seqlens_in_batch,
                causal=causal,
            )
        else:
            context_layer = self.local_attn(
                q,
                k,
                v,
                *args,
                dropout_p=dropout_p,
                softmax_scale=softmax_scale,
                causal=causal,
                # window_size=window_size,
                # alibi_slopes=alibi_slopes,
                # deterministic=deterministic,
                # return_attn_probs=return_attn_probs,
            )
        if isinstance(context_layer, tuple):
            context_layer = context_layer[0]

        # (bs, seq_len, head_cnt/N, head_size) -> (bs, seq_len/N, head_cnt, head_size)

        # scatter 1, gather 2
        output = SeqAllToAll4D.apply(self.spg, context_layer, self.gather_idx, self.scatter_idx)

        # out e.g., [s/p::h]
        return output
