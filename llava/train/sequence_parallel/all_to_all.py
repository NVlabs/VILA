# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# This file is modified from https://github.com/feifeibear/long-context-attention
# Implementation refers to USP Paper: https://arxiv.org/abs/2405.07719

from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module

from llava.train.sequence_parallel.globals import (
    get_ulysses_seq_len,
    get_ulysses_sp_pg,
    get_ulysses_sp_rank,
    get_ulysses_sp_size,
    set_ulysses_seq_len,
)


def all_to_all_4D(input: torch.tensor, scatter_idx: int = 2, gather_idx: int = 1, group=None) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert input.dim() == 4, f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    # seq_world_size = dist.get_world_size(group)
    # (DL): Change to ulysses size to handle hybrid parallelism.
    seq_world_size = get_ulysses_sp_size()
    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        # (Dacheng): For multi-modality use case, sequence length is different, causing unknown behavior for a2a.
        # Pad it first.
        # (Dacheng): This will trigger for each attention to make sure the second a2a is correct.
        # (TODO) Maybe can optimize to per forward call.
        ulysses_seq_len = [torch.zeros(1, dtype=torch.int64, device=input.device) for _ in range(get_ulysses_sp_size())]
        dist.barrier(group=get_ulysses_sp_pg())
        dist.all_gather(ulysses_seq_len, torch.tensor(shard_seqlen, device=input.device), group=get_ulysses_sp_pg())
        set_ulysses_seq_len(ulysses_seq_len)

        max_global_length = max(ulysses_seq_len)
        # pad to the second dimension to the longest
        input = torch.nn.functional.pad(input, (0, 0, 0, 0, 0, max_global_length - shard_seqlen))

        seqlen = max_global_length * seq_world_size  # shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            # input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            input.reshape(bs, max_global_length, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        dist.barrier(group=group)
        dist.all_to_all_single(output, input_t, group=group)

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # then we will unpad it back
        output_list = torch.split(output, max_global_length, dim=0)
        assert len(output_list) == get_ulysses_sp_size()
        unpadded_output_list = [_output[: _seqlen.item()] for _output, _seqlen in zip(output_list, ulysses_seq_len)]

        # Concatenate the unpadded tensors back together
        output = torch.cat(unpadded_output_list)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, sum(ulysses_seq_len), shard_hc, hs)

        # assert False

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        ulysses_seq_len = get_ulysses_seq_len()
        assert ulysses_seq_len is not None, "the second a2a (scatter 1, gather 2) is called at first."
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, _, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size

        # First we need to recover how to pad
        max_global_length = max(ulysses_seq_len)

        unpadded_input_list = torch.split(input, ulysses_seq_len, dim=1)
        padded_input_list = [
            torch.nn.functional.pad(_unpadded_input, (0, 0, 0, 0, 0, max_global_length - _unpadded_input.shape[1]))
            for _unpadded_input in unpadded_input_list
        ]
        input = torch.cat(padded_input_list, dim=1)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, max_global_length, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, max_global_length, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        dist.barrier(group=group)
        dist.all_to_all_single(output, input_t, group=group)

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, max_global_length, bs, hs)

        # unpad the output
        self_length = ulysses_seq_len[get_ulysses_sp_rank()]
        # print(f"Self length {self_length}")
        output = output[:, :self_length, :, :]

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, self_length, hc, hs)
        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
    ) -> Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return all_to_all_4D(input, scatter_idx, gather_idx, group=group)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (
            None,
            SeqAllToAll4D.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx),
            None,
            None,
        )


def all_to_all_5D(input: torch.tensor, scatter_idx: int = 3, gather_idx: int = 1, group=None) -> torch.tensor:
    """
    all-to-all for QKV
    forward (bs, seqlen/N, 3, hc, hs) -> (bs, seqlen, 3, hc/N, hs)

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, 3, hc, hs)
    """
    assert input.dim() == 5, f"input must be 5D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 3 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, 3, hc, hs) output: (bs, seqlen, 3, hc/P, hs)
        bs, shard_seqlen, t_cnt, hc, hs = input.shape

        assert t_cnt == 3
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, 3, hc, hs) -reshape-> (bs, seq_len/P, 3, P, hc/P, hs) -transpose(0,3)-> (P, seq_len/P, 3, bs, hc/P, hs)
        input_t = input.reshape(bs, shard_seqlen, 3, seq_world_size, shard_hc, hs).transpose(0, 3).contiguous()

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, 3, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, 3, bs, hc/P, hs) scatter head
        dist.barrier(group=group)
        dist.all_to_all_single(output, input_t, group=group)

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, 3, bs, shard_hc, hs)

        # (seq_len, 3, bs, hc/P, hs) -trans-> (bs, seq_len, 3, hc/P, hs)
        output = output.transpose(0, 2).transpose(1, 2).contiguous()

        return output.reshape(bs, seqlen, 3, shard_hc, hs).contiguous()
    elif scatter_idx == 1 and gather_idx == 3:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, _, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, 3, hc/P, hs) -reshape-> (bs, P, seq_len/P, 3, hc/P, hs) -transpose(0, 4)-> (hc/P, P, seqlen/P, 3, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, 3, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, 3, shard_hc, hs)
            .transpose(0, 4)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, 3, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        dist.barrier(group=group)
        dist.all_to_all_single(output, input_t, group=group)

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, 3, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 3).contiguous()

        return output.reshape(bs, shard_seqlen, 3, hc, hs).contiguous()
    else:
        raise RuntimeError("scatter_idx must be 1 or 3 and gather_idx must be 1 or 3")


class SeqAllToAll5D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int = 3,
        gather_idx: int = 1,
    ) -> Tensor:

        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return all_to_all_5D(input, scatter_idx, gather_idx, group=group)

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (
            None,
            SeqAllToAll5D.apply(ctx.group, *grad_output, ctx.gather_idx, ctx.scatter_idx),
            None,
            None,
        )


class SeqAllGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, group: dist.ProcessGroup, input: Any) -> Tensor:
        # ctx.group = group
        ctx.save_for_backward(input[0])
        all_gather_list = input[0]
        all_gather_tensor = input[1]
        dist.all_gather(all_gather_list, all_gather_tensor, group=group)
        # torch.concat
        return torch.stack(all_gather_list, dim=0)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[None, Tensor]:
        (tensor,) = ctx.saved_tensors
        return None, (None, tensor)
