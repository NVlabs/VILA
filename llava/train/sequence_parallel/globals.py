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

import os

import deepspeed.comm as dist
import torch


class Singleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if not self.__initialized:
            self.__initialized = True


class ProcessGroupManager(Singleton):
    """
    sp_degree = sp_ring_degree x sp_ulysses_degree
    """

    def __init__(self, ulysses_degree, ring_degree, dp_degree, use_ulysses_low, ring_type):
        if not hasattr(self, "__initialized"):
            super().__init__()
            self.ulysses_degree = ulysses_degree
            self.ring_type = ring_type
            self.ulysses_seq_len = None

            self.ring_degree = ring_degree
            self.sp_degree = ring_degree * ulysses_degree
            self.dp_degree = dp_degree

            self.rank = dist.get_rank()

            if self.ring_degree == 1:
                # Using Ulysses Sequence Parallelism only
                num_ulysses_pgs = self.dp_degree
                self.ring_pg = None
                self.ring_rank = None

                for i in range(num_ulysses_pgs):
                    ulysses_ranks = list(range(i * self.ulysses_degree, (i + 1) * self.ulysses_degree))
                    group = dist.new_group(ulysses_ranks)
                    if self.rank in ulysses_ranks:
                        self.ulysses_pg = group

                for sp_rank in range(self.sp_degree):
                    dp_ranks = list(range(sp_rank, self.dp_degree * self.sp_degree, self.sp_degree))
                    group = dist.new_group(dp_ranks)
                    if self.rank in dp_ranks:
                        self.dp_pg = group

                self.ulysses_rank = dist.get_rank(self.ulysses_pg)
                self.sp_rank = self.ulysses_rank
                self.dp_rank = dist.get_rank(self.dp_pg)
                self.sp_pg = self.ulysses_pg

                print(f"GPU {torch.cuda.current_device()} Ulysses rank: {self.ulysses_rank} out of {self.sp_degree}")
            else:
                # Using Hybrid Sequence Parallelism
                assert self.ring_degree > 1
                num_ulysses_pgs = self.ring_degree  # world_size // self.ulysses_degree
                num_ring_pgs = self.ulysses_degree  # world_size // self.ring_degree

                # Set up process groups
                if use_ulysses_low:
                    for dp_rank in range(dp_degree):
                        offset = dp_rank * self.sp_degree
                        for i in range(num_ulysses_pgs):
                            ulysses_ranks = list(
                                range(
                                    i * self.ulysses_degree + offset,
                                    (i + 1) * self.ulysses_degree + offset,
                                )
                            )
                            group = dist.new_group(ulysses_ranks)
                            if self.rank in ulysses_ranks:
                                self.ulysses_pg = group

                        for i in range(num_ring_pgs):
                            ring_ranks = list(range(i + offset, self.sp_degree + offset, num_ring_pgs))
                            group = dist.new_group(ring_ranks)
                            if self.rank in ring_ranks:
                                self.ring_pg = group

                else:
                    for dp_rank in range(dp_degree):
                        offset = dp_rank * self.sp_degree
                        for i in range(num_ring_pgs):
                            ring_ranks = list(range(i * self.ring_degree + offset, (i + 1) * self.ring_degree + offset))
                            group = dist.new_group(ring_ranks)
                            if self.rank in ring_ranks:
                                self.ring_pg = group

                        for i in range(num_ulysses_pgs):
                            ulysses_ranks = list(range(i + offset, self.sp_degree + offset, num_ulysses_pgs))
                            group = dist.new_group(ulysses_ranks)
                            if self.rank in ulysses_ranks:
                                self.ulysses_pg = group

                for sp_rank in range(self.sp_degree):
                    dp_ranks = list(range(sp_rank, self.dp_degree * self.sp_degree, self.sp_degree))
                    group = dist.new_group(dp_ranks)
                    if self.rank in dp_ranks:
                        self.dp_pg = group

                for i in range(self.dp_degree):
                    sp_ranks = list(range(i * self.sp_degree, (i + 1) * self.sp_degree))
                    group = dist.new_group(sp_ranks)
                    if self.rank in sp_ranks:
                        self.sp_pg = group

                self.ulysses_rank = dist.get_rank(self.ulysses_pg)
                self.ring_rank = dist.get_rank(self.ring_pg)
                self.dp_rank = dist.get_rank(self.dp_pg)

                if use_ulysses_low:
                    self.sp_rank = self.ulysses_rank + self.ring_rank * self.ulysses_degree
                else:
                    self.sp_rank = self.ring_rank + self.ulysses_rank * self.ring_degree

                print(
                    f"Rank {self.rank}, GPU {torch.cuda.current_device()} Hybrid SP rank: {self.sp_rank} out of {self.sp_degree} (Ulysses: {self.ulysses_rank}/{self.ulysses_degree}, Ring: {self.ring_rank}/{self.ring_degree})"
                )

            print("--------------ProcessGroupManager Initialized---------------------")


PROCESS_GROUP_MANAGER = None


def set_pg_manager(sp_degree, sp_ring_degree=1, use_ulysses_low=True, ring_type=None):
    """
    Set the process group manager for sequence parallelism.
    sp_degree = sp_ring_degree x sp_ulysses_degree
    """

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

        torch.cuda.set_device(dist.get_rank() % local_world_size)

    world_size = dist.get_world_size()

    assert sp_degree <= world_size
    assert world_size % sp_degree == 0, f"world_size {world_size} % sp_degree {sp_degree} != 0"

    if sp_ring_degree < 1:
        sp_ring_degree = 1

    sp_ulysses_degree = sp_degree // sp_ring_degree
    assert sp_degree % sp_ring_degree == 0, f"sp_degree {sp_degree} % sp_ring_degree {sp_ring_degree} != 0"

    dp_degree = world_size // sp_degree

    # Init the process group manager
    global PROCESS_GROUP_MANAGER
    PROCESS_GROUP_MANAGER = ProcessGroupManager(
        sp_ulysses_degree, sp_ring_degree, dp_degree, use_ulysses_low, ring_type
    )


def get_pg_manager():
    return PROCESS_GROUP_MANAGER


def get_sequence_parallel_size():
    """Get the size of the sequence parallel group."""
    return PROCESS_GROUP_MANAGER.sp_degree


def get_sequence_parallel_rank():
    """Get the rank of this process in the sequence parallel group the caller rank belongs to."""
    return PROCESS_GROUP_MANAGER.sp_rank


def get_sequence_parallel_pg():
    """Get the overall sequence parallel process group (include Ring and Ulysses)."""
    return PROCESS_GROUP_MANAGER.sp_pg


def get_ulysses_sp_size():
    """Get the size of the Ulysses sequence parallel group."""
    return PROCESS_GROUP_MANAGER.ulysses_degree


def get_ulysses_seq_len():
    """Get the size of the Ulysses sequence parallel group."""
    return PROCESS_GROUP_MANAGER.ulysses_seq_len


def set_ulysses_seq_len(seq_len):
    """Get the size of the Ulysses sequence parallel group."""
    PROCESS_GROUP_MANAGER.ulysses_seq_len = seq_len


def get_ulysses_sp_rank():
    """Get the rank of this process in the Ulysses sequence parallel group the caller rank belongs to."""
    return PROCESS_GROUP_MANAGER.ulysses_rank


def get_ulysses_sp_pg():
    """Get the Ulysses sequence parallel process group."""
    return PROCESS_GROUP_MANAGER.ulysses_pg


def get_ring_sp_size():
    """Get the size of the RingAttn sequence parallel group."""
    return PROCESS_GROUP_MANAGER.ring_degree


def get_ring_sp_rank():
    """Get the rank of this process in the RingAttn sequence parallel group the caller rank belongs to."""
    return PROCESS_GROUP_MANAGER.ring_rank


def get_ring_sp_pg():
    """Get the RingAttn sequence parallel process group."""
    return PROCESS_GROUP_MANAGER.ring_pg


def get_ring_type():
    """Get the RingAttn implementation type."""
    return PROCESS_GROUP_MANAGER.ring_type


def get_data_parallel_size():
    """Get the size of the data parallel group."""
    return PROCESS_GROUP_MANAGER.dp_degree


def get_data_parallel_rank():
    """Get the rank of this process in the data parallel group the caller rank belongs to."""
    return PROCESS_GROUP_MANAGER.dp_rank
