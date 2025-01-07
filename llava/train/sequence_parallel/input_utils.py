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

import torch


def extract_local_zigzag(value, rank, world_size, device, dim=1):
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat([value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim)
    return local_value.to(device)


def extract_local_from_list(value_list, sp_rank, sp_size):
    quotient, remainder = divmod(len(value_list), sp_size)
    start_idx = sp_rank * quotient + min(sp_rank, remainder)
    end_idx = (sp_rank + 1) * quotient + min(sp_rank + 1, remainder)
    return value_list[start_idx:end_idx]


def extract_local_from_list_zigzag(value_list, sp_rank, sp_size):
    chunk_size, remainder = divmod(len(value_list), (2 * sp_size))
    value_chunks = []
    start_idx = 0
    for i in range(2 * sp_size):
        extra = 1 if i < remainder else 0
        end_idx = start_idx + chunk_size + extra
        value_chunks.append(value_list[start_idx:end_idx])
        start_idx = end_idx

    local_value = value_chunks[sp_rank] + value_chunks[2 * sp_size - sp_rank - 1]
    return local_value


def extract_local_input_ids(input_ids, image_positions, sp_rank, sp_size, bos_token_id=1, image_token_len=3):
    quotient, remainder = divmod(len(image_positions), sp_size)
    start_idx = sp_rank * quotient + min(sp_rank, remainder)
    end_idx = (sp_rank + 1) * quotient + min(sp_rank + 1, remainder)

    start_position_idx = image_positions[start_idx]
    if sp_rank != sp_size - 1:
        end_position_idx = image_positions[end_idx]
    else:
        end_position_idx = len(input_ids)

    if sp_rank == 0:  # Handle the head of the sequence
        return input_ids[0:end_position_idx]
    elif sp_rank == sp_size - 1:  # Handle the tail of the sequence
        return input_ids[start_position_idx:]
    else:
        return input_ids[start_position_idx:end_position_idx]


def extract_local_position_ids(input_ids, image_positions, image_ids, sp_rank, sp_size, image_token_len=198):
    quotient, remainder = divmod(len(image_ids), sp_size)
    start_idx = sp_rank * quotient + min(sp_rank, remainder)
    end_idx = (sp_rank + 1) * quotient + min(sp_rank + 1, remainder)
    start_position_idx = image_positions[start_idx] + image_ids[start_idx] * image_token_len
    if sp_rank != sp_size - 1:  # Handle the tail of the sequence
        end_position_idx = image_positions[end_idx] + image_ids[end_idx] * image_token_len  # image_token_len + 3
    else:
        end_position_idx = len(input_ids)
    if sp_rank == 0:  # Handle the head of the sequence
        return input_ids[0:end_position_idx]
    elif sp_rank == sp_size - 1:  # Handle the tail of the sequence
        return input_ids[start_position_idx:]
    else:
        return input_ids[start_position_idx:end_position_idx]


