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

import copy
import os
import pathlib
import re
import warnings
from dataclasses import dataclass

import torch
import torch.distributed as dist
from accelerate.hooks import add_hook_to_module
from transformers import PretrainedConfig, PreTrainedModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from llava.train.sequence_parallel.globals import get_pg_manager, get_ulysses_sp_pg


def rprint(*args, **kwargs):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and dist.is_initialized():
        return print(f"[dist-{rank}-of-{world_size}]", *args, **kwargs)
    else:
        return print(*args, **kwargs)


def mprint(*args, **kwargs):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1 and dist.is_initialized():
        if rank == 0:
            return print(f"[dist-{rank}-of-{world_size}]", *args, **kwargs)
        else:
            return
    else:
        return print(*args, **kwargs)


def is_local(model_name_or_path: str) -> bool:
    return os.path.isdir(model_name_or_path)


def get_checkpoint_path(output_dir: str, checkpoint_prefix: str = "checkpoint") -> str | None:
    output_dir = os.path.abspath(output_dir)
    pathlib_dir = pathlib.Path(output_dir)

    if list(pathlib_dir.glob("config.json")):
        # training has been finished
        return output_dir, False
    else:
        try:
            ordering_and_checkpoint_path = []
            glob_checkpoints = [
                str(x) for x in pathlib.Path(output_dir).glob(f"{checkpoint_prefix}-*") if os.path.isdir(x)
            ]
            for path in glob_checkpoints:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))
            checkpoints_sorted = sorted(ordering_and_checkpoint_path)
            return checkpoints_sorted[-1][1], True
        except:
            return None, True


def prepare_config_for_training(
    config: PretrainedConfig, model_args: dataclass, training_args: dataclass, data_args: dataclass
) -> None:
    assert model_args.vision_tower is not None, "requires vision tower"
    # set module configurations
    if getattr(config, "llm_cfg", None) is None:
        config.llm_cfg = model_args.model_name_or_path
    if getattr(config, "vision_tower_cfg", None) is None:
        config.vision_tower_cfg = model_args.vision_tower
    if getattr(config, "mm_projector_cfg", None) is None:
        config.mm_projector_cfg = model_args.mm_projector
    # set default dtype
    config.model_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    config.model_dtype = config.model_dtype.__str__()
    # set tuning modules
    config.tune_language_model = training_args.tune_language_model
    config.tune_vision_tower = training_args.tune_vision_tower
    config.tune_mm_projector = training_args.tune_mm_projector
    # set data args
    # Get the image_aspect_ratio from the config if is defined there
    # (case of resuming from a checkpoint) or from the data_args
    # (i.e. from the command line when starting a new training).
    if getattr(data_args, "image_aspect_ratio", None) is not None:
        if getattr(config, "image_aspect_ratio", None) is None:
            config.image_aspect_ratio = data_args.image_aspect_ratio
    elif getattr(config, "image_aspect_ratio", None) is not None:
        data_args.image_aspect_ratio = config.image_aspect_ratio
    else:
        raise ValueError("image_aspect_ratio must be set either in data_args or in the pretrained config")

    if (
        hasattr(training_args, "deepspeed")
        and training_args.deepspeed is not None
        and "mics" in training_args.deepspeed
    ):
        config.deepspeed = training_args.deepspeed

    # extra vision tower configuration
    if getattr(config, "vision_tower_cfg", None) is not None:
        # Set the vision config as per the command-line flags, except
        # if the vision config is already defined in the config file (case
        # of resuming from a checkpoint).
        if getattr(config, "mm_vision_select_layer", None) is None:
            config.mm_vision_select_layer = model_args.mm_vision_select_layer
        if getattr(config, "mm_vision_select_feature", None) is None:
            config.mm_vision_select_feature = model_args.mm_vision_select_feature
        # vision tower configurations
        config.vision_resolution = model_args.vision_resolution
        config.interpolate_mode = model_args.interpolate_mode
        config.drop_path_rate = model_args.drop_path_rate
        config.s2 = model_args.s2
        config.s2_scales = model_args.s2_scales
        config.s2_max_split_size = model_args.s2_max_split_size


def vision_resolution_elevation(model: PreTrainedModel, config: PretrainedConfig):
    vision_tower = model.get_vision_tower()
    if vision_tower is not None and "radio" not in vision_tower.__class__.__name__.lower():
        vision_tower._maybe_resize_pos_embeds(
            model=vision_tower.vision_tower,
            image_processor=vision_tower.image_processor,
            resolution=getattr(config, "vision_resolution", -1),
            interpolate_mode=getattr(config, "interpolate_mode", "linear"),
        )


def unit_test_rope_scaling(model: PreTrainedModel, config: PretrainedConfig, training_args: dataclass):
    return False


def calculate_loss_weight(labels, ignore_index=-100):
    # (Qinghao): Weighted loss based on num_active_elements
    # To achieve accurate sequence parallel loss calculation, we need to get
    # the real active_elements of each sequence partitions.
    # For data parallelism, the loss almost remains the same (also more accurate).
    shift_labels = labels[..., 1:].contiguous()
    shift_labels = shift_labels.view(-1)

    padding_mask = shift_labels.eq(ignore_index)  # IGNORE_INDEX = -100 by default
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()
    global_active_sum = copy.deepcopy(num_active_elements)
    dist.all_reduce(global_active_sum)
    loss_weight = num_active_elements / global_active_sum * dist.get_world_size()
    return loss_weight


def reshard_hiddne_states_and_labels(hidden_states, labels):
    PROCESS_GROUP_MANAGER = get_pg_manager()
    sp_degree = PROCESS_GROUP_MANAGER.sp_degree
    sp_rank = PROCESS_GROUP_MANAGER.sp_rank
    sp_group = PROCESS_GROUP_MANAGER.ulysses_pg
    from llava.constants import IGNORE_INDEX

    # Get the seq len on different sp ranks
    bs, shard_seqlen = labels.shape
    ulysses_seq_len = [torch.zeros(1, dtype=torch.int64, device=labels.device) for _ in range(sp_degree)]
    dist.barrier(group=sp_group)
    dist.all_gather(ulysses_seq_len, torch.tensor(shard_seqlen, device=labels.device), group=sp_group)
    dist.barrier(group=sp_group)
    global_seq_len = torch.cat(ulysses_seq_len, dim=0)
    # Gather all labels and flaten them
    all_labels = [
        torch.zeros(bs, seq_len, dtype=labels.dtype, device=labels.device).contiguous() for seq_len in ulysses_seq_len
    ]
    dist.all_gather(all_labels, labels.contiguous(), group=sp_group)
    # flatten_global_labels = torch.cat(all_labels, dim=1)[:, 1:].view(-1)
    flatten_global_labels = torch.cat(all_labels, dim=1)[:, 1:].contiguous().view(-1)
    # Get the label!=IGNORE_INDEX's index
    flatten_label_mask = flatten_global_labels.ne(IGNORE_INDEX)
    flatten_effective_label_index = flatten_label_mask.nonzero(as_tuple=True)
    # padding the effective_label_index if the length is smaller than sp_degree
    if flatten_effective_label_index[0].shape[0] < sp_degree:
        warnings.warn(
            f"The effective label length {flatten_effective_label_index[0].shape[0]} is smaller than sp_degree {sp_degree}, padding the index"
        )
        repeat_num = sp_degree // flatten_effective_label_index[0].shape[0] + 1
    else:
        repeat_num = 1
    # Reconstruct the labels by selecting from the global labels
    effective_global_labels = flatten_global_labels[flatten_effective_label_index]
    if repeat_num > 1:
        effective_global_labels = effective_global_labels.repeat(repeat_num)
    # Global effective seqence length
    global_effective_seq_len = effective_global_labels.shape[0]
    reshard_size = global_effective_seq_len // sp_degree
    # Hyper parameters to reshard the hidden states and labels
    if sp_rank == 0:
        original_start_id = 0
        original_end_id = torch.sum(global_seq_len[: sp_rank + 1]).item()
        start_id = 0
        end_id = reshard_size * (sp_rank + 1)
    elif sp_rank == sp_degree - 1:
        original_start_id = torch.sum(global_seq_len[:sp_rank]).item()
        original_end_id = torch.sum(global_seq_len[: sp_rank + 1]).item()
        start_id = reshard_size * sp_rank
        end_id = global_effective_seq_len
    else:
        original_start_id = torch.sum(global_seq_len[:sp_rank]).item()
        original_end_id = torch.sum(global_seq_len[: sp_rank + 1]).item()
        start_id = reshard_size * sp_rank
        end_id = reshard_size * (sp_rank + 1)
    # Get the local labels
    effective_local_labels = torch.narrow(effective_global_labels, 0, start_id, end_id - start_id)
    # Gather all hidden states and flaten them
    # all_hidden_states = [torch.zeros(bs, seq_len, hidden_states.shape[-1], dtype=hidden_states.dtype, device=hidden_states.device, requires_grad=True).contiguous() for seq_len in ulysses_seq_len]
    all_hidden_states = torch.zeros(
        bs, torch.sum(global_seq_len), hidden_states.shape[-1], dtype=hidden_states.dtype, device=hidden_states.device
    ).contiguous()
    all_hidden_states[:, original_start_id:original_end_id, :] += hidden_states
    dist.barrier(group=sp_group)
    dist.all_reduce(all_hidden_states, group=sp_group)
    dist.barrier(group=sp_group)
    flatten_global_hidden_states = all_hidden_states[:, :-1, :].contiguous().view(-1, hidden_states.shape[-1])
    # Get the local hidden states
    effective_flatten_global_hidden_states = flatten_global_hidden_states[flatten_effective_label_index]
    if repeat_num > 1:
        effective_flatten_global_hidden_states = effective_flatten_global_hidden_states.repeat(repeat_num, 1)
    effective_local_hidden_states = torch.narrow(effective_flatten_global_hidden_states, 0, start_id, end_id - start_id)

    return effective_local_hidden_states, effective_local_labels


def sp_loss_rescale(shift_labels, loss):
    from llava.constants import IGNORE_INDEX

    PROCESS_GROUP_MANAGER = get_pg_manager()
    labels_mask = shift_labels.ne(IGNORE_INDEX)  # IGNORE_INDEX = -100 by default
    num_active_elements = torch.sum(labels_mask)
    global_active_sum = copy.deepcopy(num_active_elements)
    # dist.barrier(group=get_ulysses_sp_pg())
    dist.all_reduce(global_active_sum, group=get_ulysses_sp_pg())
    # print(loss.shape, num_active_elements.shape, global_active_sum.shape)
    loss = loss * num_active_elements / global_active_sum
    dist.all_reduce(loss, group=get_ulysses_sp_pg())
    return loss

    # # if sp_rank == 0:
    # #     start_id = 0
    # # else:
    # #     start_id = torch.sum(ulysses_seq_len[:sp_rank]).item()
    # # end_id = torch.sum(ulysses_seq_len[:sp_rank+1]).item()
    # local_labels = copy.deepcopy(labels[:, 1:])
    # local_labels_mask = local_labels.ne(IGNORE_INDEX)
    # # Get the label!=IGNORE_INDEX's index
    # if local_labels_mask.sum() == 0:
    #     # If all the labels are IGNORE_INDEX, we can just return the first label
    #     effective_local_lable_index = tuple(torch.tensor([0], device=labels.device),)
    #     warnings.warn("All the labels are IGNORE_INDEX, return the first label")
    # else:
    #     effective_local_lable_index = local_labels_mask.nonzero(as_tuple=True)
    # effective_local_labels = local_labels[effective_local_lable_index]

    #


# def sp_loss_reduce(loss):
#     # (Qinghao): Weighted loss based on num_active_elements
#     # To achieve accurate sequence parallel loss calculation, we need to get
#     # the real active_elements of each sequence partitions.
#     # For data parallelism, the loss almost remains the same (also more accurate).
#     # PROCESS_GROUP_MANAGER = get_pg_manager()
#     # if PROCESS_GROUP_MANAGER is None:
#     #     return 1.0


#     # padding_mask = shift_labels.eq(ignore_index)  # IGNORE_INDEX = -100 by default
#     # num_active_elements = padding_mask.numel() - padding_mask.long().sum()
#     # global_active_sum = copy.deepcopy(num_active_elements)
#     # # dist.barrier(group=get_ulysses_sp_pg())
#     # dist.all_reduce(global_active_sum, group=get_ulysses_sp_pg())
#     # loss_weight = num_active_elements / global_active_sum * PROCESS_GROUP_MANAGER.sp_degree
#     # return loss_weight
