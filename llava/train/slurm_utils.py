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

import datetime
import logging
import logging.handlers
import os
import sys
import time
import warnings

import requests
import torch
import transformers
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, get_last_checkpoint


def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def get_local_rank():
    if not torch.distributed.is_initialized():
        return 0
    num_gpus = torch.cuda.device_count()
    return get_rank() % num_gpus


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = 0

    def start(self):
        self.start_time = time.time()

    def reset(self):
        self.start_time = None
        self.elapsed_time = 0

    def get_elapsed_time(self):
        if self.start_time is not None:
            return self.elapsed_time + (time.time() - self.start_time)


timer = Timer()


def set_timer():
    timer.start()


def rank_print(*s):
    if not torch.distributed.is_initialized():
        rank = 0
    else:
        rank = torch.distributed.get_rank()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] Rank {rank}:", *s)


class TimeoutTerminateCallback(transformers.TrainerCallback):
    def __init__(self, total_time_limit=240, pre_terminate_time=10):
        self.total_time_limit = total_time_limit
        self.pre_terminate_time = pre_terminate_time
        elapsed_time = timer.get_elapsed_time()
        rank_print(
            f"Timer for terminate callback has been set.\nTotal limit: {total_time_limit}min\nPre terminate time: {pre_terminate_time}min elapsed_time: {elapsed_time}s"
        )

        self.time_to_kill = (total_time_limit - pre_terminate_time) * 60

    def on_step_end(self, args, state, control, model, **kwargs):
        elapsed_time = timer.get_elapsed_time()

        if elapsed_time is None:
            # no timer has been set
            return control

        if elapsed_time > self.time_to_kill:
            rank_print("Timeout, start to save checkpoint....")
            control.should_save = True
            control.should_training_stop = True

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if state.global_step < state.max_steps:
            exit(124)


