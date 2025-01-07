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

import matplotlib.pyplot as plt
import numpy as np
import torch


def list_has_common_element(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    return len(set1.intersection(set2)) > 0


def calculate_scale_num(input, row_block, col_block):
    if len(input.shape) > 2:
        input = input.reshape(-1, input.shape[2])
    elif len(input.shape) == 2:
        pass
    else:
        raise ValueError(f"input shape {input.shape} does not match for block cut, {input}")
    M, N = input.shape[0], input.shape[1]

    if row_block == -1:
        row_block = M
    if col_block == -1:
        col_block = N

    return input.numel() / (row_block * col_block)


def quant_get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK") or 0)


def format_string_with_condition(
    input_string,
    condition_config,
    symm,
    bits,
    blocksize_config,
    input_pad=20,
):
    padded_string = input_string.ljust(input_pad)
    output_string = padded_string

    for k, v in condition_config.items():
        if v:
            output_string = output_string + k.ljust(10) + "True".ljust(6) + "".ljust(6)
        else:
            output_string = output_string + k.ljust(10) + "".ljust(6) + "False".ljust(6)

    output_string = output_string + f"Symm {symm}".ljust(10)

    for k, v in bits.items():
        output_string = output_string + f"{k} bit".ljust(10) + v.ljust(10)
    for k, v in blocksize_config.items():
        output_string += f"{k}: {v}".ljust(15)

    return output_string


def print_warning(sentence):
    print("*" * (len(sentence) + 4))
    print(f"* {sentence} *")
    print("*" * (len(sentence) + 4))


def check_nan_inf(tensor, check_nan, check_inf):
    if check_nan:
        contain_nan = torch.isnan(tensor).any()
    else:
        contain_nan = False
    if check_inf:
        contain_inf = torch.isinf(tensor).any()
    else:
        contain_inf = False
    return contain_nan, contain_inf


def move_torch_to_numpy(tensor):
    if tensor is None:
        return None

    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.detach().float().numpy()


def flatten_to_1d(tensor):
    if tensor is None:
        return None

    return tensor.reshape(-1)


