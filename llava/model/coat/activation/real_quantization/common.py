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
import triton

SCALE_MIN_THRES = 1e-10

FP8_MAX_VALUE = {
    torch.float8_e4m3fn: 448.0,
    torch.float8_e5m2: 57344.0,
}

convert_str_to_fp8 = {"E4M3": torch.float8_e4m3fn, "E5M2": torch.float8_e5m2}
convert_fp8_to_embit = {
    torch.float8_e4m3fn: (4.0, 3.0),
    torch.float8_e5m2: (5.0, 2.0),
}


def get_configs_io_block():
    configs = []
    for nstages in [3, 4, 5]:
        for block_m in [32, 64]:
            for block_n in [64, 128]:
                for nwarps in [4, 8]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n},
                            num_stages=nstages,
                            num_warps=nwarps,
                        )
                    )
    return configs


# from .common import SCALE_MIN_THRES, FP8_MAX_VALUE
#                     SCALE_MIN_THRES: tl.constexpr,
#  + SCALE_MIN_THRES
# SCALE_MIN_THRES=SCALE_MIN_THRES,


