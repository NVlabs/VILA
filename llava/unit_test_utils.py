# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from unittest.case import _id as __id
from unittest.case import skip as __skip


def requires_gpu(reason=None):
    import torch

    reason = "no GPUs detected. Only test in GPU environemnts" if reason is None else reason
    if not torch.cuda.is_available():
        return __skip(reason)
    return __id


def requires_lustre(reason=None):
    import os.path as osp

    if not (osp.isdir("/lustre") or osp.isdir("/mnt")):
        reason = "lustre path is not avaliable." if reason is None else reason
        return __skip(reason)
    return __id
