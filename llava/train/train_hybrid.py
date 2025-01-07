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
# This file is modified from https://github.com/haotian-liu/LLaVA/

from unittest import mock

from llava.model.utils.packing import _get_unpad_data
from llava.train.sequence_parallel.monkey_patch import _flash_attention_forward, _update_causal_mask
from llava.train.train import train
from llava.train.transformer_normalize_monkey_patch import patched_normalize


def __len__(self):
    return len(self.batch_sampler)


def __iter__(self):
    return self.batch_sampler.__iter__()


if __name__ == "__main__":
    with (
        mock.patch("transformers.models.llama.modeling_llama._flash_attention_forward", new=_flash_attention_forward),
        mock.patch("transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask", new=_update_causal_mask),
        mock.patch("transformers.models.qwen2.modeling_qwen2._flash_attention_forward", new=_flash_attention_forward),
        mock.patch("transformers.models.qwen2.modeling_qwen2.Qwen2Model._update_causal_mask", new=_update_causal_mask),
        mock.patch("transformers.image_processing_utils.normalize", new=patched_normalize),
        mock.patch("accelerate.data_loader.BatchSamplerShard.__len__", new=__len__),
        mock.patch("accelerate.data_loader.BatchSamplerShard.__iter__", new=__iter__),
    ):
        train()


