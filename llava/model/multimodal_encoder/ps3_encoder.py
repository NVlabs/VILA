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
from transformers import AutoConfig, AutoModel, PretrainedConfig

try:
    from ps3 import PS3Config, PS3ImageProcessor, PS3VisionConfig, PS3VisionModel
except ImportError:
    print("PS3 is not installed. Please install it using the following command:")
    print("pip install ps3")
    raise

from llava.model.multimodal_encoder.vision_encoder import VisionTower


class PS3VisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig) -> None:
        super().__init__(model_name_or_path, config)
        # TODO(ligengl): why pass config here leading to errors?
        print(model_name_or_path)
        self.vision_tower = PS3VisionModel.from_pretrained(
            model_name_or_path,
            # attn_implementation="flash_attention_2",
            torch_dtype=eval(config.model_dtype),
        )
        self.image_processor = PS3ImageProcessor.from_pretrained(model_name_or_path)
        self.is_loaded = True

        self.vision_tower.vision_model.num_hidden_layers_to_return = 2

        if getattr(config, "ps3_grad_checkpointing", False):
            self.vision_tower.vision_model.trunk.set_grad_checkpointing(True)

    @property
    def hidden_size(self):
        return self.vision_tower.width

    @property
    def dtype(self):
        return list(self.vision_tower.parameters())[0].dtype

    @property
    def device(self):
        return list(self.vision_tower.parameters())[0].device


AutoConfig.register("ps3", PS3Config)
AutoConfig.register("ps3_vision_model", PS3VisionConfig)
# AutoModel.register(PS3VisionConfig, PS3VisionModel)
