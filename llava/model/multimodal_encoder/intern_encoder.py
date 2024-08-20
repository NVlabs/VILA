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
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoConfig, AutoModel
from transformers.image_processing_utils import BaseImageProcessor

from llava.model.multimodal_encoder.intern.configuration_intern_vit import InternVisionConfig
from llava.model.multimodal_encoder.intern.modeling_intern_vit import InternVisionModel
from llava.model.multimodal_encoder.vision_encoder import VisionTower


def build_transform(input_size):
    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return transform


class InternVisionPreprocessor(BaseImageProcessor):
    @property
    def size(self):
        return {"height": 448, "width": 448}

    def preprocess(self, image, return_tensors):
        transform = build_transform(448)
        if isinstance(image, list):
            image_tensor = [transform(img) for img in image]
            return {"pixel_values": image_tensor}
        else:
            image_tensor = transform(image)
            return {"pixel_values": [image_tensor]}


class InternVisionTower(VisionTower):
    def __init__(self, vision_tower, config, drop_path_rate=0.0):
        super().__init__(vision_tower, config)
        self._drop_path_rate = drop_path_rate

        self.image_processor = InternVisionPreprocessor()
        vision_config = InternVisionConfig.from_pretrained(vision_tower)
        vision_config.drop_path_rate = self._drop_path_rate
        self.vision_tower = InternVisionModel.from_pretrained(
            vision_tower, torch_dtype=eval(config.model_dtype), config=vision_config
        )

        self.is_loaded = True


AutoConfig.register("intern_vit_6b", InternVisionConfig)
AutoModel.register(InternVisionConfig, InternVisionModel)
