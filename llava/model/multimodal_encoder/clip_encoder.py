# This file is modified from https://github.com/haotian-liu/LLaVA/
import torch

from llava.model.multimodal_encoder.vision_encoder import VisionTower, VisionTowerS2
from transformers import (
    PretrainedConfig,
    CLIPVisionModel,
    CLIPImageProcessor,
)


class CLIPVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__(model_name_or_path, config)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name_or_path)
        self.vision_tower = CLIPVisionModel.from_pretrained(
            model_name_or_path, torch_dtype=eval(config.model_dtype)
        )
        self.is_loaded = True


class CLIPVisionTowerS2(VisionTowerS2):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__(model_name_or_path, config)
        self.image_processor = CLIPImageProcessor.from_pretrained(model_name_or_path)
        self.vision_tower = CLIPVisionModel.from_pretrained(
            model_name_or_path, torch_dtype=eval(config.model_dtype)
        )

        # Make sure it crops/resizes the image to the largest scale in self.scales to maintain high-res information
        self.image_processor.size['shortest_edge'] = self.scales[-1]
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.scales[-1]

        self.is_loaded = True