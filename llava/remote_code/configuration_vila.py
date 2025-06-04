import json
import math
import os
import os.path as osp
from copy import deepcopy
from threading import Thread
from typing import List, Optional

import torch
import torchvision
from PIL import Image
from transformers import (
    AutoProcessor,
    PretrainedConfig,
    PreTrainedModel,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2PreTrainedModel,
    TextIteratorStreamer,
)


class VILAConfig(PretrainedConfig):
    model_type = "vila"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        llm_cfg=None,
        vision_tower_cfg=None,
        mm_projector_cfg=None,
        architectures=None,
        resume_path=None,
        hidden_size=None,
        mm_hidden_size=None,
        image_aspect_ratio=None,
        num_video_frames=None,
        fps=None,
        mm_vision_select_layer=None,
        mm_vision_select_feature=None,
        mm_use_im_start_end=False,
        mm_use_im_patch_token=False,
        mm_projector_lr=None,
        vision_tower_lr=None,
        vision_resolution=None,
        interpolate_mode=None,
        s2=None,
        dynamic_s2=None,
        s2_scales=None,
        s2_max_split_size=None,
        s2_resize_output_to_scale_idx=0,
        min_tiles: Optional[int] = 1,
        max_tiles: Optional[int] = 12,
        num_time_tokens=None,
        time_token_format=None,
        image_encoder: str = '{"_target_": "llava.model.encoders.BasicImageEncoder"}',
        video_encoder: str = '{"_target_": "llava.model.encoders.BasicVideoEncoder"}',
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.architectures = architectures
        self.llm_cfg = llm_cfg
        self.vision_tower_cfg = vision_tower_cfg
        self.mm_projector_cfg = mm_projector_cfg
        self.resume_path = resume_path

        self.hidden_size = hidden_size
        self.mm_hidden_size = mm_hidden_size
        self.image_aspect_ratio = image_aspect_ratio
        self.num_video_frames = num_video_frames
        self.fps = fps
        self.mm_vision_select_layer = mm_vision_select_layer
        self.mm_vision_select_feature = mm_vision_select_feature
        self.mm_use_im_start_end = mm_use_im_start_end
        self.mm_use_im_patch_token = mm_use_im_patch_token
        self.mm_projector_lr = mm_projector_lr
        self.vision_tower_lr = vision_tower_lr
        self.vision_resolution = vision_resolution
        self.interpolate_mode = interpolate_mode
        self.s2 = s2
        self.dynamic_s2 = dynamic_s2
        self.s2_scales = s2_scales
        self.s2_max_split_size = s2_max_split_size
        self.s2_resize_output_to_scale_idx = s2_resize_output_to_scale_idx
        self.min_tiles = min_tiles
        self.max_tiles = max_tiles
        self.num_time_tokens = num_time_tokens
        self.time_token_format = time_token_format

        self.image_encoder = image_encoder
        self.video_encoder = video_encoder
