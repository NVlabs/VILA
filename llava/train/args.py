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

from dataclasses import dataclass, field
import transformers
from typing import Dict, Optional, Sequence, List


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = "square"
    data_mixture: str = "llava_1_5_mm_align"
    eval_data_mixture: str = None
    vflan_no_system_prompt: bool = False
    downsample_video: bool = False

    # for video training
    num_video_frames: int = 8


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    mm_projector: Optional[str] = field(default="mlp2x_gelu")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    mm_vision_select_feature: Optional[str] = field(default="patch")
    vision_resolution: Optional[int] = field(default=-1)
    interpolate_mode: Optional[str] = field(default="linear")
    drop_path_rate: Optional[float] = field(default=0.)
    s2: bool = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")
    s2_max_split_size: int = field(default=336)



@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    tune_vision_tower: bool = field(default=False)
    tune_language_model: bool = field(default=False)
    tune_mm_projector: bool = field(default=False)
    model_dtype: str = field(default="torch.bfloat16")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    total_time_limit: int = field(
        default=-1, metadata={"help": "Timeout limit for this job (in minutes)."}
    )
    pre_terminate_time: int = field(
        default=10,
        metadata={
            "help": "Time to terminate the task inadvance (minutes), saveing checkpoints needs time."
        },
    )
