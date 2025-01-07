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
from typing import Optional

import transformers


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: Optional[str] = "resize"
    min_tiles: Optional[int] = 1
    max_tiles: Optional[int] = 12
    data_mixture: str = "llava_1_5_mm_align"
    eval_data_mixture: str = None
    vflan_no_system_prompt: bool = False
    downsample_video: bool = False

    # for video training
    num_video_frames: int = 8
    fps: float = 0.0  # 0.0 means we do not use fps at all. Always sample the same number of frames.


@dataclass
class ModelArguments:
    version: Optional[str] = field(default="auto")
    chat_template: Optional[str] = field(default=None)
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    vision_tower: Optional[str] = field(default="google/siglip-so400m-patch14-384")
    mm_projector: Optional[str] = field(default="mlp2x_gelu")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(default=-1)  # default to the last layer
    mm_vision_select_feature: Optional[str] = field(default="patch")
    vision_resolution: Optional[int] = field(default=-1)
    interpolate_mode: Optional[str] = field(default="linear")
    drop_path_rate: Optional[float] = field(default=0.0)
    mlp_path: Optional[str] = field(default=None)
    s2: bool = field(default=False)
    dynamic_s2: bool = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")
    s2_max_split_size: int = field(default=336)
    num_time_tokens: int = field(default=0)
    time_token_format: str = field(default="<t{t}>")
    soft_ce_std: float = field(default=1.0)

    image_encoder: str = field(default='{"_target_": "llava.model.encoders.BasicImageEncoder"}')
    video_encoder: str = field(default='{"_target_": "llava.model.encoders.BasicVideoEncoder"}')
    s2_resize_output_to_scale_idx: int = field(default=0)

    # Quantization and low precision training
    quantize_model: Optional[str] = field(default="false")
    symm: Optional[bool] = field(default=True)

    epsilon: Optional[float] = field(default=1e-10)
    fabit: Optional[str] = field(default="E4M3")
    fwbit: Optional[str] = field(default="E4M3")
    bobit: Optional[str] = field(default="E5M2")
    row_blocksize: Optional[int] = -1  # -1 means only 1 quantization group along row axis
    col_blocksize: Optional[int] = -1  # -1 means only 1 quantization group along column axis
    qchoice: Optional[list[str]] = field(
        default_factory=lambda: [
            "none",
            "all",
            "linear",
            "mlp",
            "attn",
            "gelu",
            "layernorm",
            "backbone",
            "residual",
            "backbone",
        ],
    )

    pad_to_multiple_of: int = 0  # if sequence length * batch size can not be divided by 128, the triton implementation of fp8 matmul when calculating weight gradient will become highly inefficient. Therefore, I want to pad the sequence length to a multiple of some exponent of 2. This will be used in prepare_inputs_labels_for_multimodal()

    # Memory Efficient FP8 related
    Ubit: str = field(default="100")
    quantize_model: str = field(default="false", metadata={"help": "Enable model quantization"})
    symm: bool = field(default=True, metadata={"help": "Use symmetric quantization"})
    epsilon: float = field(default=1e-10, metadata={"help": "Small epsilon for numerical stability"})
    fabit: str = field(default="E4M3", metadata={"help": "Bit format for forward activation"})
    fwbit: str = field(default="E4M3", metadata={"help": "Bit format for forward weights"})
    fobit: str = field(default="E4M3", metadata={"help": "Bit format for forward output"})
    babit: str = field(default="E5M2", metadata={"help": "Bit format for backward activation"})
    bwbit: str = field(default="E5M2", metadata={"help": "Bit format for backward weights"})
    bobit: str = field(default="E5M2", metadata={"help": "Bit format for backward output"})
    qchoice: str = field(default="none", metadata={"help": "Quantization choice"})
    group_size: int = field(default=-1, metadata={"help": "Group size for quantization"})
    weight_memory_efficient: bool = field(default=True, metadata={"help": "Enable memory-efficient weights"})

    min_blockunit_row: int = field(default=4)
    min_blockunit_col: int = field(default=4)
    refine_residual_fp: bool = field(default=False)
    refine_ln_pertoken: bool = field(default=False)
    refine_ln_blocksize: bool = field(default=False)
    refine_ln_blocksize_but_only_forward: bool = field(default=False)
    refine_ln_blocksize_but_only_backward: bool = field(default=False)
    refine_attn_blocksize: bool = field(default=False)
    refine_mlp_blocksize: bool = field(default=False)
    refine_row_blocksize: int = field(default=4)
    refine_col_blocksize: int = field(default=4)
    draw_distribution_forward: bool = field(default=False)
    draw_distribution_backward: bool = field(default=False)

    # Quantize Optimizer Related
    use_quantize_optimizer: bool = field(default=False)
    row_blocksize_optimizer: int = field(default=1)
    col_blocksize_optimizer: int = field(default=128)
    pad_block: bool = field(default=False)
    first_order_bit: Optional[str] = field(default=None)
    first_order_quant_type: Optional[str] = field(default=None)
    second_order_bit: Optional[str] = field(default=None)
    second_order_quant_type: Optional[str] = field(default=None)
    epsilon_optimizer: float = field(default=1e-15)

    # Quantization and low precision training
    quantize_model: Optional[str] = field(default="false")
    symm: Optional[bool] = field(default=True)

    epsilon: Optional[float] = field(default=1e-10)
    fabit: Optional[str] = field(default="E4M3")
    fwbit: Optional[str] = field(default="E4M3")
    bobit: Optional[str] = field(default="E5M2")
    row_blocksize: Optional[int] = -1  # -1 means only 1 quantization group along row axis
    col_blocksize: Optional[int] = -1  # -1 means only 1 quantization group along column axis
    qchoice: Optional[list[str]] = field(
        default_factory=lambda: [
            "none",
            "all",
            "linear",
            "mlp",
            "attn",
            "gelu",
            "layernorm",
            "backbone",
            "residual",
            "backbone",
        ],
    )

    pad_to_multiple_of: int = 0  # if sequence length * batch size can not be divided by 128, the triton implementation of fp8 matmul when calculating weight gradient will become highly inefficient. Therefore, I want to pad the sequence length to a multiple of some exponent of 2. This will be used in prepare_inputs_labels_for_multimodal()

    # Memory Efficient FP8 related
    Ubit: str = field(default="100")
    quantize_model: str = field(default="false", metadata={"help": "Enable model quantization"})
    symm: bool = field(default=True, metadata={"help": "Use symmetric quantization"})
    epsilon: float = field(default=1e-10, metadata={"help": "Small epsilon for numerical stability"})
    fabit: str = field(default="E4M3", metadata={"help": "Bit format for forward activation"})
    fwbit: str = field(default="E4M3", metadata={"help": "Bit format for forward weights"})
    fobit: str = field(default="E4M3", metadata={"help": "Bit format for forward output"})
    babit: str = field(default="E5M2", metadata={"help": "Bit format for backward activation"})
    bwbit: str = field(default="E5M2", metadata={"help": "Bit format for backward weights"})
    bobit: str = field(default="E5M2", metadata={"help": "Bit format for backward output"})
    qchoice: str = field(default="none", metadata={"help": "Quantization choice"})
    group_size: int = field(default=-1, metadata={"help": "Group size for quantization"})
    weight_memory_efficient: bool = field(default=True, metadata={"help": "Enable memory-efficient weights"})

    min_blockunit_row: int = field(default=4)
    min_blockunit_col: int = field(default=4)
    refine_residual_fp: bool = field(default=False)
    refine_ln_pertoken: bool = field(default=False)
    refine_ln_blocksize: bool = field(default=False)
    refine_ln_blocksize_but_only_forward: bool = field(default=False)
    refine_ln_blocksize_but_only_backward: bool = field(default=False)
    refine_attn_blocksize: bool = field(default=False)
    refine_mlp_blocksize: bool = field(default=False)
    refine_row_blocksize: int = field(default=4)
    refine_col_blocksize: int = field(default=4)
    draw_distribution_forward: bool = field(default=False)
    draw_distribution_backward: bool = field(default=False)

    # Quantize Optimizer Related
    use_quantize_optimizer: bool = field(default=False)
    row_blocksize_optimizer: int = field(default=1)
    col_blocksize_optimizer: int = field(default=128)
    pad_block: bool = field(default=False)
    first_order_bit: Optional[str] = field(default=None)
    first_order_quant_type: Optional[str] = field(default=None)
    second_order_bit: Optional[str] = field(default=None)
    second_order_quant_type: Optional[str] = field(default=None)
    epsilon_optimizer: float = field(default=1e-15)


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
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."},
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."},
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    # lora-related
    lora_enable: bool = False
    use_dora: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    lora_llm: bool = False
    lora_vt: bool = False
    dpo: bool = False
    longvila_sampler: bool = False
    dpo_beta: float = field(default=0.1)
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    total_time_limit: int = field(default=-1, metadata={"help": "Timeout limit for this job (in minutes)."})
    pre_terminate_time: int = field(
        default=10,
        metadata={"help": "Time to terminate the task inadvance (minutes), saveing checkpoints needs time."},
    )
    seq_parallel_size: int = field(
        default=-1,
        metadata={"help": "The degree of sequence parallelism (SP). SP is disabled by default (value: -1). "},
    )
    seq_parallel_ring_size: int = field(
        default=-1,
        metadata={
            "help": "The communication process group size using optimized Ring Attention approach in SP, where `seq_parallel_size` = `seq_parallel_ring_size` x `seq_parallel_ulysses_size` (determined by other two terms). Ring Attention approach is disabled by default in SP. This setting is adjustable only when `seq_parallel_size` > 1."
        },
    )
    seq_parallel_ring_type: str = field(
        default="ring_varlen",
        metadata={
            "help": "Ring Attention implementation. Support ['ring_varlen', 'zigzag_ring_varlen'] in 2D attention. Only works when `seq_parallel_ring_size` > 1."
        },
    )
    debug_e2e: bool = field(
        default=False,
        metadata={"help": "Whether enter debug mode."},
    )


