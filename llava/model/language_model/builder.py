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

import math
import os.path as osp
import warnings
from typing import Tuple

import torch
from huggingface_hub import file_exists, repo_exists
from huggingface_hub.utils import HFValidationError
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


def has_tokenizer(repo_id_or_path: str) -> bool:
    # Check if the tokenizer is in a local directory
    if osp.exists(osp.join(repo_id_or_path, "tokenizer_config.json")):
        return True

    # Check if the tokenizer is in a Hugging Face Hub repo
    try:
        return repo_exists(repo_id_or_path) and file_exists(repo_id_or_path, "tokenizer_config.json")
    except HFValidationError:
        return False


def context_length_extension(config):
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    model_max_length = getattr(config, "model_max_length", None)
    if orig_ctx_len and model_max_length > orig_ctx_len:
        print(f"Scaling RoPE from {orig_ctx_len} to {model_max_length}")
        scaling_factor = float(math.ceil(model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    return config


def build_llm_and_tokenizer(
    model_name_or_path: str,
    config: PretrainedConfig,
    attn_implementation=None,
    model_max_length=None,
    *args,
    **kwargs,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    llm_cfg = AutoConfig.from_pretrained(model_name_or_path)
    llm_cfg._attn_implementation = attn_implementation
    llm_cfg.model_max_length = model_max_length
    if model_max_length is not None:
        context_length_extension(llm_cfg)

    llm = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, config=llm_cfg, torch_dtype=eval(config.model_dtype), *args, **kwargs
    )

    # Locate the tokenizer.
    llm_path = model_name_or_path
    if not has_tokenizer(llm_path):
        llm_path = osp.join(llm_path, "llm")
    if not has_tokenizer(llm_path):
        raise ValueError(f"Cannot find tokenizer in {llm_path}.")

    # TODO(ligeng): use LLM class to judge to better compability.
    try:
        llm_arch = getattr(llm_cfg, "architectures")[0].lower()
    except BaseException:
        warnings.warn(f'Cannot find LLM architecture, please check the "config.json" under "{llm_path}".')

    if "mpt" in llm_arch:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            model_max_length=llm_cfg.model_max_length,
            padding_side="right",
        )
    elif "yi" in llm_path or (
        getattr(llm_cfg, "num_hidden_layers", -1) == 60 and getattr(llm_cfg, "num_attention_heads", -1) == 56
    ):
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            model_max_length=llm_cfg.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            model_max_length=llm_cfg.model_max_length,
            padding_side="right",
            use_fast=False,
            legacy=False,
        )

    # TODO(ligeng): is this necessary for llava?
    config.hidden_size = llm.config.hidden_size
    return llm, tokenizer
