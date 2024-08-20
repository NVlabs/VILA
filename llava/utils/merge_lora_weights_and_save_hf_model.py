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
# This file is modified from https://github.com/dvlab-research/LongLoRA

import argparse
import os
from typing import Dict

import torch
import transformers
from peft import PeftModel


def parse_config():
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument("--base_model", type=str, default="/data/pretrained-models/llama-7b-hf")
    parser.add_argument("--peft_model", type=str, default=None, help="")
    parser.add_argument("--save_path", type=str, default=None, help="")
    parser.add_argument("--cache_dir", type=str, default=None, help="./cache_dir")
    parser.add_argument("--rope_theta", type=int, default=15300000, help="")
    parser.add_argument("--max_position_embeddings", type=int, default=65536, help="")
    args = parser.parse_args()
    return args


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def main():
    args = parse_config()
    device = "cuda:0"
    torch.cuda.set_device(device)

    print("base model", args.base_model)
    print("peft model", args.peft_model)

    config = transformers.AutoConfig.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
    )

    config.rope_theta = args.rope_theta
    config.max_position_embeddings = args.max_position_embeddings
    config.model_max_length = args.max_position_embeddings
    config.tokenizer_model_max_length = args.max_position_embeddings

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.base_model,
        config=config,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.base_model,
    )

    model = PeftModel.from_pretrained(
        model,
        args.peft_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = model.merge_and_unload()
    model.save_pretrained(args.save_path)
    tokenizer.save_pretrained(args.save_path)


if __name__ == "__main__":
    main()
