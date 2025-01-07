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

"""
Usage:
python3 -m llava.model.consolidate --src ~/model_weights/llava-7b --dst ~/model_weights/llava-7b_consolidate
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llava.model import *
from llava.model.utils import auto_upgrade


def consolidate_ckpt(src_path, dst_path):
    print("Loading model")
    auto_upgrade(src_path)
    src_model = AutoModelForCausalLM.from_pretrained(src_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    src_tokenizer = AutoTokenizer.from_pretrained(src_path, use_fast=False)
    src_model.save_pretrained(dst_path)
    src_tokenizer.save_pretrained(dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True)
    parser.add_argument("--dst", type=str, required=True)

    args = parser.parse_args()

    consolidate_ckpt(args.src, args.dst)


