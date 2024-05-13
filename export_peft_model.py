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
# This file is originated from: https://github.com/haotian-liu/LLaVA/

import os
import sys
import transformers
from llava.model import *
from peft import PeftModel
import torch

model_name_or_path = "lmsys/vicuna-7b-v1.5"
peft_dir = sys.argv[1]
assert os.path.exists(peft_dir)
save_dir = os.path.join(peft_dir, "exported")
os.makedirs(save_dir, exist_ok=True)

config = LlavaConfig.from_pretrained(model_name_or_path)
config._attn_implementation = "flash_attention_2"
torch.set_default_dtype(torch.bfloat16)
model = LlavaLlamaForCausalLM.from_pretrained(
    model_name_or_path,
    config=config,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, peft_dir)
print(model)
model = model.merge_and_unload()
model.config.save_pretrained(save_dir)
model.save_pretrained(save_dir)
