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
# This file is adopted from https://github.com/EvolvingLMMs-Lab/LongVA


import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model


def main(args):
    model_path = args.model
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model, model_name, None)
    model.config.image_aspect_ratio = "pad"
    model.config.mm_patch_merge_type = "flat"
    dataset = load_dataset(args.needle_dataset)["test"]
    for index, instance in enumerate(dataset):
        image = instance["image"].convert("RGB")
        image = process_images([image], image_processor, model.config).half()
        image_features = model.encode_images(image)
        if args.pooling_size != 0:
            B, _, F = image_features.shape
            image_features_spatial = image_features.view(B, int(math.sqrt(_)), int(math.sqrt(_)), F).permute(
                0, 3, 1, 2
            )  # B, F, 24, 24
            image_features_spatial_pool = torch.nn.functional.avg_pool2d(
                image_features_spatial, args.pooling_size, args.pooling_size
            )  # B, F, 12, 12
            image_features = image_features_spatial_pool.flatten(2).transpose(1, 2).contiguous()  # B, 144, F
        image_features = image_features.squeeze(0)
        torch.save(image_features, f"{args.output_dir}/{index}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="output/LLaVA-NeXT-Video-7B-Vicuna")
    parser.add_argument("--needle_dataset", type=str, default="lmms-lab/v_niah_needles")
    parser.add_argument("--output_dir", type=str, default="video_needle_haystack/data/needle_vicuna_embeddings")
    parser.add_argument("--pooling_size", type=int, default=0)
    args = parser.parse_args()
    main(args)
