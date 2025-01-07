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

import json
import os
import pickle

import torch
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

# download M3IT to the dataset_path directory
dataset_path = "/dataset/llava-data/instruction-tuning/M3IT"
save_path = "/dataset/llava-data/instruction-tuning/new-vflan"
os.makedirs(save_path, exist_ok=True)

dataset_types = [
    "captioning",
    "captioning",
    "generation",
    "generation",
    "reasoning",
    "reasoning",
    "reasoning",
    "vqa",
    "vqa",
    "vqa",
    "vqa",
    "vqa",
    "vqa",
    "vqa",
]
dataset_names = [
    "image-paragraph-captioning",
    "textcap",
    "multi30k",
    "visual-dialog",
    "clevr",
    "nlvr",
    "visual-mrc",
    "docvqa",
    "gqa",
    "ivqa",
    "ocr-vqa",
    "st-vqa",
    "viquae",
    "vqa-v2",
]


assert len(dataset_types) == len(dataset_names)

for dataset_type, dataset_name in zip(dataset_types, dataset_names):
    print("Processing", dataset_name, "...")
    dataset = list(load_dataset(dataset_path, dataset_name)["train"])
    for item in dataset:
        question = item["instruction"] + item["inputs"]
        answer = item["outputs"]
        image = item["image_base64_str"]
        item.pop("instruction")
        item.pop("inputs")
        item.pop("outputs")
        item.pop("image_base64_str")
        item["question"] = question
        item["answer"] = answer
        item["image"] = image
    print(len(dataset), dataset[-1].keys())
    save_filename = f"{dataset_type}_{dataset_name}_train.pkl"
    save_filename = os.path.join(save_path, save_filename)
    with open(save_filename, "wb") as f:
        pickle.dump(dataset, f)


