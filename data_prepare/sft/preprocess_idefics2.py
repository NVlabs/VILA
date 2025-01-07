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
from multiprocessing import Pool

import torch
from datasets import concatenate_datasets, load_dataset
from PIL import Image
from tqdm import tqdm


def general_conversation_preprocessor(item, dataset_name, id):
    # process the conversation item to llava format.
    conversations = []
    ret_item = dict(id=id)
    # ret_item["images"] = item["images"]
    img_paths = []
    for img_idx, img in enumerate(item["images"]):
        save_path_to_append = os.path.join("images", dataset_name, f"{id}_{img_idx}.png")
        img_path = os.path.join(save_path, save_path_to_append)
        if img.mode == "CMYK":
            img = img.convert("RGB")
        img.save(img_path)
        img_paths.append(save_path_to_append)
    ret_item["images"] = img_paths
    old_conversations = item["texts"]
    for idx, conv in enumerate(old_conversations):
        if "user" in conv:
            if idx > 0:
                cur_conv = conv["user"]
                new_conv = {"from": "human", "value": cur_conv}
            else:
                cur_conv = conv["user"]
                new_conv = {"from": "human", "value": "<image>\n" * len(item["images"]) + cur_conv}
            conversations.append(new_conv)
        if "assistant" in conv:
            cur_conv = conv["assistant"]
            if cur_conv.startswith("Answer: "):
                cur_conv = cur_conv.replace("Answer: ", "")
            new_conv = {"from": "gpt", "value": cur_conv}
            conversations.append(new_conv)
    ret_item["conversations"] = conversations
    return ret_item


def process_dataset(args):
    dataset_name, dataset_path, metadata_path, save_path = args
    if os.path.exists(os.path.join(metadata_path, dataset_name + "_train.jsonl")):
        return
    print("Processing", dataset_name, "...")
    loaded = load_dataset(dataset_path, dataset_name)["train"]
    dataset = list(loaded)
    cnt = 0
    cur_llava_format_dataset = []
    for item in tqdm(dataset):
        new_item = general_conversation_preprocessor(item, dataset_name, cnt)
        if cnt == 0:
            print(item["texts"], item["images"][0], new_item)
            print(new_item)
        cnt += 1
        cur_llava_format_dataset.append(new_item)

    with open(os.path.join(metadata_path, dataset_name + "_train.jsonl"), "w") as f:
        for item in cur_llava_format_dataset:
            json.dump(item, f)
            f.write("\n")


# download M3IT to the dataset_path directory
dataset_path = "/home/jasonlu/workspace/idefics2-sft/the_cauldron"
save_path = "/home/jasonlu/workspace/idefics2-sft/new-vflan/"
metadata_path = os.path.join(save_path, "metadata")
os.makedirs(metadata_path, exist_ok=True)

skipped_datasets = [
    "ai2d",  # internvl-sft
    "chartqa",  # internvl-sft
    "clevr",  # vflan, HAS BUG
    "clevr_math",  # HAS BUG
    "docvqa",  # internvl-sft
    "dvqa",  # internvl-sft
    "nlvr2",  # vflan
    "ocrvqa",  # vflan
    "st_vqa",  # vflan
    "textcaps",  # vflan, llava1.5
    "visualmrc",  # vflan
    "vqav2",  # vflan, llava1.5
    "okvqa",  # llava1.5
    "aokvqa",  # llava1.5
    "plotqa",  # has problem to load (very slow)
    "localized_narratives",  # has problem to load (very slow)
]

_dataset_names = sorted(os.listdir(dataset_path))
dataset_names = []
for name in _dataset_names:
    if name.startswith("."):
        continue
    if name in skipped_datasets:
        continue
    if os.path.isdir(os.path.join(dataset_path, name)):
        dataset_names.append(name)
        os.makedirs(os.path.join(save_path, "images", name), exist_ok=True)
print(dataset_names, len(dataset_names))

# sequential version
# for dataset_name in dataset_names:
#     process_dataset((dataset_name, dataset_path, metadata_path, save_path))
# parallel version
with Pool(processes=min(48, len(dataset_names))) as pool:
    # Prepare the arguments for the process_dataset function
    args = [(dataset_name, dataset_path, metadata_path, save_path) for dataset_name in dataset_names]

    # Map the process_dataset function to the arguments
    for _ in tqdm(pool.imap_unordered(process_dataset, args), total=len(args), desc="Processing datasets"):
        pass


