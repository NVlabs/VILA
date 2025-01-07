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

import pandas
import torch
from tqdm import tqdm

# download OpenORCA/FLAN to the dataset_path directory
dataset_path = "/dataset/llava-data/instruction-tuning/FLAN/"
save_path = "/dataset/llava-data/instruction-tuning/new-vflan"
os.makedirs(save_path, exist_ok=True)
dataset_files = sorted(os.listdir(dataset_path))


filtered_folders = []
for folder in dataset_files:
    if folder.endswith("_data"):
        filtered_folders.append(os.path.join(dataset_path, folder))


all_inputs = []
all_targets = []

# small scale experiment
# filtered_folders = filtered_folders[:3]

for folder in tqdm(filtered_folders):
    parquet_files = sorted(os.listdir(folder))
    for cur_parquet in parquet_files:
        # print(folder, cur_parquet)
        cur_parquet_fn = os.path.join(dataset_path, folder, cur_parquet)
        loaded = pandas.read_parquet(cur_parquet_fn)
        all_inputs.extend(list(loaded["inputs"]))
        all_targets.extend(list(loaded["targets"]))
        print(folder, cur_parquet, len(all_inputs), len(all_targets))

print(min([len(x) for x in all_targets]))

targeted_dataset_size = 1_000_000
filtered_samples = []
selected_indices = torch.linspace(0, len(all_inputs) - 1, targeted_dataset_size).int().cpu().numpy().tolist()
cnt = 0
for index in selected_indices:
    filtered_samples.append(
        dict(
            question=all_inputs[index],
            answer=all_targets[index],
            id="text_flan_%08d" % cnt,
            image=[],
        )
    )
    cnt += 1

with open(os.path.join(save_path, "text_flan_1m.pkl"), "wb") as f:
    pickle.dump(filtered_samples, f)


