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

dataset_path = "./idefics2-sft/the_cauldron"
save_path = "./idefics2-sft/new-vflan/"
metadata_path = os.path.join(save_path, "metadata")
dataset_names = sorted(os.listdir(metadata_path))


def load_jsonl(file_path):
    data = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


all_data = []
for dataset_name in dataset_names:
    if "websight" in dataset_name or "datikz" in dataset_name:
        # skip the snapshot => code datasets for now.
        continue
    loaded = load_jsonl(os.path.join(metadata_path, dataset_name))
    id_offset = len(all_data)
    for item in loaded:
        item["id"] += id_offset
    all_data += loaded
    print(dataset_name, len(all_data), all_data[-1])

with open(os.path.join(save_path, "idefics2_sft_train.jsonl"), "w") as f:
    for item in all_data:
        json.dump(item, f)
        f.write("\n")


