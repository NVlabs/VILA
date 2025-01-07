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

from tqdm import tqdm

data_path = "/dataset/mmc4-core/jsonl-filtered"
pkl_path = "/dataset/mmc4-core/pkl"  # where the image stores
output_path = "/dataset/mmc4-core/pkl-core"

shard_names = [f for f in os.listdir(data_path) if f.endswith(".jsonl")]
# now load data
for shard_name in tqdm(shard_names):
    # load shard
    with open(os.path.join(data_path, shard_name)) as json_file:
        json_list = list(json_file)
    data_list = [json.loads(json_str) for json_str in json_list]

    with open(os.path.join(pkl_path, shard_name.replace(".jsonl", ".pkl")), "rb") as f:
        image_dict = pickle.load(f)

    assert len(data_list) == len(image_dict)

    for data in data_list:
        org_idx = data.pop("org_idx")
        assert len(data["image_info"]) == len(image_dict[org_idx])  # check number is correct
        image_list = [image_dict[org_idx][k] for k in sorted(image_dict[org_idx].keys())]
        for i, image in enumerate(image_list):
            data["image_info"][i]["image_base64"] = image

    with open(os.path.join(output_path, shard_name.replace(".jsonl", ".pkl")), "wb") as f:
        pickle.dump(data_list, f)


