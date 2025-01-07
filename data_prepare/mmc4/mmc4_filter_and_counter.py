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

import io
import json
import os
import pickle
import sys

start_idx = int(sys.argv[1])
end_idx = int(sys.argv[2])
print(start_idx, end_idx)


pkl_path = "/dataset/mmc4-core/pkl"
jsonl_path = "/dataset/mmc4-core/jsonl"
output_path = "/dataset/mmc4-core/jsonl-filtered"


pkl_list = sorted(os.listdir(pkl_path))[start_idx:end_idx]

for pkl in pkl_list:
    pickle_path = os.path.join(pkl_path, pkl)
    with open(pickle_path, "rb") as f:
        image_dict = pickle.load(f)
    with open(os.path.join(jsonl_path, pkl.replace(".pkl", ".jsonl"))) as json_file:
        json_list = list(json_file)
    annotation = [json.loads(json_str) for json_str in json_list]

    print(len(annotation), len(image_dict))
    filtered_annotation = []
    for i, anno in enumerate(annotation):
        if i in image_dict:
            assert len(image_dict[i]) == len(anno["image_info"])
            anno["org_idx"] = i
            filtered_annotation.append(anno)
    assert len(filtered_annotation) == len(image_dict)

    with open(os.path.join(output_path, pkl.replace(".pkl", ".jsonl")), "w") as outfile:
        for record in filtered_annotation:
            json.dump(record, outfile)
            outfile.write("\n")

    with open(os.path.join(output_path, pkl.replace(".pkl", ".count")), "w") as f:
        f.write(str(len(filtered_annotation)))


