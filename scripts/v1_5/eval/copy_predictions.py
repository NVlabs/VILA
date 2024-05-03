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

import os
import shutil
import sys

def main(model_name):
    # Define the dataset and prediction file mapping
    datasets = {
        "vqav2": "playground/data/eval/vqav2/answers_upload/llava_vqav2_mscoco_test-dev2015/{}.json".format(model_name),
        "vizwiz": "playground/data/eval/vizwiz/answers_upload/{}.json".format(model_name),
        "mmbench": "playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712/{}.xlsx".format(model_name),
        "mmbench_cn": "playground/data/eval/mmbench_cn/answers_upload/mmbench_dev_cn_20231003/{}.xlsx".format(model_name),
        "mmmu": "playground/data/eval/MMMU/test_results/{}.json".format(model_name),
    }

    # Create the base directory
    base_dir = "playground/data/predictions_upload/{}".format(model_name)
    os.makedirs(base_dir, exist_ok=True)

    # Copy each prediction file to its new location
    for dataset, pred_file in datasets.items():
        # Create the directory for the dataset
        dataset_dir = os.path.join(base_dir, dataset)
        os.makedirs(dataset_dir, exist_ok=True)

        # Copy the prediction file
        dest_file = os.path.join(dataset_dir, os.path.basename(pred_file))
        shutil.copy(pred_file, dest_file)
        print("Copied {} to {}".format(pred_file, dest_file))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <model_name>")
    else:
        model_name = sys.argv[1]
        main(model_name)
