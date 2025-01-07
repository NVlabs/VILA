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

non_repeat_datasets = [
    "GEOS(MathV360K)",
    "figureqa(cauldron,llava_format)",
    "chart2text(cauldron)",
    "chrome_writting",
    "iiit5k",
    "image_textualization(filtered)",
    "k12_printing",
    "mavis_math_metagen",
    "mavis_math_rule_geo",
    "scienceqa(nona_context)",
    "UniGeo(MathV360K)",
    "sroie",
    # counted in intern
    "tqa(cauldron,llava_format)",
    "PMC-VQA(MathV360K)",
    "Super-CLEVR(MathV360K)",
]


def filter_valid_templates(data_list):
    def is_valid_template(template):
        return all(message["from"] in ["human", "gpt"] for message in template["conversations"])

    ret = list(filter(is_valid_template, data_list))
    print("original:", len(data_list), "filtered:", len(ret))
    return ret


def main(save_path="/raid/kentang/datasets/LLaVA-OneVision-Data-processed/"):

    metadata_path = os.path.join(save_path, "metadata")
    dataset_names = sorted(os.listdir(metadata_path))

    def load_jsonl(file_path):
        data = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    all_data = []
    cnt = 0
    for dataset_name in dataset_names:
        if dataset_name.replace("_train.jsonl", "") not in non_repeat_datasets:
            continue
        _loaded = load_jsonl(os.path.join(metadata_path, dataset_name))
        loaded = filter_valid_templates(_loaded)

        id_offset = len(all_data)
        for item in loaded:
            item["id"] += id_offset
        all_data += loaded
        print(cnt, dataset_name, len(all_data), all_data[-1])
        cnt += 1

    with open(os.path.join(save_path, "llava_onevision_sft_images_only_adlr_train.jsonl"), "w") as f:
        for item in all_data:
            json.dump(item, f)
            f.write("\n")


if __name__ == "__main__":
    import fire

    fire.Fire(main)


