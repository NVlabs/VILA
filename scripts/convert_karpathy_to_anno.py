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

# import json

# # coco for reference
# # dict_keys(['images', 'dataset']) -> ["images"]
# karpathy = json.load(open("/home/jil/datasets/karpathy_json/dataset_coco.json"))
# # dict_keys(['info', 'images', 'licenses', 'annotations']) -> ['images', 'annotations']]
# anno = json.load(open("/tmp/coco/annotations/captions_val2014.json"))
# # assert len(karpathy["images"]) == len(anno["images"])  == len(anno["annotations"]), (
# #     len(karpathy["images"]), len(anno["images"]), len(anno["annotations"])  # (123287, 40504, 202654)
# # )


# karpathy_flickr = json.load(open("/home/jil/datasets/karpathy_json/dataset_coco.json"))
# anno_flickr = {
#     "images": [],
#     "annotations": [],
# }

# print(karpathy["images"][0])
# print(anno["images"][0])
# print(anno["annotations"][:3])

# image_id_set = set([_["id"] for _ in anno["images"]])
# anno_set = set([_["id"] for _ in anno["annotations"]])

# print(len(anno_set))


import argparse
import json

from tqdm import tqdm


def main(input_json, output_json, split):
    annot_format = {
        "info": {
            "year": 2014,
            "version": "1.0",
            "description": "This is stable 1.0 version of the 2014 MS COCO dataset.",
            "contributor": "Microsoft COCO group",
            "url": "http://mscoco.org",
            "date_created": "2015-01-27 09:11:52.357475",
        },
        "licenses": [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
            },
            {
                "url": "http://creativecommons.org/licenses/by-nc/2.0/",
                "id": 2,
                "name": "Attribution-NonCommercial License",
            },
            {
                "url": "http://creativecommons.org/licenses/by-nc-nd/2.0/",
                "id": 3,
                "name": "Attribution-NonCommercial-NoDerivs License",
            },
            {"url": "http://creativecommons.org/licenses/by/2.0/", "id": 4, "name": "Attribution License"},
            {
                "url": "http://creativecommons.org/licenses/by-sa/2.0/",
                "id": 5,
                "name": "Attribution-ShareAlike License",
            },
            {"url": "http://creativecommons.org/licenses/by-nd/2.0/", "id": 6, "name": "Attribution-NoDerivs License"},
            {"url": "http://flickr.com/commons/usage/", "id": 7, "name": "No known copyright restrictions"},
            {"url": "http://www.usa.gov/copyright.shtml", "id": 8, "name": "United States Government Work"},
        ],
        "type": "captions",
        "images": [],
        "annotations": [],
    }

    with open(input_json) as f:
        dataset = json.load(f)
        annotations = dataset["images"]
        dataset_name = dataset["dataset"]

    count = 0
    print(f"Converting Karpathy {dataset_name} {split} to COCO Format...")
    for annot in tqdm(annotations):
        if split == "all" or (annot["split"] == split):
            image_id = str(annot["filename"].split(".")[0])  # annot['imgid']
            annot_format["images"].append(
                {
                    "id": image_id,
                    "width": 512,
                    "height": 512,
                    "filename": annot["filename"],
                    "license": 1,
                    "flickr_url": "",
                    "coco_url": "",
                    "date_captured": "",
                }
            )

            for sent in annot["sentences"]:
                annot_format["annotations"].append({"id": sent["sentid"], "image_id": image_id, "caption": sent["raw"]})
                count += 1

    with open(output_json, "w") as f:
        json.dump(annot_format, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", type=str, default="/home/jil/datasets/karpathy_json/dataset_flickr30k.json")
    parser.add_argument("--output-json", type=str, default="/home/jil/datasets/flickr30k/flickr30k_coco_all.json")
    parser.add_argument("--split", type=str, default="all")
    args = parser.parse_args()

    main(args.input_json, args.output_json, args.split)
