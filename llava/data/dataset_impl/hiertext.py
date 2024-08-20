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

# refernced from https://github.com/CVC-DAG/OCR_datasets/blob/master/src/datasets/ocr/hiertext.py
import base64
import copy
import io
import json
import logging
import os
import os.path
import os.path as osp
import pathlib
import pickle
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np
import PIL
import torch
import transformers
from PIL import Image
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import Resize

import llava.data.datasets_mixture as datasets_mixture
from llava import conversation as conversation_lib
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.data.dataset_impl.textocr import GenericDataset, preprocess_OCR
from llava.data.datasets_mixture import DATASETS
from llava.train.args import DataArguments, TrainingArguments

DEFAULT_HIERTEXT = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/hiertext"


def bbx_from_vertices_list(vertices):
    # Care about the index, potential source of errors
    return (
        min(vertices, key=lambda x: x[0])[0],
        min(vertices, key=lambda x: x[1])[1],
        max(vertices, key=lambda x: x[0])[0],
        max(vertices, key=lambda x: x[1])[1],
    )


class HierTextDataset(GenericDataset):
    name = "hiertext_dataset"

    def __init__(
        self,
        base_folder=DEFAULT_HIERTEXT,
        split="train",
        handwritten=[True, False],
        legibility=[True, False],
        mode="words",
    ) -> None:
        self.split = f"{split}_legibility-{legibility}_handwritten-{handwritten}"

        annotation_file = json.load(
            open(
                os.path.join(
                    base_folder,
                    "gt",
                    "train.jsonl" if split == "train" else "validation.jsonl",
                ),
            )
        )
        images_path = os.path.join(base_folder, "train" if split == "train" else "validation")
        self.base_images = images_path

        self.samples = []
        self.unique_fpath = set()
        self.unique_samples = defaultdict(list)
        for num, annotation in enumerate(annotation_file["annotations"]):
            image_path = os.path.join(images_path, annotation["image_id"] + ".jpg")
            for paragraph in annotation["paragraphs"]:
                for line in paragraph["lines"]:
                    x, y, x2, y2 = bbx_from_vertices_list(line["vertices"])

                    if x2 * y2 < 225:
                        # skip too small texts
                        continue
                    if x2 - x < y2 - y:
                        continue  # TODO: Evaluation without vertical lines. Not fair.
                    if line["legible"] in legibility and line["handwritten"] in handwritten and not line["vertical"]:
                        if mode == "lines":
                            data = {
                                "bbx": bbx_from_vertices_list(line["vertices"]),
                                "image_path": image_path,
                                "transcription": line["text"],
                                "vertical": line["vertical"],
                            }
                            self.samples.append(data)
                            self.unique_samples[image_path].append(data)
                            self.unique_fpath.add(image_path)
                        else:
                            for word in line["words"]:
                                if not word["vertical"]:
                                    data = {
                                        "bbx": bbx_from_vertices_list(word["vertices"]),
                                        "image_path": image_path,
                                        "transcription": word["text"],
                                        "vertical": word["vertical"],
                                    }
                                    self.samples.append(data)
                                    self.unique_samples[image_path].append(data)
                                    self.unique_fpath.add(image_path)
        self.unique_fpath = list(self.unique_fpath)

    def __len__(self):
        return len(self.unique_fpath)

    def __getitem__(self, idx):
        # metadata = self.samples[idx]
        # img_path = os.path.join(self.base_images, metadata["image_path"])
        # image = Image.open(img_path).convert("RGB")

        img_path = self.unique_fpath[idx]
        metadatas = self.unique_samples[img_path]

        annotations = []
        for metadata in metadatas:
            annotations.append(metadata["transcription"])
        image = Image.open(img_path).convert("RGB")

        return {
            "image_path": img_path,
            "origin_image": image,
            "annotation": annotations,
            "dataset": self.name,
            "split": self.split,
        }


class VILAHierText(Dataset):
    """
    Dataset class for VILA OCR data.

    Args:
        data_path (str): The path to the data.
        image_folder (str): The folder containing the images.
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer for text processing.
        data_args (DataArguments): The data arguments.
        training_args (TrainingArguments): The training arguments.
        split (str, optional): The split of the dataset (default: "train").
        min_area (float, optional): The minimum area of the text (default: 0.001).
    """

    def __init__(
        self,
        data_path,
        image_folder,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        split="train",
        min_area=0.001,
    ) -> None:
        super().__init__()

        data_path = osp.expanduser(data_path)
        self.dataset = HierTextDataset(data_path, split)

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        meta = self.dataset[index]

        img = meta["origin_image"]
        fpath = meta["image_path"]
        texts = " ".join(meta["annotation"])

        return preprocess_OCR(image=img, texts=texts, data_args=self.data_args, tokenizer=self.tokenizer)


if __name__ == "__main__":
    dst = HierTextDataset()
    print(len(dst))
    for i in range(3):
        print(dst[i])
