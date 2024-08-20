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
from llava.data.dataset import LazySupervisedDataset
from llava.data.datasets_mixture import DATASETS
from llava.data.simple_vila_webdataset import VILAWebDataset
from llava.mm_utils import is_gemma_tokenizer, tokenizer_image_token
from llava.model import *
from llava.train.args import DataArguments, TrainingArguments
from llava.train.llava_trainer import LLaVATrainer

DEFAULT_TEXTOCR = "~/nvr_elm_llm/dataset/TextOCR"
DEFAULT_TEXTOCR = osp.expanduser(DEFAULT_TEXTOCR)


class GenericDataset:
    """
    A class representing a generic dataset.

    This class provides methods for adding datasets and resizing images.

    Attributes:
        image_height (int): The desired height of the image.
        patch_width (int): The width of the patch.

    Methods:
        add(dataset): Adds a dataset to the current dataset.
        resize_image(image): Resizes the given image to the desired height and width.

    """

    def add(self, dataset):
        return SummedDataset(self, dataset)

    def resize_image(self, image):
        original_width, original_height = image.size

        original_height = max(original_height, 1)
        original_width = max(original_width, 1)

        scale = self.image_height / original_height

        resized_width = int(round(scale * original_width, 0))
        new_width = resized_width + (self.patch_width - (resized_width % self.patch_width))  # Adjusted this line

        return image.resize((new_width, self.image_height))

    def __add__(self, dataset):
        return self.add(dataset)


class SummedDataset(GenericDataset):
    def __init__(self, dataset_left, dataset_right) -> None:
        self.left = dataset_left
        self.right = dataset_right

    def __len__(self):
        return len(self.left) + len(self.right)

    def __getitem__(self, idx):
        if idx > (len(self.left) - 1):
            idx_corrected = idx % len(self.left)
            return self.right[idx_corrected]

        return self.left[idx]


class TextOCRDataset(GenericDataset):
    name = "text_ocr_dataset"

    def __init__(
        self,
        base_folder=DEFAULT_TEXTOCR,
        split="train",
        transforms=lambda x: x,
        min_area=0.001,
    ) -> None:
        super().__init__()

        self.split = split
        self.transforms = transforms
        self.data = []
        self.img2text = {}

        annotations = json.load(open(os.path.join(base_folder, f"TextOCR_0.1_{split}.json")))
        valid_images = [
            {
                "size": (
                    annotations["imgs"][img]["width"],
                    annotations["imgs"][img]["height"],
                ),
                "path": os.path.join(
                    base_folder,
                    annotations["imgs"][img]["file_name"].replace("train/", "train_images/"),
                ),
                "annots": [str(i) for i in annotations["imgToAnns"][img]],
            }
            for img in annotations["imgs"]
        ]

        for image in valid_images:
            for ann in image["annots"]:
                annotation = annotations["anns"][ann]
                if annotation["utf8_string"] == ".":
                    continue  # Unreadable characters

                x, y, w, h = (int(x) for x in annotation["bbox"])
                img_area = image["size"][0] * image["size"][1]
                if (w * h) / img_area < min_area:
                    continue  # skip too small texts

                fpath = image["path"]
                self.data.append(
                    {
                        "image_path": fpath,
                        "bbx": [int(x) for x in annotation["bbox"]],
                        "transcription": annotation["utf8_string"],
                    }
                )

                if fpath not in self.img2text:
                    self.img2text[fpath] = []
                self.img2text[fpath].append(
                    {
                        "bbx": [int(x) for x in annotation["bbox"]],
                        "transcription": annotation["utf8_string"],
                    }
                )

        self.image_ids = list(self.img2text.keys())

    def __len__(self):
        # return len(self.data)
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        metadata = self.img2text[img_id]
        origin_image = Image.open(img_id).convert("RGB")

        annotation = [_["transcription"] for _ in metadata]
        bboxes = [_["bbx"] for _ in metadata]

        return {
            "image_path": img_id,
            "origin_image": origin_image,
            "annotation": annotation,
            "bboxes": bboxes,
            "dataset": self.name,
            "split": self.split,
        }

    def __getitem__single__(self, idx):
        metadata = self.data[idx]
        origin_image = Image.open(metadata["image_path"])
        x, y, w, h = metadata["bbx"]
        cropped_image = Image.open(metadata["image_path"]).crop((x, y, x + w, y + h)).convert("RGB")
        return {
            "image_path": metadata["image_path"],
            "origin_image": origin_image,
            "cropped_image": cropped_image,
            "annotation": metadata["transcription"],
            "dataset": self.name,
            "split": self.split,
            "tokens": [char for char in metadata["transcription"]],
        }


def preprocess_OCR(image, texts: list, data_args, tokenizer):
    text = " ".join(texts)
    caption = f"Please read the texts on image and type it below, each word separated by space.\n{text}"

    caption = DEFAULT_IMAGE_TOKEN + caption + tokenizer.eos_token
    vila_img = LazySupervisedDataset._process_image(image, data_args, image_folder=None)

    input_ids = tokenizer_image_token(
        caption,
        tokenizer,
        return_tensors="pt",
    )

    targets = copy.deepcopy(input_ids)
    # mask image tokens is unnecessary for llava-1.5
    # targets[targets == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
    for i in range(len(targets)):
        targets[i][targets[i] == tokenizer.pad_token_id] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=targets,
        image=vila_img.unsqueeze(0),
    )


class VILATextOCR(Dataset):
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
        self.dataset = TextOCRDataset(data_path, split, min_area=min_area)

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        meta = self.dataset[index]

        img = meta["origin_image"]
        fpath = meta["image_path"]
        text = " ".join(meta["annotation"])

        caption = f"Please read the texts on image and type it below, each word separated by space.\n{text}"

        data = preprocess_OCR(image=img, texts=text, data_args=self.data_args, tokenizer=self.tokenizer)
        return data


if __name__ == "__main__":
    from pprint import pprint

    dataset = TextOCRDataset()
    # dataset = VILATextOCR()
    print(len(dataset))

    for idx in range(2):
        pprint(dataset[idx])
