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
from llava.data.dataset import LazySupervisedDataset, lru_json_load
from llava.data.datasets_mixture import DATASETS
from llava.data.simple_vila_webdataset import VILAWebDataset
from llava.mm_utils import is_gemma_tokenizer, opencv_extract_frames, process_image, tokenizer_image_token
from llava.model import *
from llava.train.args import DataArguments, TrainingArguments
from llava.train.llava_trainer import LLaVATrainer

# @lru_cache(maxsize=16)
# def lru_json_load(fpath):
#     return json.load(open(fpath, "r"))


class LazySAMWebDataset(Dataset):
    """Dataset for SAM high resolution data.
    This class is implemented by Ligeng Zhu."""

    num_image_tokens = 576

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        n_samples_per_idx=1,
    ):
        super().__init__()

        print("[DEBUG] ", osp.abspath(data_path))
        self.dataset = VILAWebDataset(
            data_path=osp.abspath(data_path),
        )
        if data_args.start_idx >= 0 and data_args.end_idx >= 0:
            # Ligeng: support slicing for ablate different subsets.
            total = len(self.dataset)
            start_idx = int(total * data_args.start_idx)
            end_idx = int(total * data_args.end_idx)
            print(f"loading subset from {start_idx} to {end_idx}, total {total}")
            self.dataset = torch.utils.data.Subset(self.dataset, range(start_idx, end_idx))

        # For caption choice,
        #   if None: use original caption
        #   if a folder path: use specified caption to override original one (choice1)
        #   if a folder path: use specified caption and concat with original one (choice2)
        self.caption_choice = None
        self.caption_choice_2 = None
        self.data_path = data_path

        if data_args.caption_choice is not None:
            self.caption_choice = data_args.caption_choice
            print("[recap] Override coyo caption using ", self.caption_choice)

        if data_args.caption_choice_2 is not None:
            self.caption_choice_2 = data_args.caption_choice_2
            print("[recapv2] Override coyo caption using ", self.caption_choice_2)

        print("total samples", len(self.dataset))
        # rank = int(os.environ["RANK"])
        # world_size = int(os.environ["WORLD_SIZE"])

        self.n_samples_per_idx = n_samples_per_idx
        # self.n_samples = len(self.dataset) // n_samples_per_idx
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        return len(self.dataset) // self.n_samples_per_idx

    @property
    def modality_lengths(self):
        # Estimate the number of tokens after tokenization, used for length-grouped sampling
        length_list = []
        for samples in self.data_list:
            cur_len = sum([len(conv["text" if "text" in conv else "caption"].split()) for conv in samples])
            # The unit of cur_len is "words". We assume 1 word = 2 tokens.
            cur_len = cur_len + len(samples) * self.num_image_tokens // 2
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        CONCAT_SAMPLES = False
        # info_list = self.dataset[i - self.idx_offset]

        begin_idx, end_idx = (
            i * self.n_samples_per_idx,
            (i + 1) * self.n_samples_per_idx,
        )
        end_idx = min(end_idx, len(self.dataset))

        text_list = []
        image_list = []

        for idx in range(begin_idx, end_idx):
            info = self.dataset[idx]
            if ".jpg" in info:
                image = info[".jpg"]
            elif ".png" in info:
                image = info[".png"]
            elif ".webp" in info:
                image = info[".webp"]
            elif ".bmp" in info:
                image = info[".bmp"]
            elif ".tiff" in info:
                image = info[".tiff"]
            else:
                print(info.keys())
                print(info)
                raise KeyError

            assert self.caption_choice is not None or self.caption_choice_2 is not None
            # load new captions
            shard = info["__shard__"]
            # shard_key = info["__key__"].replace("./", "")
            # url = osp.join(shard, shard_key)
            url = osp.join(info["__shard__"], str(info["__shardindex__"]))

            tar_name = osp.relpath(osp.realpath(shard), osp.realpath(self.data_path))
            # tar_name = osp.dirname(shard)

            caption = None
            if self.caption_choice is not None:
                # load new captions
                shard_json_path = osp.join(self.caption_choice, tar_name + ".json")
                try:
                    shard_json = lru_json_load(shard_json_path)
                    caption = shard_json[url]["output"]
                except KeyError:
                    print(f"{url} not in caption. fallback to original caption temporarially")

            if self.caption_choice_2 is not None:
                shard_json_path = osp.join(self.caption_choice_2, tar_name + ".json")
                try:
                    shard_json = lru_json_load(shard_json_path)
                    new_caption = shard_json[url]["output"]
                    caption = f"### Short description: \n{caption} \n\n" + f"### Long description: \n{new_caption}"
                    # print(f"{url} in caption. use new caption")
                except (KeyError, FileNotFoundError):
                    print(f"{url} not in caption. fallback to original caption temporarially")

            if caption is None:
                caption = ""
                print(f"{url} not found in {self.caption_choice} or {self.caption_choice_2} ")

            caption = caption.replace("<image>", "<IMAGE>")
            text_list.append(DEFAULT_IMAGE_TOKEN + caption + self.tokenizer.eos_token)

            if isinstance(image, io.BytesIO):
                image = Image.open(image).convert("RGB")

            if not isinstance(image, PIL.Image.Image):
                print(image)
                print(info.keys())
                print(type(image))
                raise NotImplementedError

            image_list.append(image)

        image_list = torch.stack([process_image(image, self.data_args, self.image_folder) for image in image_list])

        input_ids = [
            tokenizer_image_token(
                prompt,
                self.tokenizer,
                return_tensors="pt",
            )
            for prompt in text_list
        ]

        targets = copy.deepcopy(input_ids)
        # mask image tokens is unnecessary for llava-1.5
        # targets[targets == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        for i in range(len(targets)):
            targets[i][targets[i] == self.tokenizer.pad_token_id] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=targets, image=image_list)


if __name__ == "__main__":
    data_path = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/sam-reformat"
    dst = VILAWebDataset(
        data_path=osp.abspath(data_path),
    )
    # print(dst[0])
    for idx, data in enumerate(dst):
        print(idx, data.keys())
