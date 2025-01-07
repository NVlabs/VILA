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
from llava.data.dataset import LazySupervisedDataset
from llava.data.simple_vila_webdataset import VILAWebDataset
from llava.mm_utils import is_gemma_tokenizer, tokenizer_image_token
from llava.model import *
from llava.train.args import DataArguments, TrainingArguments
from llava.train.llava_trainer import LLaVATrainer


@lru_cache(maxsize=16)
def lru_json_load(fpath):
    return json.load(open(fpath))


from llava.data.dataset import LazyCoyoWebDataset


class LazyCoyoWebRecapDataset(LazyCoyoWebDataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ligeng Zhu."""

    num_image_tokens = 576

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        n_samples_per_idx=4,
    ):
        super().__init__(
            data_path=data_path,
            image_folder=image_folder,
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=training_args,
            n_samples_per_idx=n_samples_per_idx,
        )
        if getattr(data_args, "caption_choice", None) is None:
            self.caption_choice = "~/workspace/coyo-25m-recap"
            # nvcode: on
            self.caption_choice = "/home/ligengz/workspace/coyo-25m-recap"
            # nvcode: off
        else:
            self.caption_choice = data_args.caption_choice
        print(f"Current caption choice: {self.caption_choice}.")


