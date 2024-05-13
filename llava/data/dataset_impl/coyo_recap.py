import os, os.path as osp
import base64
import copy
import llava.data.datasets_mixture as datasets_mixture

import PIL
from llava.data.datasets_mixture import DATASETS
from dataclasses import dataclass, field
import io
import numpy as np
import random
import json
import logging
import pathlib
import pickle
import time
from typing import Dict, Optional, Sequence, List
import re

import torch

import transformers

from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from torch.utils.data import ConcatDataset, Dataset
from llava.train.llava_trainer import LLaVATrainer
from llava.train.args import TrainingArguments, DataArguments

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token, is_gemma_tokenizer

from torchvision.transforms import Resize
from pytorchvideo.data.encoded_video import EncodedVideo

from PIL import Image
from functools import lru_cache

from llava.data.simple_vila_webdataset import VILAWebDataset
from llava.data.dataset import LazySupervisedDataset

@lru_cache(maxsize=16)
def lru_json_load(fpath):
    return json.load(open(fpath, "r"))

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
            n_samples_per_idx=n_samples_per_idx
        )
        self.caption_choice = "/home/ligengz/workspace/coyo-25m-recap"