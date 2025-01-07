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

import llava.data.datasets_mixture as datasets_mixture
from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from llava.data.dataset import preprocess, preprocess_multimodal
from llava.mm_utils import process_image, tokenizer_image_token
from llava.model import *
from llava.train.args import DataArguments, TrainingArguments


@lru_cache(maxsize=16)
def lru_json_load(fpath):
    return json.load(open(fpath))


format2questions = {
    "spatial": [
        "Elaborate on the visual and narrative elements of the image in detail, with a focus on spatial relations.",
        "Describe the image in details, with a focus on spatial relations.",
        "Give a detailed description of the image, focusing on both visual and narrative elements, and the spatial information.",
    ],
    "ocr": [
        "Describe the textual content in the image.",
        "Identify the text visible in this image." "What words or phrases can you identify in the image?",
    ],
    "bbox_interleaved": [
        "Generate a thorough caption for the image and specify where the main elements are positioned.",
        "Write a detialed caption for the image and specify the location of the main objects in [xmin,ymin,xmax,ymax].",
        "Provide a caption for the image, including the coordinates of the main visual elements.",
    ],
}

from llava.data.dataset import LazyCoyoWebDataset


class LazyCoyoWebQADataset(LazyCoyoWebDataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Yunhao."""

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
            self.caption_choice = "/home/ligengz/workspace/coyo-25m-recap"
        else:
            self.caption_choice = data_args.caption_choice.split("+")
        print(f"Current caption choice: {self.caption_choice}.")

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        CONCAT_SAMPLES = False
        # info_list = self.dataset[i - self.idx_offset]

        begin_idx, end_idx = (
            i * self.n_samples_per_idx,
            (i + 1) * self.n_samples_per_idx,
        )
        end_idx = min(end_idx, len(self.dataset))

        input_ids = []
        targets = []
        image_list = []
        block_sizes = []

        for idx in range(begin_idx, end_idx):
            info = self.dataset[idx]
            if ".jpg" in info:
                caption, image_path = info[".txt"], info[".jpg"]
            elif ".png" in info:
                caption, image_path = info[".txt"], info[".png"]
            elif ".webp" in info:
                caption, image_path = info[".txt"], info[".webp"]
            elif ".bmp" in info:
                caption, image_path = info[".txt"], info[".bmp"]
            elif ".tiff" in info:
                caption, image_path = info[".txt"], info[".tiff"]
            else:
                print(info.keys())
                print(info)
                raise KeyError

            if isinstance(image_path, io.BytesIO):
                image_path = Image.open(image_path).convert("RGB")

            if not isinstance(image_path, PIL.Image.Image):
                print(image_path)
                print(info.keys())
                print(type(image_path))
                raise NotImplementedError

            if self.data_args.image_aspect_ratio == "dynamic_s2":
                images, block_size = process_image(
                    image_path, self.data_args, image_folder=None, enable_dynamic_s2=True
                )
                image_list.append(images)
                block_sizes.append(block_size)
                n_images = 1
            elif self.data_args.image_aspect_ratio == "dynamic":
                images = process_image(
                    image_path, self.data_args, image_folder=None, enable_dynamic_res=True, max_tiles=6
                )
                image_list.append(images)
                n_images = len(images)
            else:
                image = process_image(image_path, self.data_args, image_folder=None)
                image_list.append(image.unsqueeze(0))
                n_images = 1

            ## always use original caption
            caption = caption.replace("<image>", "<IMAGE>")
            caption = caption[0].upper() + caption[1:]
            prompt = (
                self.tokenizer.bos_token + (DEFAULT_IMAGE_TOKEN + "\n") * n_images + caption + self.tokenizer.eos_token
            )
            input_ids_i = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")
            targets_i = copy.deepcopy(input_ids_i)
            ## add more qas if exist
            sources = []
            if self.caption_choice is not None:
                # load new captions
                shard = info["__shard__"]
                url = info[".json"]["url"]
                tar_name = osp.relpath(osp.realpath(shard), osp.realpath(self.data_path))
                random.shuffle(self.caption_choice)
                for caption_choice_i in self.caption_choice:
                    shard_json_path = osp.join(caption_choice_i, tar_name + ".json")
                    try:
                        shard_json = lru_json_load(shard_json_path)
                        try:
                            answer = shard_json[url]["output"]
                        except KeyError:
                            # print(f"{url} not in caption. fallback to original caption temporarially")
                            continue
                    except:
                        # print(f"shard_json_path {shard_json_path} not found. fallback to original caption temporarially")
                        continue
                    if "spatial" in caption_choice_i:
                        q = format2questions["spatial"]
                    elif "ocr" in caption_choice_i:
                        q = format2questions["ocr"]
                    elif "bbox_interleaved" in caption_choice_i:
                        q = format2questions["bbox_interleaved"]
                    sources.extend([{"from": "human", "value": np.random.choice(q)}, {"from": "gpt", "value": answer}])
            if sources:
                sources = [sources]
                # sources = preprocess_multimodal(copy.deepcopy([sources]), self.data_args)
                data_dict_qas = preprocess(sources, self.tokenizer, has_image=False, no_system_prompt=True)
                input_ids_qas = data_dict_qas["input_ids"][0, 1:]
                targets_qas = data_dict_qas["labels"][0, 1:]
                input_ids_i = torch.cat([input_ids_i[:-1], input_ids_qas])
                targets_i = torch.cat([targets_i[:-1], targets_qas])

            input_ids.append(input_ids_i)
            targets.append(targets_i)

        input_ids = [
            torch.concat([torch.tensor([self.tokenizer.bos_token_id]), input_ids_i])
            if input_ids_i[0] != self.tokenizer.bos_token_id
            else input_ids_i
            for input_ids_i in input_ids
        ]

        for i in range(len(targets)):
            targets[i][targets[i] == self.tokenizer.pad_token_id] = IGNORE_INDEX

        data_dict = dict(input_ids=input_ids, labels=targets, image=image_list, video=[[], [], [], []])
        if self.data_args.image_aspect_ratio == "dynamic_s2":
            data_dict["block_sizes"] = block_sizes

        return data_dict


