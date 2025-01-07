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
import glob
import io
import json
import logging
import os
import os.path as osp
import pathlib
import pickle
import random
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from io import BytesIO
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import PIL
import torch

# import transformers
from iopath.common.file_io import g_pathmgr
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import Resize

import llava.data.datasets_mixture as datasets_mixture
from llava import conversation as conversation_lib
from llava.data.dataset import LazySupervisedDataset
from llava.data.dataset_impl.textocr import GenericDataset, preprocess_OCR
from llava.data.simple_vila_webdataset import VILAWebDataset
from llava.data.utils import VILAEncodedVideo
from llava.mm_utils import is_gemma_tokenizer, tokenizer_image_token
from llava.train.args import DataArguments, TrainingArguments


def str2time(s):
    t = datetime.strptime(s, "%H:%M:%S.%f")
    init = datetime.strptime("0:00:00.000", "%H:%M:%S.%f")
    return t, (t - init).total_seconds()


def load_video(video_path, jinfo, idx=0, num_video_frames=8, image_size=334):
    import torch

    # video_path = io.BytesIO(open(video_path, "rb").read())
    # print(jinfo.keys(), jinfo)
    timestamps = jinfo["timestamp"]  # [idx]
    caption = jinfo["caption"]  # [idx]
    duration = jinfo["duration"]

    # begin_t, begin_s = str2time(timestamps[0])
    # end_t, end_s = str2time(timestamps[1])
    try:
        video = VILAEncodedVideo.from_bytesio(video_path, decoder="decord", decode_audio=False)
        duration = float(video.duration)
        # print("DEBUG", duration)
        assert duration >= 0.25
        video_outputs = video.get_clip(start_sec=0, end_sec=video.duration)["video"]
        assert video_outputs.size(1) > num_video_frames
        num_frames = video_outputs.shape[1]
        # step = (num_frames - 1) // 8 + 1
        # NOTE(ligeng): current impl loads the whole (decompressed video) first then slice, may face OOMs for long videos.
        step = num_frames // num_video_frames
        num_frames = num_frames - (num_frames % 8)
        indices = torch.floor(torch.arange(0, num_frames, step)).long()
        video_outputs = video_outputs[:, indices, :, :]
    # except (FileNotFoundError, decord._ffi.base.DECORDError) as e:
    except Exception as e:
        print(f"bad data path {video_path}")
        print(f"Error processing {video_path}: {e}")
        video_outputs = torch.zeros(3, 8, image_size, image_size, dtype=torch.uint8)

    c, b, h, w = video_outputs.size()
    image_tensor = torch.zeros(b, c, image_size, image_size, dtype=torch.uint8)
    video_frames = video_outputs.permute(1, 0, 2, 3).contiguous()
    video_frames = Resize(size=[image_size, image_size], antialias=True)(video_frames)
    image_tensor[:, :, :, :] = video_frames
    # print(begin_s, end_s, caption)
    return image_tensor, caption  # , (begin_s, end_s)


class VILAPanda70m(Dataset):
    def __init__(
        self,
        data_path,
        image_folder,
        tokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ) -> None:
        super().__init__()

        data_path = osp.expanduser(data_path)
        # self.dataset = VILAWebDataset(data_path)
        self.dataset = VILAWebDataset(
            data_path=data_path,
        )

        self.data_path = data_path
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.num_video_frames = data_args.num_video_frames if hasattr(data_args, "num_video_frames") else 8
        self.loader_fps = data_args.fps if hasattr(data_args, "fps") else 0.0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]

        # TODO: we shall make sure no key is missing in panda70m.
        try:
            video_path = data[".mp4"]
        except KeyError:
            video_path = None
            print("bad data", data)

        if ".json" in data:
            jinfo = data[".json"]
            caption = jinfo["caption"]
        else:
            caption = "This is a sample video from Youtube."
        from llava.mm_utils import opencv_extract_frames

        imgs, frames_loaded = opencv_extract_frames(video_path, self.num_video_frames, self.loader_fps)
        cap = caption
        if frames_loaded == 0:
            cap = "Empty video."
        frames_loaded_successfully = len(imgs)

        prompt = "<image>\n" * frames_loaded_successfully + cap
        processor = self.data_args.image_processor
        image_tensor = [
            # processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in torch.unbind(image_tensor)
            processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
            for image in imgs
        ]
        image_tensor = torch.stack(image_tensor)

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            return_tensors="pt",
        )
        targets = copy.deepcopy(input_ids)
        data_dict = dict(input_ids=input_ids, labels=targets, image=image_tensor)

        return data_dict


def with_opencv(filename):
    video = cv2.VideoCapture(filename)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    return duration, fps, frame_count


def cleanup_corrupted_videos(
    workdir="~/dataset/panda70m/panda70m_training_2m",
    shards=0,
    total=-1,
):
    workdir = osp.expanduser(workdir)
    print("workdir", workdir)
    video_list = glob.glob(f"{workdir}/*.mp4")
    video_list = sorted(video_list)
    if total > 0:
        chunk = len(video_list) // total
        begin_idx = shards * chunk
        end_idx = (shards + 1) * chunk
        if shards == total - 1:
            end_idx = len(video_list)
        video_list = video_list[begin_idx:end_idx]
    print(f"checking total {len(video_list)} videos")
    # return

    debug_info = {}
    for idx, video_path in enumerate(video_list):
        print(f"[{idx}/{len(video_list)}]", video_path)
        json_path = video_path.replace(".mp4", ".json")

        try:
            assert osp.exists(json_path) and osp.exists(video_path)
            jinfo = json.load(open(json_path))
            info = with_opencv(video_path)
            print(info)
            video = VILAEncodedVideo.from_bytesio(video_path, decoder="decord", decode_audio=False)
        except (RuntimeError, ZeroDivisionError) as e:
            debug_info[video_path] = str(e)
            print(f"!! deleting wrong [{idx}/{len(video_list)}]", video_path)  # , type(e))
            os.remove(video_path)
            os.remove(json_path)
            # input()
            time.sleep(3)

    print(debug_info)


if __name__ == "__main__":
    import fire

    fire.Fire(cleanup_corrupted_videos)
    exit(0)

    jinfo = json.load(open(json_path))
    img_t = load_video(video_path, jinfo=jinfo)
    print(img_t)


