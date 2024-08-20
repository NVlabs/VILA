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
import glob
import json
import os
import os.path as osp
import tempfile
import time
from io import BytesIO

import vertexai
import vertexai.preview.generative_models as generative_models
from vertexai.generative_models import FinishReason, GenerativeModel
from vertexai.generative_models import Image as vertextaiImage
from vertexai.generative_models import Part

# from PIL import Image
from llava.mm_utils import opencv_extract_frames

vertexai.init(project="gemini-pro-15-420722", location="us-central1")
mname = "gemini-1.5-pro-preview-0409"
model = GenerativeModel("gemini-1.5-pro-preview-0409")
# model = GenerativeModel("gemini-1.5-flash-001")

responses = model.generate_content(
    ["how are you today?"],
    stream=True,
)

for response in responses:
    print(response.text, end="")


frames = 15

# base_folder = "/lustre/fsw/portfolios/nvr/users/yukangc/download_videos/videos"
base_folder = "/lustre/fs3/portfolios/nvr/users/ligengz/workspace/Video-Benchmark/videos"
output_path = f"vertex-ai-gemini-15_pexel_1k_new_prompt.json"

question_formats = [
    "Create a narrative representing the video presented",
    "Share a interpretation of the video provided",
    "Offer a explanation of the footage presented",
    "Render a summary of the video below",
    "Summarize the visual content of the following video",
    "Write an informative summary of the video",
    "Present a description of the clip's key features",
    "Relay an account of the video shown",
    "Provide a description of the given video",
    "Describe the following video",
    "Give a explanation of the subsequent video",
]
import random

question = "Elaborate on the visual and narrative elements of the video in detail, particularly the motion behavior"
output_text = {}

if osp.exists(output_path):
    output_text = json.load(open(output_path))


def pil2vertexIMG(pil_imgs):
    os.makedirs("tmp", exist_ok=True)
    vertex_imgs = []
    for idx, img in enumerate(pil_imgs):
        img.save(f"tmp/{idx}.png")
        vertex_imgs.append(vertextaiImage.load_from_file(f"tmp/{idx}.png"))
    return vertex_imgs


import cv2
from tqdm import tqdm

# jinfo = json.load(
#     open(
#         "/lustre/fs3/portfolios/nvr/users/ligengz/workspace/Video-Benchmark/video_benchmark_label_data.json"
#     )
# )
# for vvpath in tqdm(jinfo.keys()):
for vvpath in tqdm(glob.glob(osp.join(base_folder, "**/*.mp4"), recursive=True)):
    _vpath = osp.realpath(vvpath)
    print(_vpath, _vpath in output_text)
    if _vpath in output_text:
        print("\tAlready processed ")
        continue
    vpath = BytesIO(open(_vpath, "rb").read())
    try:
        videos = opencv_extract_frames(vpath, frames)
    except (cv2.error, ZeroDivisionError):
        print("error: ", vpath)
        continue
    videos = pil2vertexIMG(videos)

    print("\tLaunch labeling through vertex API.")
    try:
        response = model.generate_content(
            [
                # "Please describe the video in details",
                *videos,
                # question,
                random.choice(question_formats),
            ]
        )
    except:
        print("google.api_core.exceptions.ResourceExhausted")
        time.sleep(10)
        continue
    # print(response)
    try:
        # print(response.text)
        output = response.text
    except Exception as e:
        # print(response)
        output = "[NA] Gemini refused to answer."

    # output_text[osp.relpath(_vpath, base_folder)] = response.text
    # output_text[_vpath] = output
    output_text[osp.realpath(_vpath)] = {
        "input": question,
        "output": output,
        "labeler": mname,
    }
    with open(output_path, "w") as f:
        json.dump(output_text, f, indent=2)

    # time.sleep(10)
