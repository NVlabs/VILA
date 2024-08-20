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

### This is the updated script that people should use ###
###TESTED to work with updated OpenAI APIs####
###Please note create() / AzureOpenAI() instance creation ###

import base64
import glob
import json
import os
import os.path as osp
import time
from pathlib import Path

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import openai
import requests
from openai import AzureOpenAI, OpenAI


def get_oauth_token(p_token_url, p_client_id, p_client_secret, p_scope):
    file_name = "py_llm_oauth_token.json"
    try:
        base_path = Path(__file__).parent
        file_path = Path.joinpath(base_path, file_name)
    except Exception as e:
        print(f"Error occurred while setting file path: {e}")
        return None

    try:
        # Check if the token is cached
        # if os.path.exists(file_path):
        #     with open(file_path, "r") as f:
        #         token = json.load(f)
        # else:
        # Get a new token from the OAuth server
        response = requests.post(
            p_token_url,
            data={
                "grant_type": "client_credentials",
                "client_id": p_client_id,
                "client_secret": p_client_secret,
                "scope": p_scope,
            },
        )
        response.raise_for_status()
        token = response.json()
        with open(file_path, "w") as f:
            json.dump(token, f, indent=2)
    except Exception as e:
        print(f"Error occurred while getting OAuth token: {e}")
        return None

    try:
        # Check if the token is expired
        expires_in = time.time() + token["expires_in"]
        if time.time() > expires_in:
            # Refresh the token
            token = get_oauth_token(p_token_url, p_client_id, p_client_secret, p_scope)
    except Exception as e:
        print(f"Error occurred while while getting OAuth token: {e}")
        return None

    authToken = token["access_token"]
    return authToken


client_id = os.environ.get("NVHOST_OAI_CLIENT_ID", None)
client_secret = os.environ.get("NVHOST_OAI_client_secret", None)
assert (
    client_id is not None and client_secret is not None
), "Please set NVHOST_OAI_CLIENT_ID and NVHOST_OAI_client_secret in your environment variables"
# Please use this URL for retrieving token https://prod.api.nvidia.com/oauth/api/v1/ssa/default/token
token_url = "https://prod.api.nvidia.com/oauth/api/v1/ssa/default/token"
# Please use this Scope for Azure OpenAI: azureopenai-readwrite
scope = "azureopenai-readwrite"

token = get_oauth_token(token_url, client_id, client_secret, scope)
# Define OPENAI Variables and URL
api_base = "https://prod.api.nvidia.com/llm/v1/azure/openai"
deployment_name = "gpt-4-vision-preview"
api_version = "2023-12-01-preview"  # this might change in the future

client = AzureOpenAI(
    api_key=token,
    api_version=api_version,
    base_url=api_base,
)

question = "Elaborate on the visual and narrative elements of the video in detail, particularly the motion behavior"
# "These are frames from a video and please generate a detailed description",
# "These are frames from a video that I want to upload. Generate a detailed description that I can upload along with the video.",
# "These are frames from a video. Please generate a detailed caption to describe.",
# "These are frames from a video that I want to upload. Generate a detailed caption",
# question = "These are frames from a video that I want to upload. Elaborate on the visual and narrative elements of the video in detail that I can upload along with the video,  particularly the motion behavior"


def gpt4_caption_video(_vpath, num_frames):
    # num_frames = 15
    video = cv2.VideoCapture(_vpath)
    base64Frames = []
    count = 0
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = frame_count // num_frames
    print("[video-info]", sample_interval, frame_count, num_frames, _vpath)

    while video.isOpened():
        success, frame = video.read()
        if not success or len(base64Frames) >= num_frames:
            break
        if count % sample_interval == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        count += 1
    video.release()

    print(len(base64Frames), f"frames read from {_vpath}")

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                question,
                # "These are frames from a video and please generate a detailed description",
                # "These are frames from a video that I want to upload. Generate a detailed description that I can upload along with the video.",
                # "These are frames from a video. Please generate a detailed caption to describe.",
                # "These are frames from a video that I want to upload. Generate a detailed caption",
                *map(lambda x: {"image": x}, base64Frames),
            ],
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 1024,
    }

    result = client.chat.completions.create(**params)
    print(result.choices[0].message.content)
    return result.choices[0].message.content


def gpt4_caption_image_then_summarize(_vpath, num_frames):
    # num_frames = 15
    video = cv2.VideoCapture(_vpath)
    base64Frames = []
    count = 0
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = frame_count // num_frames
    print("[video-info]", sample_interval, frame_count, num_frames)

    while video.isOpened():
        success, frame = video.read()
        if not success or len(base64Frames) >= num_frames:
            break
        if count % sample_interval == 0:
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        count += 1
    video.release()

    print(len(base64Frames), f"frames read from {_vpath}")

    image_captions = []
    for frame in base64Frames:
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    "Please describe the image",
                    {"image": frame},
                ],
            },
        ]
        params = {
            "model": "gpt-4-vision-preview",
            "messages": PROMPT_MESSAGES,
            "max_tokens": 1024,
        }

        result = client.chat.completions.create(**params)
        print(result.choices[0].message.content)
        image_captions.append(result.choices[0].message.content)

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": "Below are captions from samples that are sampled from a given video. Please summarize them and make a short caption:\n"
            + "\n".join(image_captions),
        },
    ]
    params = {
        "model": "gpt-4-1106-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 1024,
    }

    result = client.chat.completions.create(**params)
    print(result.choices[0].message.content)
    return result.choices[0].message.content


output_text = {}
num_frames = 10

base_folder = "/lustre/fs3/portfolios/nvr/users/ligengz/workspace/Video-Benchmark/videos"
output_path = f"gpt4v_pexel_1k.json"

if osp.exists(output_path):
    output_text = json.load(open(output_path))

from tqdm import tqdm

for vvpath in glob.iglob(osp.join(base_folder, "**/*.mp4"), recursive=True):
    vvpath = osp.realpath(vvpath)
    if vvpath in output_text:
        print("skip ", vvpath)
        continue

    count = 0
    while True:
        output = None
        try:
            output = gpt4_caption_video(vvpath, num_frames)
        except (openai.RateLimitError, openai.RateLimitError, openai.AuthenticationError) as e:
            print(f"got error {e}, sleep for 10")
            time.sleep(10)
            token = get_oauth_token(token_url, client_id, client_secret, scope)
            client = AzureOpenAI(
                api_key=token,
                api_version=api_version,
                base_url=api_base,
            )
            count += 1
            print(f"[{count}] reinit openai api key")
        except openai.BadRequestError:
            print("invalid request, filted by OAI")
            output = "[openai.BadRequestError] filtered by OAI"
        if output is not None:
            break

    output_text[vvpath] = {
        "question": question,
        "labeler": "gpt4v-nvhost",
        "output": output,
    }
    with open(output_path, "w") as f:
        json.dump(output_text, f, indent=2)
