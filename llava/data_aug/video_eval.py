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

# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import glob
import json
import math
import os
import os.path as osp
import signal

import numpy as np
import shortuuid
import torch
from PIL import Image
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Resize
from tqdm import tqdm

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


# This function will be called when the timeout is reached
def handler(signum, frame):
    raise TimeoutError()


# Set the signal handler
signal.signal(signal.SIGALRM, handler)

import shutil

from filelock import FileLock


def safely_merge_info(out_fpath, info):
    out_folder = osp.dirname(out_fpath)
    if len(out_folder) > 2:
        os.makedirs(out_folder, exist_ok=True)
    with FileLock(out_fpath + ".lock"):
        if osp.exists(out_fpath):
            try:
                new_info = json.load(
                    open(out_fpath, "r+"),
                )
                info.update(new_info)
            except json.decoder.JSONDecodeError:
                pass
        json.dump(info, open(out_fpath + ".meta", "w+"), indent=2)
        shutil.move(out_fpath + ".meta", out_fpath)
    return info


def get_model_output(
    model,
    image_processor,
    tokenizer,
    video_path,
    qs,
    conv_mode="vicuna_v1",
    num_video_frames=8,
    temperature=0.2,
    num_beams=1,
):
    from llava.mm_utils import opencv_extract_frames

    imgs, num_frames = opencv_extract_frames(video_path, num_video_frames)
    # print(imgs)

    image_tensor = [
        # processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in torch.unbind(image_tensor)
        image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        for image in imgs
    ]
    image_tensor = torch.stack(image_tensor)

    qs = "<image>\n" * num_video_frames + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    input_ids = torch.unsqueeze(input_ids, 0)
    input_ids = torch.as_tensor(input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = True
    if temperature == 0:
        do_sample = False
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.half().cuda(),
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=1024,
            num_beams=num_beams,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs


template = r""" Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{question}
The best answer is:
"""


def eval_model(args):
    from pprint import pprint

    pprint(vars(args))
    output_name = osp.basename(args.model_path) + f"_tmp={args.temperature}_beams={args.num_beams}" + "video_mme.json"
    output_json = []
    labeled_key = {}
    if osp.exists(output_name):
        labeled_key = json.load(open(output_name))
    print("already answered ", len(labeled_key.keys()), output_name)

    jinfo = json.load(open("/home/ligengz/workspace/video-mme/Video-MME.json"))
    folder = "/home/ligengz/workspace/video-mme/ytb_videos"

    if args.convert:
        for vmeta in jinfo:
            for question in vmeta["questions"]:
                qid = question["question_id"]
                if qid in labeled_key:
                    question["response"] = labeled_key[qid]["response"]
                else:
                    print("missing", qid)
                    question["response"] = "C"
        with open(output_name.replace(".json", "_converted.json"), "w") as fp:
            json.dump(jinfo, fp, indent=2)
        return 0

    begin_idx = 0
    end_idx = len(jinfo)
    if args.total > 0:
        chunk = len(jinfo) // args.total
        begin_idx = chunk * args.shard
        end_idx = min(chunk * (args.shard + 1), len(jinfo))
        print(f"labeling btw {begin_idx}-{end_idx}, {chunk} from {len(jinfo)}")

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    print(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)

    for idx, vmeta in tqdm(enumerate(jinfo), total=len(jinfo)):
        if not (idx >= begin_idx and idx < end_idx):
            continue
        url = vmeta["url"]
        video_id = vmeta["video_id"]
        uid = osp.basename(url).split("?v=")[-1]
        vpath = osp.join(folder, f"{uid}.mp4")

        if not osp.exists(vpath):
            print("[video not downloaded] Skip", vpath)
            continue

        for questions in vmeta["questions"]:
            qid = questions["question_id"]
            if qid in labeled_key:
                print("[question id answered] Skip", qid, url)
                continue
            qa = questions["question"] + "\n" + "\n".join(questions["choices"])
            qs = template.format(question=qa)
            output = get_model_output(
                model,
                image_processor,
                tokenizer,
                vpath,
                qs,
                conv_mode=args.conv_mode,
                temperature=args.temperature,
                num_beams=args.num_beams,
            )
            questions["response"] = output
            labeled_key[questions["question_id"]] = questions
        # break
        # output_json.append(vmeta)
        safely_merge_info(output_name, labeled_key)
        # with open(output_name, "w") as fp:
        #     json.dump(output_json, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("-c", "--convert", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--total", type=int, default=-1)
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-3b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model_max_length", type=int, required=False, default=5120)
    # parser.add_argument('--video_dir', help='Directory containing video files.', default="~/workspace/vila-captioner-avfm/videos")
    parser.add_argument(
        "--output_name", help="Name of the file for storing results JSON.", default="video_inference_dev.json"
    )
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    args = parser.parse_args()

    eval_model(args)
