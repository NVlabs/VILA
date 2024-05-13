# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Inference test to run all examples from the paper and compare w/ expected output.
Both the inference results and expected output will be printed out.

Currently do not support multi-turn chat. Each time an image and question are input and answer is output.
"""

import argparse
import json
import os

import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, process_images,
                            tokenizer_image_token)
from llava.model import *

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"


from llava.model.builder import load_pretrained_model


def eval_model(args, model, tokenizer, image_processor):
    # read json file
    with open(args.test_json_path) as f:
        all_test_cases = json.load(f)

    result_list = []
    print(len(all_test_cases["test_cases"]))

    for test_case in all_test_cases["test_cases"]:
        # read images first
        image_file_list = test_case["image_paths"]
        image_list = [
            Image.open(os.path.join(args.test_image_path, image_file)).convert("RGB") for image_file in image_file_list
        ]
        image_tensor = process_images(image_list, image_processor, model.config)

        # image_tokens = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len

        for i in range(len(test_case["QAs"])):
            query = test_case["QAs"][i]["question"]
            query_text = query

            if 1:
                # query = query.replace("<image>", image_tokens)
                if len(image_list) < 3:
                    conv = conv_templates["vicuna_v1"].copy()
                else:
                    conv = conv_templates["vicuna_v1_nosys"].copy()
                conv.append_message(conv.roles[0], query)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
            else:
                conv = conv_templates[args.conv_mode].copy()
                if not "<image>" in query:
                    assert "###" not in query  # single query
                    query = image_tokens + "\n" + query  # add <image>
                    query_list = [query]
                else:
                    query_list = query.split("###")
                    assert len(query_list) % 2 == 1  # the last one is from human

                    new_query_list = []
                    for idx, query in enumerate(query_list):
                        if "<image>" in query:
                            assert idx % 2 == 0  # only from human
                            # assert query.startswith("<image>")
                        # query = query.replace("<image>", image_tokens)
                        new_query_list.append(query)
                    query_list = new_query_list

                for idx, query in enumerate(query_list):
                    conv.append_message(conv.roles[idx % 2], query)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

            print("%" * 10 + " " * 5 + "VILA Response" + " " * 5 + "%" * 10)

            # inputs = tokenizer([prompt])
            inputs = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX)
            input_ids = torch.as_tensor(inputs).cuda().unsqueeze(0)

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

            # outputs = run_llava.process_outputs(args, model, tokenizer, input_ids, image_tensor, stopping_criteria, stop_str)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=0.7,
                    # top_p=args.top_p,
                    # num_beams=args.num_beams,
                    max_new_tokens=512,
                    # use_cache=True,
                    stopping_criteria=[stopping_criteria],
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            outputs = outputs.strip()

            print(f"Question: {query_text}")
            print(f"VILA output: {outputs}")
            print(f'Expected output: {test_case["QAs"][i]["expected_answer"]}')

            result_list.append(
                dict(question=query_text, output=outputs, expected_output=test_case["QAs"][i]["expected_answer"])
            )
    return result_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--test_json_path", type=str, default=None)
    parser.add_argument("--test_image_path", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--pad", action="store_true")

    args = parser.parse_args()

    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_name, "llava_llama", None)
    result_list = eval_model(args, model, tokenizer, image_processor)
    save_name = f"inference-test_{args.model_name.split('/')[-1]}"
    if "nosys" in args.conv_mode:
        save_name += "_nosys"
    save_name += ".json"
    result_list_str = json.dumps(result_list, indent=2)
