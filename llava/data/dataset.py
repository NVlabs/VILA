# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

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
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import PIL
import torch
import transformers
from datasets import concatenate_datasets, load_dataset
from PIL import Image, ImageFile
from pytorchvideo.data.encoded_video import EncodedVideo
from torch.utils.data import ConcatDataset, Dataset
from torchvision.transforms import Resize
from transformers import PreTrainedTokenizer

import llava.data.datasets_mixture as datasets_mixture
from llava import conversation as conversation_lib
from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                             IMAGE_TOKEN_INDEX)
from llava.data.datasets_mixture import DATASETS
from llava.eval.mmmu_utils.data_utils import (CAT_SHORT2LONG, construct_prompt,
                                              load_yaml, process_single_sample)
from llava.mm_utils import is_gemma_tokenizer, tokenizer_image_token, opencv_extract_frames, process_image
from llava.model import *
from llava.train.args import DataArguments, TrainingArguments
from llava.train.llava_trainer import LLaVATrainer

# torch.backends.cudnn.enabled = False


ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = 1000000000
# local_rank = None

# def rank0_print(*args):
#     if local_rank == 0:
#         print(*args)


def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        concat_values = "".join([sentence["value"] for sentence in source])
        for sid, sentence in enumerate(source):
            # In multimodal conversations, we automatically prepend '<image>' at the start of the first sentence if it doesn't already contain one.
            if sid == 0 and DEFAULT_IMAGE_TOKEN not in concat_values:
                sentence["value"] = f"{DEFAULT_IMAGE_TOKEN}\n" + sentence["value"]
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence_chunks = [chunk.strip() for chunk in sentence["value"].split(DEFAULT_IMAGE_TOKEN)]
                sentence_chunks = [
                    chunk + " " if not (chunk.endswith("\n")) else chunk for chunk in sentence_chunks[:-1]
                ] + [sentence_chunks[-1]]
                sentence["value"] = f"{DEFAULT_IMAGE_TOKEN}\n".join(sentence_chunks).strip()

                replace_token = DEFAULT_IMAGE_TOKEN
                if "mmtag" in conversation_lib.default_conversation.version:
                    replace_token = "<Image>" + replace_token + "</Image>"
                if data_args.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    is_mistral: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    # print(f"{conversations[0]}")

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert (
        conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2
        or conv.sep_style == conversation_lib.SeparatorStyle.MISTRAL
    )

    # Mask targets
    if is_mistral:
        sep = "[/INST]"
    else:
        sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}." f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_llama_3(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    no_system_prompt: bool = False,
) -> Dict:
    # Note: implemented by yukang2017@, verified by kentang-mit@
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    if no_system_prompt:
        conv.system = ""

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_3

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                print(f"WARNING: parts!=: {parts}")
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            # include <|eot_id|> for all rounds
            round_len += 1
            instruction_len += 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. {sources}" f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    no_system_prompt: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    if no_system_prompt:
        conv.system = ""
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
                if i > 0 and not is_gemma_tokenizer(tokenizer):
                    round_len = round_len - 1
                    instruction_len = instruction_len - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2
                if i > 0 and not is_gemma_tokenizer(tokenizer):
                    round_len = round_len - 1
                    instruction_len = instruction_len - 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. {sources}" f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    no_system_prompt: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    if no_system_prompt:
        conv.system = ""
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack(
            [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations], dim=0
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target, its in zip(conversations, targets, input_ids):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]  # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx : conv_idx + 2]))  # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            else:
                round_len = len(tokenizer(rou).input_ids) + len(tokenizer(conv.sep).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids)

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. {len(re_rounds)} {sources}" f" (ignored)")

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]["value"] + source[1]["value"] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    no_system_prompt: bool = False,  # only work for v1
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "mpt" or conversation_lib.default_conversation.version == "hermes-2":
        return preprocess_mpt(sources, tokenizer, has_image=has_image, no_system_prompt=no_system_prompt)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.MISTRAL:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image, is_mistral=True)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_3:
        return preprocess_llama_3(sources, tokenizer, has_image=has_image, no_system_prompt=no_system_prompt)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image, no_system_prompt=no_system_prompt)

    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)

    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors="pt") for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


from llava.data.utils import VILAEncodedVideo


class DummyDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and modified by
    Ji Lin and Haotian Tang.
    """

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 image_folder: str,
                 training_args: TrainingArguments):
        super(DummyDataset, self).__init__()
        # list_data_dict = json.load(open(data_path, "r"))
        self.num_dummy_samples = 32768
        import random
        import string

        def generate_random_string(length):
            letters = string.ascii_letters
            result_str = ''.join(random.choice(letters) for _ in range(length))
            return result_str
        self.list_data_dict = []
        for i in range(self.num_dummy_samples):
            question = generate_random_string(32)
            answer = question + generate_random_string(8)
            data_dict = {
                "id": i,
                "image": "empty",
                "conversations": [
                    {
                        "from": "human",
                        "value": question,
                    },
                    {
                        "from": "gpt", 
                        "value": answer,
                    },
                ]
            }
            self.list_data_dict.append(data_dict)

        # rank0_print("Formatting inputs...Skip in lazy mode")
        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            image = process_image(image_file, self.data_args, self.image_folder)
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=(
                "image" in self.list_data_dict[i]
                or "video" in self.list_data_dict[i]
                or "video_id" in self.list_data_dict[i]
            ),
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image.unsqueeze(0)
        else:
            data_dict["image"] = None
        return data_dict

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and modified by
    Ji Lin and Haotian Tang.
    """

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        super(LazySupervisedDataset, self).__init__()
        try:
            with open(data_path, "r") as fp:
                list_data_dict = json.load(fp)
        except:
            with open(data_path, "r") as fp:
                list_data_dict = [json.loads(q) for q in fp]

        # rank0_print("Formatting inputs...Skip in lazy mode")
        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv["value"].split()) for conv in sample["conversations"])
            cur_len = cur_len if "image" in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    

    
    @staticmethod
    def _load_video(video_path, num_video_frames, data_args, fps=None, frame_count=None):
        from llava.mm_utils import opencv_extract_frames
        from torchvision import transforms
        video_loading_succeed = True
        if "shortest_edge" in data_args.image_processor.size:
            image_size = data_args.image_processor.size["shortest_edge"]
        else:
            image_size = data_args.image_processor.size["height"]
        try:
            pil_imgs = opencv_extract_frames(video_path, num_video_frames, fps, frame_count)
        except Exception as e:
            video_loading_succeed = False
            print(f"bad data path {video_path}")
            print(f"[DEBUG] Error processing {video_path}: {e}")
            # video_outputs = torch.zeros(3, 8, image_size, image_size, dtype=torch.uint8)
            pil_imgs = [torch.zeros(3, image_size, image_size, dtype=torch.float32)] * num_video_frames
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * num_video_frames

        return pil_imgs, video_loading_succeed
            


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if isinstance(image_file, list):
                image = torch.stack(
                    [process_image(img, self.data_args, self.image_folder) for img in image_file]
                )
            else:
                image = process_image(image_file, self.data_args, self.image_folder)
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
        elif ("video" in sources[0]) or ("video_id" in sources[0]):
            num_video_frames = self.data_args.num_video_frames
            if "video" in sources[0]:
                video_file = sources[0]["video"]
            else:
                video_file = sources[0]["video_id"] + ".mp4"
            video_folder = self.image_folder
            video_path = os.path.join(video_folder, video_file)
            if 'fps' in sources[0]:
                fps = sources[0]['fps']
            else:
                fps = None
            if 'frame_count' in sources[0]:
                frame_count = sources[0]['frame_count']
            else:
                frame_count = None

            images, video_loading_succeed = self._load_video(video_path, num_video_frames, self.data_args, fps=fps, frame_count=frame_count)

            image_tensor = torch.stack(
                [process_image(image, self.data_args, None) for image in images]
            )

            if "video" in sources[0]:
                question = sources[0]["conversations"][0]["value"].rstrip()
                answer = sources[0]["conversations"][1]["value"].rstrip()
            else:
                question = sources[0]["q"]
                answer = sources[0]["a"]

            if not video_loading_succeed:
                answer = "Empty video."

            question = question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            question = question.replace("<video>\n", "").replace("\n<video>", "").replace("<video>", "")
            question = "<image>\n" * num_video_frames + question
            conversation = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ]

            sources = [conversation]
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        # data_dict = preprocess(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=(
                "image" in self.list_data_dict[i]
                or "video" in self.list_data_dict[i]
                or "video_id" in self.list_data_dict[i]
            ),
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            if len(image.shape) == 4:
                data_dict["image"] = image
            else:
                data_dict["image"] = image.unsqueeze(0)
        elif ("video" in self.list_data_dict[i]) or ("video_id" in self.list_data_dict[i]):
            data_dict["image"] = image_tensor
            if not video_loading_succeed:
                data_dict['labels'][:] = IGNORE_INDEX
        else:
            # llava 1.5 way
            # image does not exist in the data, but the model is multimodal
            # crop_size = self.data_args.image_processor.crop_size
            # data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            # vila way
            data_dict["image"] = None
        return data_dict


class LazyMMC4Dataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ji Lin and Haotian Tang."""

    num_image_tokens = 576

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        image_following_text_only=False,
        text_only=False,
    ):
        super().__init__()

        import pickle

        n_samples = []
        # actually shards and stats info
        n_shards = len(os.listdir(data_path)) // 2
        # n_shards = 100
        count_info_list = sorted([f for f in os.listdir(data_path) if f.endswith(".count")])[:n_shards]
        n_samples = [int(open(os.path.join(data_path, f), "r").read().strip()) for f in count_info_list]

        print("total MMC4 samples", sum(n_samples))  # 10,881,869

        rank = training_args.process_index  # int(os.environ["RANK"])
        world_size = training_args.world_size  # int(os.environ["WORLD_SIZE"])
        shared_size = n_shards // world_size

        gpu_samples = [sum(n_samples[i * shared_size : (i + 1) * shared_size]) for i in range(world_size)]
        self.n_samples = min(gpu_samples) * world_size  # total size
        self.idx_offset = rank * min(gpu_samples)
        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        shard_names = [d.replace(".count", ".pkl") for d in count_info_list]
        shard_names = shard_names[shard_start:shard_end]

        full_data_list = []
        # now load data
        for shard_name in shard_names:
            # load shard
            with open(os.path.join(data_path, shard_name), "rb") as f:
                data_list = pickle.load(f)

            full_data_list.extend(data_list)

        print("* loaded totally {} samples".format(len(full_data_list)))

        self.data_list = full_data_list

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

        self.image_following_text_only = image_following_text_only
        self.text_only = text_only

    def __len__(self):
        # return len(self.data_list)
        return self.n_samples

    @property
    def modality_lengths(self):
        # Estimate the number of tokens after tokenization, used for length-grouped sampling
        length_list = []
        for info in self.data_list:
            num_images = min(6, len(info["image_info"]))
            sentences = [info["text_list"][x["matched_text_index"]] for x in info["image_info"][:num_images]]
            # The unit of cur_len is "words". We assume 1 word = 2 tokens.
            cur_len = num_images * self.num_image_tokens // 2 + sum([len(x) for x in sentences])
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        info = self.data_list[i - self.idx_offset]

        sentences = info["text_list"]
        # kentang-mit@: remove existing <image> tokens in the sentences
        for ix in range(len(sentences)):
            # if this is an html tag, we still preserve its semantic meaning
            sentences[ix] = sentences[ix].replace("<image>", "<IMAGE>")
        sim_matrix = info["similarity_matrix"]  # we do not use this...

        # convert images from base64 to PIL and filter based on image-text similarity
        images, sentence_ixs = [], []
        if not self.text_only:
            for sample_image, sim_vec in zip(info["image_info"], sim_matrix):
                image_base64 = sample_image["image_base64"]
                rawbytes = base64.b64decode(image_base64)

                sim_ix = sample_image["matched_text_index"]
                # sim_ix = np.argmax(sim_vec)
                # sim_score = sim_vec[sim_ix]

                # filter to images >= 5KB
                # if len(rawbytes) // 1000 <= 5:
                #     continue
                # if sim_score < 0.24:
                #     continue
                image = Image.open(io.BytesIO(rawbytes)).convert("RGB")

                images.append(image)
                sentence_ixs.append(sim_ix)

        # constrain max num 6 images
        max_num_images = 6
        if len(images) > max_num_images:
            images = images[:max_num_images]
            sentence_ixs = sentence_ixs[:max_num_images]

        # reorder images according to text insertion
        images = [images[iii] for iii in np.argsort(sentence_ixs)]

        # preprocess and tokenize text
        for ix in sentence_ixs:
            sentences[ix] = f"<image>{sentences[ix]}"

        if self.image_following_text_only:
            # use pad tokens to divide sentence pieces
            text = self.tokenizer.pad_token.join(sentences)
        else:
            text = " ".join(sentences)
        # whitespace cleanup
        text = text.replace("<image> ", "<image>").replace(" <image>", "<image>")
        text = f"{text}{self.tokenizer.eos_token}"  # add eos token

        if len(images) > 0:
            images = torch.stack(
                [process_image(image, self.data_args, self.image_folder) for image in images]
            )

            # the same size for all images, so we concat
            # cur_token_len = (
            #     images[0].shape[-2] // self.multimodal_cfg["patch_size"]
            # ) * (images[0].shape[-1] // self.multimodal_cfg["patch_size"])
            # cur_token_len += self.multimodal_cfg["n_extra_patch"]
        else:
            images = None
            # cur_token_len = 0

        # im_patch_token = self.tokenizer.convert_tokens_to_ids(
        #     [DEFAULT_IMAGE_PATCH_TOKEN]
        # )[0]
        # print(text, len(images))
        input_ids = tokenizer_image_token(
            text,
            self.tokenizer,
            return_tensors="pt",
        )
        assert len(input_ids.shape) == 1

        # now check the case where the last token is image patch token
        if input_ids[-1] == IMAGE_TOKEN_INDEX:  # need to remove one last image
            last_non_im_patch_indices = torch.where(input_ids != IMAGE_TOKEN_INDEX)[0][-1] + 1
            input_ids = input_ids[:last_non_im_patch_indices]

        n_im_patch = (input_ids == IMAGE_TOKEN_INDEX).sum().item()

        images = images[:n_im_patch]
        assert len(images) == n_im_patch, print(text, input_ids)

        targets = input_ids.clone()

        if self.image_following_text_only:  # keep only text after leading image token
            # remove loss for any token before the first <image> token
            label_idx = 0
            while label_idx < targets.shape[-1] and targets[label_idx] != IMAGE_TOKEN_INDEX:
                targets[label_idx] = IGNORE_INDEX
                label_idx += 1

            pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]

            pad_token_idxs = torch.where(targets == pad_token)[0]
            for pad_token_idx in pad_token_idxs:
                token_idx = pad_token_idx + 1
                while token_idx < targets.shape[-1] and targets[token_idx] != IMAGE_TOKEN_INDEX:
                    targets[token_idx] = IGNORE_INDEX
                    token_idx += 1
            # do not train on padding tokens
            targets[targets == pad_token] = IGNORE_INDEX

        # mask image tokens is unnecessary for llava-1.5
        # targets[targets == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        # print(input_ids.shape)

        return dict(input_ids=input_ids, labels=targets, image=images)


class LazyCoyoDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ji Lin and Haotian Tang."""

    num_image_tokens = 576

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        # kentang-mit@: balance the total number of tokens for Coyo and MMC4.
        n_samples_per_idx=4,
    ):
        super().__init__()

        import pickle

        n_samples = []
        # actually shards and stats info
        n_shards = len(os.listdir(data_path)) // 2
        # n_shards = 100
        count_info_list = sorted([f for f in os.listdir(data_path) if f.endswith(".count")])[:n_shards]
        n_samples = [int(open(os.path.join(data_path, f), "r").read().strip()) for f in count_info_list]

        print("total COYO samples", sum(n_samples))

        rank = training_args.process_index  # int(os.environ["RANK"])
        world_size = training_args.world_size  # int(os.environ["WORLD_SIZE"])
        shared_size = n_shards // world_size

        gpu_samples = [
            sum(n_samples[i * shared_size : (i + 1) * shared_size]) // n_samples_per_idx for i in range(world_size)
        ]
        self.n_samples = min(gpu_samples) * world_size  # total size
        self.idx_offset = rank * min(gpu_samples)

        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        shard_names = [d.replace(".count", ".pkl") for d in count_info_list]
        shard_names = shard_names[shard_start:shard_end]

        full_data_list = []
        # now load data
        for shard_name in shard_names:
            # load shard
            with open(os.path.join(data_path, shard_name), "rb") as f:
                shard_data = pickle.load(f)
                random.seed(42)
                if "mmc4" in data_path:
                    random.shuffle(shard_data)  # shuffle for MMC4cap only
                full_data_list.extend(shard_data)

        print("* loaded totally {} samples".format(len(full_data_list)))

        # now pack the samples into groups
        n_groups = len(full_data_list) // n_samples_per_idx
        full_data_list = [
            full_data_list[i : i + n_samples_per_idx] for i in range(0, len(full_data_list), n_samples_per_idx)
        ]
        if len(full_data_list[-1]) < n_samples_per_idx:
            full_data_list = full_data_list[:-1]
        assert len(full_data_list) == n_groups
        print("split into {} groups".format(n_groups))

        self.data_list = full_data_list

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        # return len(self.data_list)
        return self.n_samples

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
        info_list = self.data_list[i - self.idx_offset]

        text_list = []
        image_list = []

        for sample in info_list:
            caption_key = (
                "text" if "text" in sample else "caption"
            )  # kentang-mit@: remove existing <image> tokens in the sentences
            # kentang-mit@: remove existing <image> token.
            # if this is an html tag, we still preserve its semantic meaning
            sample[caption_key] = sample[caption_key].replace("<image>", "<IMAGE>")
            text_list.append(DEFAULT_IMAGE_TOKEN + sample[caption_key] + self.tokenizer.eos_token)
            if "image" in sample:
                image_base64 = sample["image"]
                rawbytes = base64.b64decode(image_base64)
            else:
                rawbytes = sample["rawbytes"]
            image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
            image_list.append(image)

        image_list = torch.stack(
            [process_image(image, self.data_args, self.image_folder) for image in image_list]
        )

        # the same size for all images, so we concat
        # cur_token_len = (
        #     image_list[0].shape[-2] // self.multimodal_cfg["patch_size"]
        # ) * (image_list[0].shape[-1] // self.multimodal_cfg["patch_size"])
        # cur_token_len += self.multimodal_cfg["n_extra_patch"]

        # replace_token = DEFAULT_IMAGE_TOKEN
        # if self.multimodal_cfg["use_im_start_end"]:
        #     replace_token = (
        #         DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
        #     )
        # text_list = [
        #     text.replace(DEFAULT_IMAGE_TOKEN, replace_token) for text in text_list
        # ]

        if CONCAT_SAMPLES:
            # into <image>cap<eos><image>cap<eos>...
            text_list = "".join(text_list)

            input_ids = self.tokenizer(
                text_list,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids  # 4, seq_len

            input_ids = input_ids[0]

        else:
            input_ids = [
                tokenizer_image_token(
                    prompt,
                    self.tokenizer,
                    return_tensors="pt",
                )
                for prompt in text_list
            ]
            # print([x.shape[0] for x in input_ids], [len(x.split()) for x in text_list], [len(re.findall(r"<image[^>]*>", x)) for x in text_list])

            # input_ids = torch.nn.utils.rnn.pad_sequence(
            #     input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            # )

        targets = copy.deepcopy(input_ids)
        # mask image tokens is unnecessary for llava-1.5
        # targets[targets == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        for i in range(len(targets)):
            targets[i][targets[i] == self.tokenizer.pad_token_id] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=targets, image=image_list)


class LazyWDSDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ji Lin and Ligeng Zhu."""

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        image_folder: str,
        training_args: TrainingArguments,
    ):
        super().__init__()
        n_samples = []
        n_shards = len(os.listdir(data_path)) // 3
        for shard in range(n_shards):
            with open(os.path.join(data_path, f"{shard:05d}_stats.json")) as f:
                info = json.load(f)
                n_samples.append(info["successes"])

        print(f"[DEBUG] {data_path} total samples", sum(n_samples))  # 10,881,869
        rank = training_args.process_index  # int(os.environ["RANK"])
        world_size = training_args.world_size  # int(os.environ["WORLD_SIZE"])
        shared_size = n_shards // world_size

        gpu_samples = [sum(n_samples[i * shared_size : (i + 1) * shared_size]) for i in range(world_size)]
        self.n_samples = min(gpu_samples) * world_size  # total size
        self.idx_offset = rank * min(gpu_samples)
        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        tar_list = [f"{shard_idx:05d}.tar" for shard_idx in range(shard_start, shard_end)]

        self.data_list = []
        t1 = time.time()
        for tar in tar_list:
            tmp_path = "/tmp/ccs{}".format(tar)
            tar_path = os.path.join(data_path, tar)

            os.makedirs(tmp_path, exist_ok=True)
            os.system(f"tar -xf {tar_path} -C {tmp_path}")

            txt_list = [f for f in os.listdir(tmp_path) if f.endswith(".txt")]

            for txt in txt_list:
                caption = open(os.path.join(tmp_path, txt), "r").read().strip()
                image_path = os.path.join(tmp_path, txt.split(".")[0] + ".jpg")
                self.data_list.append({"caption": caption, "image": image_path})
        t2 = time.time()
        print("Loading done. Total time: {:.2f} seconds".format(t2 - t1))

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        info = self.data_list[i - self.idx_offset]
        caption, image_path = info["caption"], info["image"]

        rand_prompt = "<image>\n"
        sources = [
            {
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": rand_prompt},
                    {"from": "gpt", "value": caption},
                ],
            }
        ]

        # one example of sources
        # [{'id': 'GCC_train_001738742', 'image': 'GCC_train_001738742.jpg', 'conversations': [{'from': 'human', 'value': 'Provide a brief description of the given image.\n<image>'}, {'from': 'gpt', 'value': 'a sketch of an ostrich'}]}]
        if "image" in sources[0]:
            image = process_image(sources[0]["image"], self.data_args, self.image_folder)
            image = torch.unsqueeze(image, dim=0)
            # now random pick some context samples for training
            if hasattr(self.data_args, "num_shots"):
                if self.data_args.num_shots > 0:
                    raise NotImplementedError
        else:
            raise NotImplementedError

        data_dict = preprocess([sources[0]["conversations"]], self.tokenizer, has_image=True)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if image is not None:
            data_dict["image"] = image
        else:
            raise NotImplementedError

        return data_dict


class LazyVFlanDataset(Dataset):
    """Dataset for supervised fine-tuning from flan mixture.
    This class is implemented by Ji Lin and Haotian Tang."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        super().__init__()
        import pickle

        self.list_data_dict = []

        logging.warning("Loading data...")
        pkl_list = os.listdir(data_path)

        self.sharded = False
        # The original unsharded implementation loads the entire vflan dataset
        # on each GPU. So 80x8=640G host memory per device.
        # If we use the sharded implementation, only 80G per device.
        for pkl in pkl_list:
            if ".count" in pkl:
                self.sharded = True
                break
        if not self.sharded:
            for pkl in pkl_list:
                if pkl.endswith(".pkl"):
                    with open(os.path.join(data_path, pkl), "rb") as f:
                        data = pickle.load(f)
                        self.list_data_dict.extend(data)
            self.n_samples = len(self.list_data_dict)
            logging.warning(f"Loaded {len(self.list_data_dict)} samples...")
        else:
            # kentang-mit@: memory efficient loading of vflan via sharding.
            n_samples = []
            # actually shards and stats info
            n_shards = len(os.listdir(data_path)) // 2
            count_info_list = sorted([f for f in os.listdir(data_path) if f.endswith(".count")])[:n_shards]
            n_samples = [int(open(os.path.join(data_path, f), "r").read().strip()) for f in count_info_list]
            self.n_samples = sum(n_samples)
            print("total VFlan samples", sum(n_samples))  # 10,881,869

            rank = training_args.process_index  # int(os.environ["RANK"])
            world_size = training_args.world_size  # int(os.environ["WORLD_SIZE"])
            shared_size = n_shards // world_size

            gpu_samples = [sum(n_samples[i * shared_size : (i + 1) * shared_size]) for i in range(world_size)]
            self.n_samples = min(gpu_samples) * world_size  # total size
            self.idx_offset = rank * min(gpu_samples)
            shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
            print(f" * loading data from shard {shard_start}-{shard_end}")

            shard_names = [d.replace(".count", ".pkl") for d in count_info_list]
            shard_names = shard_names[shard_start:shard_end]

            full_data_list = []
            # now load data
            for shard_name in shard_names:
                # load shard
                with open(os.path.join(data_path, shard_name), "rb") as f:
                    data_list = pickle.load(f)

                full_data_list.extend(data_list)

            print("* loaded totally {} samples".format(len(full_data_list)))

            self.list_data_dict = full_data_list

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if not self.sharded:
            data = self.list_data_dict[i]
        else:
            data = self.list_data_dict[i - self.idx_offset]
        question = data["question"].rstrip()
        answer = data["answer:" if "answer:" in data else "answer"].rstrip()
        images = data["image:" if "image:" in data else "image"]

        if isinstance(images, str):
            images = [images]
        assert len(images) <= 8, "Too many images in one sample {}".format(len(images))
        if len(images) == 8:  # sample it to be 4
            if hasattr(self.data_args, "downsample_video") and self.data_args.downsample_video:
                images = images[::2]
        n_images = len(images)

        decode_images = []
        for image_str in images:
            if image_str.endswith(".jpg"):
                decode_images.append(image_str)  # a path
            else:  # jpeg bytes
                rawbytes = base64.b64decode(image_str)
                decode_images.append(Image.open(io.BytesIO(rawbytes)).convert("RGB"))

        images = [
            process_image(img, self.data_args, image_folder=self.image_folder)
            for img in decode_images
        ]

        # kentang-mit@: num_shots is not part of data_args. not included now.
        # if self.multimodal_cfg["num_shots"] > 0:
        #     raise NotImplementedError  # do not support multi-shot for FLAN

        # let's make sure there is no <image> in the question...
        if "Image Descriptions" in question:  # NOTE: specicial handlement for generation_visual-dialog_train.pkl
            question_split = question.split("\nQuestion: ")[1:]
            qa_pairs = []
            for qa in question_split:
                qa_pairs.append(qa.split("\nAnswer: "))

            qa_pairs[0][0] = "<image>\n" + qa_pairs[0][0]
            assert len(qa_pairs[-1]) == 1
            qa_pairs[-1][0] = qa_pairs[-1][0].replace("\n", "")
            qa_pairs[-1].append(answer)
            conversation = []
            for q, a in qa_pairs:
                conversation.append({"from": "human", "value": q})
                conversation.append({"from": "gpt", "value": a})
        else:
            question = question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            question = "<image>\n" * n_images + question
            conversation = [
                {"from": "human", "value": question},
                {"from": "gpt", "value": answer},
            ]

        # the same size for all images, so we concat
        if len(images) == 0:
            assert not "<image>" in question

        # sources = replace_image_patch_tokens([conversation], self.multimodal_cfg)
        sources = [conversation]

        # NOTE: here we use the simple version without the system prompt
        # if n_images == 8:
        #     conv_version = "vicuna_v1_1"
        # else:
        #     conv_version = "vicuna_v1_1_nosys"

        # kentang-mit@: the newest conversation template does not have system prompt.
        if hasattr(self.data_args, "vflan_no_system_prompt"):
            no_system_prompt = self.data_args.vflan_no_system_prompt
        else:
            no_system_prompt = False
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=len(images) > 0,
            no_system_prompt=no_system_prompt,
        )

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        if len(images) > 0:
            data_dict["image"] = torch.stack(images)
        else:
            # llava 1.5 way of handling text-only data
            # crop_size = self.data_args.image_processor.crop_size
            # data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            # data_dict['image'] = data_dict['image'].unsqueeze(0)
            # vila way of handling text-only data
            data_dict["image"] = None

        return data_dict


class LazyCCSWebDataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ligeng Zhu."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        super().__init__()
        t1 = time.time()

        from llava.data.simple_vila_webdataset import VILAWebDataset

        print("[DEBUG] ", osp.abspath(data_path))
        self.dataset = VILAWebDataset(
            data_path=osp.abspath(data_path)
        )

        t2 = time.time()
        print("Loading done. Total time: {:.2f} seconds".format(t2 - t1))

        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # info = self.data_list[i - self.idx_offset]
        # caption, image_path = info["caption"], info["image"]
        info = self.dataset[i]
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

        caption = caption.replace("<image>", "<IMAGE>")
        if isinstance(image_path, io.BytesIO):
            image_path = Image.open(image_path).convert("RGB")

        if not isinstance(image_path, PIL.Image.Image):
            print(image_path)
            print(info.keys())
            print(type(image_path))
            raise NotImplementedError

        rand_prompt = "<image>\n"
        sources = [
            {
                "image": image_path,
                "conversations": [
                    {"from": "human", "value": rand_prompt},
                    {"from": "gpt", "value": caption},
                ],
            }
        ]

        # one example of sources
        # [{'id': 'GCC_train_001738742', 'image': 'GCC_train_001738742.jpg', 'conversations': [{'from': 'human', 'value': 'Provide a brief description of the given image.\n<image>'}, {'from': 'gpt', 'value': 'a sketch of an ostrich'}]}]
        if "image" in sources[0]:
            image = process_image(sources[0]["image"], self.data_args, image_folder=None)
            image = torch.unsqueeze(image, dim=0)
            # now random pick some context samples for training
            if hasattr(self.data_args, "num_shots"):
                if self.data_args.num_shots > 0:
                    raise NotImplementedError
        else:
            raise NotImplementedError

        data_dict = preprocess([sources[0]["conversations"]], self.tokenizer, has_image=True)

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if image is not None:
            data_dict["image"] = image
        else:
            raise NotImplementedError

        return data_dict


from functools import lru_cache


@lru_cache(maxsize=16)
def lru_json_load(fpath):
    return json.load(open(fpath, "r"))


class LazyEvaluateDataset(LazySupervisedDataset):
    def __init__(
        self,
        data_path: str,
        data_args: dict,
        tokenizer: PreTrainedTokenizer,
        config_path: str = "llava/eval/mmmu_utils/configs/llava1.5.yaml",
        split="validation",
        **kwargs,
    ):
        # run for each subject
        sub_dataset_list = []
        for subject in CAT_SHORT2LONG.values():
            sub_dataset = load_dataset(data_path, subject, split=split)
            sub_dataset_list.append(sub_dataset)

        all_datasets = concatenate_datasets(sub_dataset_list)
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = None
        self.config = self.get_config(config_path)
        self.list_data_dict = self.get_processed_prompt(all_datasets)

    def get_config(self, config_path: str) -> str:
        config = load_yaml(config_path)
        for key, value in config.items():
            if key != "eval_params" and type(value) == list:
                assert len(value) == 1, "key {} has more than one value".format(key)
                config[key] = value[0]
        return config

    def get_processed_prompt(self, dataset: list) -> list:
        processed_dataset = []
        for d in dataset:
            sample = process_single_sample(d)
            processed_dict = construct_prompt(sample, self.config)

            if '<image>' in processed_dict["gt_content"]:
                processed_dict["gt_content"] = processed_dict["gt_content"].replace('<image>', 'image')
            sample["conversations"] = [
                {"from": "human", "value": processed_dict["final_input_prompt"]},
                {"from": "gpt", "value": processed_dict["gt_content"]},
            ]
            processed_dataset.append(sample)
        return processed_dataset


class LazyCoyoWebDataset(Dataset):
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
        # kentang-mit@: balance the total number of tokens for Coyo and MMC4.
        n_samples_per_idx=4,
    ):
        super().__init__()

        from llava.data.simple_vila_webdataset import VILAWebDataset

        print("[DEBUG] ", osp.abspath(data_path))
        self.dataset = VILAWebDataset(
            data_path=osp.abspath(data_path),
            meta_path=data_args.meta_path
        )

        # None: use original caption
        # Folder path: use original caption
        self.caption_choice = None
        self.data_path = data_path

        print("total samples", len(self.dataset))
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        self.n_samples_per_idx = n_samples_per_idx
        # self.n_samples = len(self.dataset) // n_samples_per_idx
        self.tokenizer = tokenizer
        self.data_args = data_args

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

            if self.caption_choice is not None:
                # load new captions
                shard = info["__shard__"]
                url = info[".json"]["url"]
                tar_name = osp.relpath(osp.realpath(shard), osp.realpath(self.data_path))
                # tar_name = osp.dirname(shard)
                shard_json_path = osp.join(self.caption_choice, tar_name + ".json")
                shard_json = lru_json_load(shard_json_path)
                # print("DEBUG:", shard, self.data_path, tar_name)
                try:
                    caption = shard_json[url]["output"]
                except KeyError:
                    print(f"{url} not in caption. fallback to original caption temporarially")

            caption = caption.replace("<image>", "<IMAGE>")
            text_list.append(DEFAULT_IMAGE_TOKEN + caption + self.tokenizer.eos_token)

            if isinstance(image_path, io.BytesIO):
                image_path = Image.open(image_path).convert("RGB")

            if not isinstance(image_path, PIL.Image.Image):
                print(image_path)
                print(info.keys())
                print(type(image_path))
                raise NotImplementedError

            image_list.append(image_path)

        image_list = torch.stack(
            [process_image(image, self.data_args, image_folder=None) for image in image_list]
        )

        if CONCAT_SAMPLES:
            # into <image>cap<eos><image>cap<eos>...
            text_list = "".join(text_list)

            input_ids = self.tokenizer(
                text_list,
                return_tensors="pt",
                padding="longest",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
            ).input_ids  # 4, seq_len

            input_ids = input_ids[0]
        else:
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


class LazyVideoWebDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        # cache_path: str,
        # n_samples_per_idx=4,
    ):
        super().__init__()

        from llava.data.simple_video_dataset import SimpleVideoDataset

        self.dataset = SimpleVideoDataset(
            data_path=osp.abspath(data_path),
            cache_dir=f"{osp.abspath(data_path)}-webds-meta",
            # cache_dir=cache_path,
        )

        # None: use original caption
        # Folder path: use original caption
        self.caption_chocie = None
        self.data_path = data_path

        print("total samples", len(self.dataset))
        # InternVid: TODO
        rank = int(os.environ["RANK"]) if "RANK" in os.environ else 2
        world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 32

        self.tokenizer = tokenizer
        self.data_args = data_args

    def __len__(self):
        return len(self.dataset)

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
        ADD_TEXT_PROMPT = False
        num_video_frames = self.data_args.num_video_frames

        info = self.dataset[i]
        
        # print(info)
        if ".mp4" in info:
            caption, video_path = info[".txt"], info[".mp4"]
        else:
            video_path = None
            caption = "Empty video."

        images, video_loading_succeed = LazySupervisedDataset._load_video(video_path, num_video_frames, self.data_args)
        
        if not video_loading_succeed:
            caption = "Empty video."

        prompt = "<image>\n" * num_video_frames + caption


        image_tensor = torch.stack(
            [process_image(image, self.data_args, None) for image in images]
        )

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            return_tensors="pt",
        )
        targets = copy.deepcopy(input_ids)
        data_dict = dict(input_ids=input_ids, labels=targets, image=image_tensor)

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and
    modified by Haotian Tang."""

    tokenizer: transformers.PreTrainedTokenizer
    data_args: DataArguments

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # input_ids, labels = tuple([instance[key] for instance in instances]
        #                           for key in ("input_ids", "labels"))
        input_ids, labels, images = [], [], []
        for instance in instances:
            if not isinstance(instance["input_ids"], list):
                input_ids.append(instance["input_ids"])
            else:
                input_ids += instance["input_ids"]
            if not isinstance(instance["labels"], list):
                labels.append(instance["labels"])
            else:
                labels += instance["labels"]
            # Note (kentang-mit@: we do not directly push tensors to
            # images, but list of tensors.
            if instance["image"] is not None:
                cur_image = instance["image"]
                assert len(cur_image.shape) == 4
                # n_images, 3, size, size
                if not isinstance(instance["input_ids"], list):
                    # datasets other than coyo, not packing >1 samples together
                    images.append(cur_image)
                else:
                    # coyo-like datasets
                    images.extend(cur_image.chunk(cur_image.size(0), 0))
            else:
                images.append([])
        # kentang-mit@: we need to make sure these two lists have
        # the same length. We will use input_ids to filter out images corresponding
        # to truncated <image> tokens later.
        for _images, _input_ids in zip(images, input_ids):
            assert (
                len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()
            ), f"Number mismatch between images and placeholder image tokens in 'len(_images) == (_input_ids == IMAGE_TOKEN_INDEX).sum().item()'.\
                Expect to have {len(_images)} images but only found {(_input_ids == IMAGE_TOKEN_INDEX).sum().item()} images in tokens. \
                Error input_ids: {_input_ids} {self.tokenizer.decode([x if x != -200 else 200 for x in _input_ids])}"

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        new_images = []
        # kentang-mit@: it is possible that some <image> tokens get removed
        # after truncation. It is important to also remove corresponding images.
        # otherwise, text and image will mismatch in the model.
        for ix in range(len(input_ids)):
            num_images = (input_ids[ix] == IMAGE_TOKEN_INDEX).sum().item()
            cur_images = images[ix]
            cur_images = cur_images[:num_images]
            if len(cur_images) > 0:
                new_images.append(cur_images)
        if len(new_images) > 0:
            batch["images"] = torch.cat(new_images, dim=0)
        else:
            # the entire batch is text-only
            if hasattr(self.data_args.image_processor, "crop_size"):
                crop_size = self.data_args.image_processor.crop_size
            else:
                crop_size = self.data_args.image_processor.size
            # we still need 1 dummy image for the vision tower
            batch["images"] = torch.zeros(1, 3, crop_size["height"], crop_size["width"])

        return batch


def make_supervised_data_module(
    tokenizer: PreTrainedTokenizer,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning.
    This function is originally implemented by the LLaVA team and
    modified by Jason Lu, Haotian Tang and Ligeng Zhu."""
    datasets_mixture.register_datasets_mixtures()
    train_dataset = build_datasets(data_args, training_args=training_args, tokenizer=tokenizer, split="train")
    eval_dataset = build_datasets(data_args, training_args=training_args, tokenizer=tokenizer, split="eval")
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def build_datasets(
    data_args: DataArguments,
    training_args: TrainingArguments,
    tokenizer: PreTrainedTokenizer,
    split: str = "train",
) -> None:
    all_datasets = []
    extra_info = []
    # mixture = datasets_mixture.DATASETS_MIXTURES[data_args.data_mixture]
    try:
        ## keep the name 'data_mixture' for development FIXME
        # mixture_names = getattr(data_args, f"{split}_data_mixture").strip().split("+")
        attr_name = "data_mixture" if split == "train" else "eval_data_mixture"
        mixture_names = getattr(data_args, attr_name).strip().split("+")
    except:
        logging.warning(f"Pay attention, split {split} is not built...")
        return None
    mixture = (DATASETS[_] for _ in mixture_names)
    print(f"[Dataset-INFO]: Loading from {mixture_names}")
    image_folder = None
    for dataset in mixture:
        dataset_type = dataset.dataset_type
        if dataset_type == "torch":
            dataset_cls = LazySupervisedDataset
            if hasattr(dataset, "image_path"):
                image_folder = dataset.image_path
        elif dataset_type == "wds":
            print(f"[DEBUG] {dataset_type}")
            dataset_cls = LazyWDSDataset
        elif dataset_type == "mmc4":
            dataset_cls = LazyMMC4Dataset
        elif dataset_type == "coyo":
            dataset_cls = LazyCoyoDataset
        elif dataset_type == "sam-wds":
            print("dataset.py: Loading SAM class")
            from llava.data.dataset_impl.sam import LazySAMWebDataset

            dataset_cls = LazySAMWebDataset
        elif dataset_type == "coyo-wds":
            dataset_cls = LazyCoyoWebDataset
        elif dataset_type == "coyo-wds-recap":
            print("dataset.py: Loading coyo-wds-recap class")
            from llava.data.dataset_impl.coyo_recap import \
                LazyCoyoWebRecapDataset

            dataset_cls = LazyCoyoWebRecapDataset
        elif dataset_type == "textocr":
            print("dataset.py: Loading textocr class")
            from llava.data.dataset_impl.textocr import VILATextOCR

            dataset_cls = VILATextOCR
        elif dataset_type == "hiertext":
            print("dataset.py: Loading hiertext class")
            from llava.data.dataset_impl.hiertext import VILAHierText

            dataset_cls = VILAHierText
        elif dataset_type == "panda70m":
            print("dataset.py: Loading VILAPanda70m class")
            from llava.data.dataset_impl.panda70m import VILAPanda70m

            dataset_cls = VILAPanda70m
        elif dataset_type == "ccs-wds":
            dataset_cls = LazyCCSWebDataset
        elif dataset_type == "vflan":
            dataset_cls = LazyVFlanDataset
        elif dataset_type == "video-wds":
            dataset_cls = LazyVideoWebDataset
        elif dataset_type == "evaluation":
            dataset_cls = LazyEvaluateDataset
        elif dataset_type == "dummy":
            dataset_cls = DummyDataset
            if hasattr(dataset, "image_path"):
                image_folder = dataset.image_path
        else:
            raise NotImplementedError(f"{dataset_type} is not supported.")
        data_args.meta_path = getattr(dataset, "meta_path", None)
        dataset = dataset_cls(
            tokenizer=tokenizer,
            data_path=dataset.data_path,
            image_folder=image_folder,
            data_args=data_args,
            training_args=training_args,
        )
        all_datasets.append(dataset)
        extra_info.append(len(dataset))

    all_datasets = ConcatDataset(all_datasets)
    if split == "train":
        training_args.sample_lens = extra_info
    elif split == "eval":
        training_args.eval_sample_lens = extra_info
    return all_datasets
