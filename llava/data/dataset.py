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
import math
import os
import os.path as osp
import random
import time
import warnings
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import PIL
import torch
import transformers
from PIL import Image, ImageFile
from torch.utils.data import Dataset, default_collate
from transformers import PreTrainedTokenizer

import llava.data.datasets_mixture as datasets_mixture
from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX
from llava.data.collate import DataCollator
from llava.mm_utils import (
    delete_extra_img_tokens,
    dynamic_process_images_and_prompt,
    dynamic_s2_process_images_and_prompt,
    get_original_image_size,
    opencv_extract_frames,
    process_image,
    tokenizer_image_token,
)
from llava.train.args import DataArguments, TrainingArguments
from llava.train.sequence_parallel import (
    extract_local_from_list,
    extract_local_input_ids,
    extract_local_position_ids,
    get_pg_manager,
)
from llava.utils.tokenizer import preprocess_conversation

ImageFile.LOAD_TRUNCATED_IMAGES = True
PIL.Image.MAX_IMAGE_PIXELS = 1000000000


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
            # ensure every DEFAULT_IMAGE_TOKEN is followed by a newline character.
            # If it has one already, we don't add another one.
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, f"{DEFAULT_IMAGE_TOKEN}\n")
                sentence["value"] = sentence["value"].replace(f"{DEFAULT_IMAGE_TOKEN}\n\n", f"{DEFAULT_IMAGE_TOKEN}\n")

    return sources


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
    no_system_prompt: bool = False,
) -> Dict:
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    return default_collate(
        [
            preprocess_conversation(conversation, tokenizer, no_system_prompt=no_system_prompt)
            for conversation in sources
        ]
    )


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
        super().__init__()
        try:
            with open(data_path) as fp:
                list_data_dict = json.load(fp)
        except:
            with open(data_path) as fp:
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
    def _load_video(video_path, num_video_frames, loader_fps, data_args, fps=None, frame_count=None):
        try:
            pil_imgs, frames_loaded = opencv_extract_frames(video_path, num_video_frames, loader_fps, fps, frame_count)
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")
            empty_num_video_frames = int(random.uniform(2, num_video_frames))
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * empty_num_video_frames
            frames_loaded = 0

        return pil_imgs, frames_loaded

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        enable_dynamic_res = self.data_args.image_aspect_ratio == "dynamic"
        enable_dynamic_res_s2 = self.data_args.image_aspect_ratio == "dynamic_s2"
        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]

            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

            # If we want to control the maximum number of images in each instance for efficient training
            if self.data_args.max_num_images is not None:
                sources = delete_extra_img_tokens(sources, self.data_args.max_num_images)
                if isinstance(image_file, list):
                    image_file = image_file[: min(self.data_args.max_num_images, len(image_file))]

            if isinstance(image_file, list):
                if enable_dynamic_res_s2:
                    processed_images, block_sizes = dynamic_s2_process_images_and_prompt(
                        image_file, sources[0][0]["value"], self.data_args, self.image_folder
                    )
                elif enable_dynamic_res:
                    processed_images, sources[0][0]["value"] = dynamic_process_images_and_prompt(
                        image_file, sources[0][0]["value"], self.data_args, self.image_folder
                    )
                else:
                    processed_images = torch.stack(
                        [process_image(img, self.data_args, self.image_folder) for img in image_file]
                    )
                original_image_sizes = [get_original_image_size(img, self.image_folder) for img in image_file]
            else:
                if enable_dynamic_res_s2:
                    processed_images, block_sizes = dynamic_s2_process_images_and_prompt(
                        [image_file], sources[0][0]["value"], self.data_args, self.image_folder
                    )
                elif enable_dynamic_res:
                    processed_images, sources[0][0]["value"] = dynamic_process_images_and_prompt(
                        [image_file], sources[0][0]["value"], self.data_args, self.image_folder
                    )
                else:
                    processed_images = process_image(
                        image_file, self.data_args, self.image_folder, enable_dynamic_res=enable_dynamic_res
                    )
                original_image_sizes = [get_original_image_size(image_file, self.image_folder)]

        elif "images" in sources[0]:
            sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)
            image_files = [
                image_file["path"] if isinstance(image_file, dict) else image_file
                for image_file in self.list_data_dict[i]["images"]
            ]

            # If we want to control the maximum number of images in each instance for efficient training
            if self.data_args.max_num_images is not None:
                sources = delete_extra_img_tokens(sources, self.data_args.max_num_images)
                if isinstance(image_file, list):
                    image_file = image_file[: min(self.data_args.max_num_images, len(image_file))]

            if enable_dynamic_res_s2:
                processed_images, block_sizes = dynamic_s2_process_images_and_prompt(
                    image_files, sources[0][0]["value"], self.data_args, self.image_folder
                )
            elif enable_dynamic_res:
                processed_images, sources[0][0]["value"] = dynamic_process_images_and_prompt(
                    image_files, sources[0][0]["value"], self.data_args, self.image_folder
                )
            else:
                all_images = []
                for image_file in self.list_data_dict[i]["images"]:
                    if isinstance(image_file, dict):
                        image_file = image_file["path"]
                    image = process_image(image_file, self.data_args, self.image_folder)
                    all_images.append(image)
                processed_images = torch.stack(all_images)
            original_image_sizes = [get_original_image_size(img, self.image_folder) for img in image_file]

        elif ("video" in sources[0]) or ("video_id" in sources[0]):
            # num_video_frames = self.data_args.num_video_frames
            if "video_path" in sources[0]:
                video_file = sources[0]["video_path"]
            elif "video" in sources[0]:
                video_file = sources[0]["video"]
            else:
                video_file = sources[0]["video_id"] + ".mp4"
            video_folder = self.image_folder
            video_path = os.path.join(video_folder, video_file)
            num_video_frames = self.data_args.num_video_frames if hasattr(self.data_args, "num_video_frames") else 8
            loader_fps = self.data_args.fps if hasattr(self.data_args, "fps") else 0.0

            if "fps" in sources[0]:
                fps = sources[0]["fps"]
            else:
                fps = None
            if "frame_count" in sources[0]:
                frame_count = sources[0]["frame_count"]
            else:
                frame_count = None

            images, frames_loaded = self._load_video(
                video_path, num_video_frames, loader_fps, self.data_args, fps=fps, frame_count=frame_count
            )

            if enable_dynamic_res_s2 and self.data_args.video_max_tiles > 1:
                image_tensor, block_sizes = dynamic_s2_process_images_and_prompt(
                    images,
                    sources[0]["conversations"][0]["value"],
                    self.data_args,
                    self.image_folder,
                    max_tiles=self.data_args.video_max_tiles,
                )
            elif enable_dynamic_res and self.data_args.video_max_tiles > 1:
                image_tensor, sources[0]["conversations"][0]["value"] = dynamic_process_images_and_prompt(
                    images,
                    sources[0]["conversations"][0]["value"],
                    self.data_args,
                    self.image_folder,
                    max_tiles=self.data_args.video_max_tiles,
                )
            else:
                image_tensor = torch.stack([process_image(image, self.data_args, None) for image in images])

            if "captions" in sources[0]:
                question = "Elaborate on the visual and narrative elements of the video in detail."
                assert sources[0]["captions"][-1]["idx"] == "-1"
                answer = sources[0]["captions"][-1]["content"]
            elif "video" in sources[0]:
                question = sources[0]["conversations"][0]["value"].rstrip()
                if isinstance(sources[0]["conversations"][1]["value"], str):
                    answer = sources[0]["conversations"][1]["value"].rstrip()
                else:
                    answer = str(sources[0]["conversations"][1]["value"]).rstrip()
            else:
                question = sources[0]["q"]
                answer = sources[0]["a"]

            if frames_loaded == 0:
                answer = "Empty video."
            num_frames_loaded_successfully = len(image_tensor)

            question = question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            question = question.replace("<video>\n", "").replace("\n<video>", "").replace("<video>", "")
            question = "<image>\n" * num_frames_loaded_successfully + question
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
                or "images" in self.list_data_dict[i]
                or "video" in self.list_data_dict[i]
                or "video_id" in self.list_data_dict[i]
            ),
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            if processed_images is None or len(processed_images.shape) == 4:
                data_dict["image"] = processed_images
            else:
                data_dict["image"] = processed_images.unsqueeze(0)
            data_dict["original_image_sizes"] = original_image_sizes
            if enable_dynamic_res_s2:
                data_dict["block_sizes"] = block_sizes
        elif "images" in self.list_data_dict[i]:
            data_dict["image"] = processed_images
            data_dict["original_image_sizes"] = original_image_sizes
            if enable_dynamic_res_s2:
                data_dict["block_sizes"] = block_sizes
        elif ("video" in self.list_data_dict[i]) or ("video_id" in self.list_data_dict[i]):
            data_dict["image"] = image_tensor
            if frames_loaded == 0:
                data_dict["labels"][:] = IGNORE_INDEX
        else:
            # llava 1.5 way
            # image does not exist in the data, but the model is multimodal
            # crop_size = self.data_args.image_processor.crop_size
            # data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            # vila way
            data_dict["image"] = None
        return data_dict


class LazyQAGroundingDataset(Dataset):
    """Dataset for QA Grounding."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        global_batch_size: int = None,
    ):
        super().__init__()

        with open(data_path) as fp:
            list_data_dict = json.load(fp)

        # rank0_print("Formatting inputs...Skip in lazy mode")
        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_folder = image_folder

        if global_batch_size is not None:
            residual = global_batch_size - len(self.list_data_dict) % global_batch_size
            if residual != global_batch_size:
                if global_batch_size // len(self.list_data_dict) >= 2:
                    self.list_data_dict = self.list_data_dict * (global_batch_size // len(self.list_data_dict))
                    residual = global_batch_size - len(self.list_data_dict) % global_batch_size
                selected_elements = random.sample(range(len(self.list_data_dict)), residual)
                additional_instance = [self.list_data_dict[i] for i in selected_elements]
                self.list_data_dict.extend(additional_instance)

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
        assert "image" in sources[0]

        image_file = self.list_data_dict[i]["image"]
        box = self.list_data_dict[i]["box"]
        image, box = process_image(image_file, self.data_args, self.image_folder, box=box)
        original_image_sizes = [get_original_image_size(image_file, self.image_folder)]
        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        # data_dict = preprocess(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True,
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        data_dict["image"] = image.unsqueeze(0)
        data_dict["original_image_sizes"] = original_image_sizes

        # process box into ground-truth selection map
        gt_selection_map = torch.zeros(image.shape[-2:], dtype=image.dtype, device=image.device)
        gt_selection_map[math.floor(box[1]) : math.ceil(box[3]) + 1, math.floor(box[0]) : math.ceil(box[2]) + 1] = 1
        data_dict["gt_selection_map"] = gt_selection_map

        # set labels to all ignore because we don't want to train next token prediction on this data
        data_dict["labels"][:] = IGNORE_INDEX

        return data_dict


class LazyHighResQADataset(Dataset):
    """Dataset for QA Grounding."""

    def __init__(
        self,
        data_path: str,
        image_folder: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
    ):
        super().__init__()
        try:
            with open(data_path) as fp:
                list_data_dict = json.load(fp)
        except:
            with open(data_path) as fp:
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

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        assert "image" in sources[0]

        image_file = self.list_data_dict[i]["image"]
        box = self.list_data_dict[i]["box"]
        image, box = process_image(image_file, self.data_args, self.image_folder, box=box)
        original_image_sizes = [get_original_image_size(image_file, self.image_folder)]
        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]), self.data_args)

        # data_dict = preprocess(sources, self.tokenizer, has_image=("image" in self.list_data_dict[i]))
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=True,
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])

        # image exist in the data
        data_dict["image"] = image.unsqueeze(0)
        data_dict["original_image_sizes"] = original_image_sizes

        # process box into ground-truth selection map
        gt_selection_map = torch.zeros(image.shape[-2:], dtype=image.dtype, device=image.device)
        gt_selection_map[math.floor(box[1]) : math.ceil(box[3]) + 1, math.floor(box[0]) : math.ceil(box[2]) + 1] = 1
        data_dict["gt_selection_map"] = gt_selection_map

        return data_dict


class LazyMMC4Dataset(Dataset):
    """Dataset for supervised fine-tuning.
    This class is implemented by Ji Lin and Haotian Tang."""

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
        n_samples = [int(open(os.path.join(data_path, f)).read().strip()) for f in count_info_list]

        print("total MMC4 samples", sum(n_samples))  # 10,881,869

        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = training_args.process_index // sequence_parallel_size  # int(os.environ["RANK"])
        world_size = training_args.world_size // sequence_parallel_size  # int(os.environ["WORLD_SIZE"])
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

        print(f"* loaded totally {len(full_data_list)} samples")

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
        if self.data_args.max_num_images is not None:
            max_num_images = min(max_num_images, self.data_args.max_num_images)
        if len(images) > max_num_images:
            images = images[:max_num_images]
            sentence_ixs = sentence_ixs[:max_num_images]

        # reorder images according to text insertion
        images = [images[iii] for iii in np.argsort(sentence_ixs)]

        # preprocess and tokenize text
        for ix in sentence_ixs:
            sentences[ix] = f"<image>\n{sentences[ix]}"

        if self.image_following_text_only:
            # use pad tokens to divide sentence pieces
            text = self.tokenizer.pad_token.join(sentences)
        else:
            text = " ".join(sentences)
        # whitespace cleanup
        text = text.replace("<image> ", "<image>").replace(" <image>", "<image>")
        text = f"{text}{self.tokenizer.eos_token}"  # add eos token

        if len(images) > 0:
            original_image_sizes = [get_original_image_size(image, self.image_folder) for image in images]
            if self.data_args.image_aspect_ratio == "dynamic_s2":
                images, block_sizes = dynamic_s2_process_images_and_prompt(
                    images, text, self.data_args, self.image_folder
                )
            elif self.data_args.image_aspect_ratio == "dynamic":
                images, text = dynamic_process_images_and_prompt(
                    images, text, self.data_args, self.image_folder, max_tiles=6
                )
            else:
                images = torch.stack([process_image(image, self.data_args, self.image_folder) for image in images])

            # the same size for all images, so we concat
            # cur_token_len = (
            #     images[0].shape[-2] // self.multimodal_cfg["patch_size"]
            # ) * (images[0].shape[-1] // self.multimodal_cfg["patch_size"])
            # cur_token_len += self.multimodal_cfg["n_extra_patch"]
        else:
            images = None
            # cur_token_len = 0

        input_ids = tokenizer_image_token(
            text,
            self.tokenizer,
            return_tensors="pt",
        )

        image_token_id = self.tokenizer.media_token_ids["image"]

        # now check the case where the last token is image patch token
        if input_ids[-1] == image_token_id:  # need to remove one last image
            last_non_im_patch_indices = torch.where(input_ids != image_token_id)[0][-1] + 1
            input_ids = input_ids[:last_non_im_patch_indices]

        n_im_patch = (input_ids == image_token_id).sum().item()

        if self.data_args.image_aspect_ratio != "dynamic_s2":
            images = images[:n_im_patch]
            original_image_sizes = original_image_sizes[:n_im_patch]
            assert len(images) == n_im_patch, print(text, input_ids)
        assert len(input_ids.shape) == 1, "Unexpected shape of 'input_ids' from MMC4."
        input_ids = (
            torch.concat([torch.tensor([self.tokenizer.bos_token_id]), input_ids])
            if self.tokenizer.bos_token_id is not None and input_ids[0] != self.tokenizer.bos_token_id
            else input_ids
        )
        targets = input_ids.clone()

        if self.image_following_text_only:  # keep only text after leading image token
            # remove loss for any token before the first <image> token
            label_idx = 0
            while label_idx < targets.shape[-1] and targets[label_idx] != image_token_id:
                targets[label_idx] = IGNORE_INDEX
                label_idx += 1

            pad_token = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_token])[0]

            pad_token_idxs = torch.where(targets == pad_token)[0]
            for pad_token_idx in pad_token_idxs:
                token_idx = pad_token_idx + 1
                while token_idx < targets.shape[-1] and targets[token_idx] != image_token_id:
                    targets[token_idx] = IGNORE_INDEX
                    token_idx += 1
            # do not train on padding tokens
            targets[targets == pad_token] = IGNORE_INDEX

        # mask image tokens is unnecessary for llava-1.5
        # targets[targets == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        # print(input_ids.shape)

        data_dict = dict(input_ids=input_ids, labels=targets, image=images, original_image_sizes=original_image_sizes)
        if self.data_args.image_aspect_ratio == "dynamic_s2":
            data_dict["block_sizes"] = block_sizes

        return data_dict


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
        n_samples = [int(open(os.path.join(data_path, f)).read().strip()) for f in count_info_list]

        print("total COYO samples", sum(n_samples))

        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = training_args.process_index // sequence_parallel_size  # int(os.environ["RANK"])
        world_size = training_args.world_size // sequence_parallel_size  # int(os.environ["WORLD_SIZE"])
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

        print(f"* loaded totally {len(full_data_list)} samples")

        # now pack the samples into groups
        n_groups = len(full_data_list) // n_samples_per_idx
        full_data_list = [
            full_data_list[i : i + n_samples_per_idx] for i in range(0, len(full_data_list), n_samples_per_idx)
        ]
        if len(full_data_list[-1]) < n_samples_per_idx:
            full_data_list = full_data_list[:-1]
        assert len(full_data_list) == n_groups
        print(f"split into {n_groups} groups")

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
            text_list.append(DEFAULT_IMAGE_TOKEN + "\n" + sample[caption_key] + self.tokenizer.eos_token)
            if "image" in sample:
                image_base64 = sample["image"]
                rawbytes = base64.b64decode(image_base64)
            else:
                rawbytes = sample["rawbytes"]
            image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
            image_list.append(image)

        image_list = torch.stack([process_image(image, self.data_args, self.image_folder) for image in image_list])
        original_image_sizes = [get_original_image_size(image, self.image_folder) for image in image_list]

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
        for i in range(len(targets)):
            targets[i][targets[i] == self.tokenizer.pad_token_id] = IGNORE_INDEX

        return dict(input_ids=input_ids, labels=targets, image=image_list, original_image_sizes=original_image_sizes)


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

        # print(f"[DEBUG] {data_path} total samples", sum(n_samples))  # 10,881,869

        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = training_args.process_index // sequence_parallel_size  # int(os.environ["RANK"])
        world_size = training_args.world_size // sequence_parallel_size  # int(os.environ["WORLD_SIZE"])
        shared_size = n_shards // world_size
        print("rank", rank, "world_size", world_size, "shared_size", shared_size)
        gpu_samples = [sum(n_samples[i * shared_size : (i + 1) * shared_size]) for i in range(world_size)]
        self.n_samples = min(gpu_samples) * world_size  # total size
        self.idx_offset = rank * min(gpu_samples)
        shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        print(f" * loading data from shard {shard_start}-{shard_end}")

        tar_list = [f"{shard_idx:05d}.tar" for shard_idx in range(shard_start, shard_end)]

        self.data_list = []
        t1 = time.time()
        for tar in tar_list:
            tmp_path = f"/tmp/ccs{tar}"
            tar_path = os.path.join(data_path, tar)

            if PROCESS_GROUP_MANAGER is not None:
                dist.barrier()
                if PROCESS_GROUP_MANAGER.sp_rank == 0:
                    os.makedirs(tmp_path, exist_ok=True)
                    os.system(f"tar -xkf {tar_path} -C {tmp_path}")
                dist.barrier()
            else:
                os.makedirs(tmp_path, exist_ok=True)
                os.system(f"tar -xkf {tar_path} -C {tmp_path}")

            txt_list = [f for f in os.listdir(tmp_path) if f.endswith(".txt")]

            for txt in txt_list:
                caption = open(os.path.join(tmp_path, txt)).read().strip()
                image_path = os.path.join(tmp_path, txt.split(".")[0] + ".jpg")
                self.data_list.append({"caption": caption, "image": image_path})
        t2 = time.time()
        print(f"Loading done. Total time: {t2 - t1:.2f} seconds")

        self.tokenizer = tokenizer
        self.data_args = data_args
        self.image_folder = image_folder

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:

        # print("i", i, "idx_offset", self.idx_offset, "len", len(self.data_list))
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
        self.dataset = VILAWebDataset(data_path=osp.abspath(data_path))

        t2 = time.time()
        print(f"Loading done. Total time: {t2 - t1:.2f} seconds")

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
    with open(fpath) as fp:
        return json.load(fp)


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
        self.dataset = VILAWebDataset(data_path=osp.abspath(data_path), meta_path=data_args.meta_path)

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
        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
            sequence_parallel_rank = PROCESS_GROUP_MANAGER.sp_rank
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = (
            training_args.process_index // sequence_parallel_size if "RANK" in os.environ else 2
        )  # int(os.environ["RANK"])
        world_size = (
            training_args.world_size // sequence_parallel_size if "WORLD_SIZE" in os.environ else 32
        )  # int(os.environ["WORLD_SIZE"])
        print(
            "rank",
            rank,
            "world_size",
            world_size,
        )

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
                try:
                    shard_json = lru_json_load(shard_json_path)
                    try:
                        caption = shard_json[url]["output"]
                    except KeyError:
                        print(f"{url} not in caption. fallback to original caption temporarially")
                except:
                    print(f"shard_json_path {shard_json_path} not found. fallback to original caption temporarially")
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

        # image_list = torch.stack([process_image(image, self.data_args, image_folder=None) for image in image_list])
        # NOTE(fix by ligeng)
        #  now image_list should return a list of image tensor where each has a dimension of (1, c, h, w)
        image_list = [process_image(image, self.data_args, image_folder=None).unsqueeze(0) for image in image_list]

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
            input_ids = [
                (
                    torch.concat([torch.tensor([self.tokenizer.bos_token_id]), input_ids_i])
                    if input_ids_i[0] != self.tokenizer.bos_token_id
                    else input_ids_i
                )
                for input_ids_i in input_ids
            ]

        targets = copy.deepcopy(input_ids)
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

        # from llava.data.simple_video_dataset import SimpleVideoDataset

        from llava.data.simple_vila_webdataset import VILAWebDataset

        print("[DEBUG] ", osp.abspath(data_path))
        self.dataset = VILAWebDataset(
            data_path=osp.abspath(data_path),
            meta_path=f"{osp.abspath(data_path)}/wids-meta.json",
            # cache_dir=cache_path,
        )

        # None: use original caption
        # Folder path: use original caption
        self.caption_choice = None
        self.data_path = data_path

        if data_args.caption_choice is not None:
            self.caption_choice = data_args.caption_choice
            print("[recap] Override LazyVideo caption using ", self.caption_choice)

        print("total samples", len(self.dataset))
        # InternVid: TODO
        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            import torch.distributed as dist

            sequence_parallel_size = training_args.seq_parallel_size
            sequence_parallel_rank = PROCESS_GROUP_MANAGER.sp_rank
        else:
            sequence_parallel_size = 1
        print("sequence_parallel_size", sequence_parallel_size)
        rank = (
            training_args.process_index // sequence_parallel_size if "RANK" in os.environ else 2
        )  # int(os.environ["RANK"])
        world_size = (
            training_args.world_size // sequence_parallel_size if "WORLD_SIZE" in os.environ else 32
        )  # int(os.environ["WORLD_SIZE"])
        print(
            "rank",
            rank,
            "world_size",
            world_size,
        )
        self.rank = rank
        # rank = int(os.environ["RANK"]) if "RANK" in os.environ else 2
        # world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 32

        self.tokenizer = tokenizer
        self.data_args = data_args

        self.missing_uids = set()

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
        num_video_frames = self.data_args.num_video_frames if hasattr(self.data_args, "num_video_frames") else 8
        loader_fps = self.data_args.fps if hasattr(self.data_args, "fps") else 0.0

        info = self.dataset[i]

        caption = ""
        # print(info)
        if ".mp4" in info:
            caption, video_path = info[".txt"], info[".mp4"]
        else:
            video_path = None
            caption = "Empty video."

        images, frames_loaded = LazySupervisedDataset._load_video(
            video_path, num_video_frames, loader_fps, self.data_args
        )

        if frames_loaded == 0:
            caption = "Empty video."

        if self.caption_choice is not None:
            shard = info["__shard__"]
            uuid = osp.join(info["__shard__"], info["__key__"])
            url = info["__key__"]
            tar_name = osp.basename(info["__shard__"])

            try:
                shard_json_path = osp.join(self.caption_choice, tar_name.replace(".tar", ".json"))
                shard_json = lru_json_load(shard_json_path)
                caption = shard_json[url]["summary"]["output"]
            except (KeyError, FileNotFoundError, json.decoder.JSONDecodeError):
                if uuid not in self.missing_uids:
                    print("override caption not found for ", uuid)
                    self.missing_uids.add(uuid)

            # print(f"[DEBUG {uuid}]", caption)

        frames_loaded_successfully = len(images)
        if caption is None:
            caption = ""
        prompt = "<image>\n" * frames_loaded_successfully + caption
        image_tensor = torch.stack([process_image(image, self.data_args, None) for image in images])

        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            return_tensors="pt",
        )
        targets = copy.deepcopy(input_ids)
        data_dict = dict(input_ids=input_ids, labels=targets, image=image_tensor)

        return data_dict


class DataCollatorForSupervisedDatasetSeqParallel:
    """Collate examples for supervised fine-tuning.
    This class is originally implemented by the LLaVA team and
    modified by Haotian Tang."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        training_args: TrainingArguments,
        sp_degree: int,
        sp_rank: int,
        ring_degree: int,
        ring_type: str,
    ):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.training_args = training_args
        self.sp_degree = sp_degree
        self.sp_rank = sp_rank
        self.ring_degree = ring_degree
        self.ring_type = ring_type

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, images = [], [], []
        image_token_id = self.tokenizer.media_token_ids["image"]
        video_token_id = self.tokenizer.media_token_ids["video"]

        pool_size = 1
        try:
            pool_size = self.data_args.pool_size
        except:
            pass

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

            if "video" in instance:
                instance["image"] = torch.cat(instance["video"])
                num_images = instance["image"].shape[0] // (pool_size * self.sp_degree) * pool_size * self.sp_degree
                instance["image"] = instance["image"][:num_images]
                video_id_pos = torch.where(input_ids[-1] == video_token_id)[0][0]
                replace_ids = torch.Tensor(
                    ([image_token_id] + self.tokenizer.encode("\n")) * num_images,
                    device=input_ids[-1].device,
                )
                input_ids[-1] = torch.cat(
                    [input_ids[-1][:video_id_pos], replace_ids, input_ids[-1][video_id_pos + 1 :]]
                ).to(input_ids[-1].dtype)
                labels[-1] = torch.cat(
                    [
                        labels[-1][:video_id_pos],
                        torch.Tensor([IGNORE_INDEX] * num_images * 2),
                        labels[-1][video_id_pos + 1 :],
                    ]
                ).to(labels[-1].dtype)
                instance.pop("video")

            if "image" in instance:
                cur_image = instance["image"]
                assert len(cur_image.shape) == 4
                # n_images, 3, size, size
                if cur_image.shape[0] == 0:
                    warnings.warn("loaded one sample without images.")
                if not isinstance(instance["input_ids"], list):
                    # datasets other than coyo, not packing >1 samples together
                    images.append(cur_image)
                else:
                    # coyo-like datasets
                    images.extend(cur_image.chunk(cur_image.size(0), 0))
            else:
                warnings.warn("loaded one sample without images.")
                images.append([])
        # kentang-mit@: we need to make sure these two lists have
        # the same length. We will use input_ids to filter out images corresponding
        # to truncated <image> tokens later.

        max_num_images = max([len(_images) for _images in images])
        for _images, _input_ids in zip(images, input_ids):
            assert (
                len(_images) == (_input_ids == image_token_id).sum().item()
            ), f"Number mismatch between images and placeholder image tokens in 'len(_images) == (_input_ids == image_token_id).sum().item()'.\
                Expect to have {len(_images)} images but only found {(_input_ids == image_token_id).sum().item()} images in tokens. \
                Error input_ids: {_input_ids} {self.tokenizer.decode([x if x != -200 else 200 for x in _input_ids])}"

        NUM_TOKENS_PER_IMAGE = self.data_args.num_image_tokens // pool_size
        if hasattr(self.data_args.image_processor, "crop_size"):
            crop_size = self.data_args.image_processor.crop_size
        else:
            crop_size = self.data_args.image_processor.size

        # Init the padding sample
        seq_id = 0
        while seq_id < len(input_ids):
            # Skip the samples without images
            dummy_image = torch.ones((1, 3, crop_size["height"], crop_size["width"]), device=input_ids[seq_id].device)
            # dummy input_ids include one bos, one image token, and one eos
            dummy_input_ids = torch.zeros_like(input_ids[seq_id][:3])
            dummy_input_ids[0] = self.tokenizer.bos_token_id
            dummy_input_ids[1] = image_token_id
            dummy_input_ids[2] = self.tokenizer.eos_token_id
            dummy_labels = copy.deepcopy(dummy_input_ids)
            dummy_labels[:2] = IGNORE_INDEX
            dummy_seqlen = NUM_TOKENS_PER_IMAGE + 2  # TODO: Check the hard coding of 2
            dummy_position_ids = torch.arange(start=0, end=dummy_seqlen, dtype=torch.int32)
            break

        # Sort with the real length of the sequence
        combined = sorted(
            zip(input_ids, labels, images),
            key=lambda x: len(x[2]) * (NUM_TOKENS_PER_IMAGE - 1) + x[0].size(-1),
            reverse=True,  # Start Packing from the sequence with most images.
        )
        sorted_ids, sorted_labels, sorted_images = zip(*combined)
        sorted_ids, sorted_labels, sorted_images = list(sorted_ids), list(sorted_labels), list(sorted_images)
        max_seq_length = self.tokenizer.model_max_length  # len(sorted_ids[0])
        max_sample_len = 0

        batches = []
        label_batches = []
        position_ids = []
        batch_images = []
        seqlens_in_batch = []

        i = 0
        while i < len(sorted_ids):
            current_batch = torch.tensor([], dtype=torch.int32)
            current_label_batch = torch.tensor([], dtype=torch.int32)
            current_position_ids = torch.tensor([], dtype=torch.int32)
            current_batch_images = []
            current_num_images = 0
            current_len = 0
            current_num_samples = 0

            # Pack a few samples into one sample
            while i < len(sorted_ids):
                num_images = (sorted_ids[i] == image_token_id).sum().item()
                num_image_tokens_added = num_images * (NUM_TOKENS_PER_IMAGE - 1)
                num_incoming_tokens = sorted_ids[i].size(-1) + num_image_tokens_added

                # Handle RingAttn_Varlen which requires `seqlens_in_batch` should be divisible by `ring_degree`
                if self.ring_degree > 1:
                    RING_PAD_TOKEN_INDEX = 2
                    if self.ring_type == "ring_varlen":
                        if num_incoming_tokens % self.sp_degree != 0:
                            pad_len = self.sp_degree - num_incoming_tokens % self.sp_degree
                            num_incoming_tokens += pad_len
                            # pad `input_ids`
                            pad_tensor = torch.full(
                                (pad_len,), RING_PAD_TOKEN_INDEX, dtype=sorted_ids[i].dtype, device=sorted_ids[i].device
                            )
                            sorted_ids[i] = torch.cat([sorted_ids[i], pad_tensor])

                            # pad `label`
                            pad_label_tensor = torch.full(
                                (pad_len,), IGNORE_INDEX, dtype=sorted_labels[i].dtype, device=sorted_labels[i].device
                            )
                            sorted_labels[i] = torch.cat([sorted_labels[i], pad_label_tensor])
                    elif self.ring_type == "zigzag_ring_varlen":
                        self.zigzag_sp_degree = self.sp_degree * 2
                        if num_incoming_tokens % self.zigzag_sp_degree != 0:
                            pad_len = self.zigzag_sp_degree - num_incoming_tokens % self.zigzag_sp_degree
                            num_incoming_tokens += pad_len
                            # pad `input_ids`
                            pad_tensor = torch.full(
                                (pad_len,), RING_PAD_TOKEN_INDEX, dtype=sorted_ids[i].dtype, device=sorted_ids[i].device
                            )
                            sorted_ids[i] = torch.cat([sorted_ids[i], pad_tensor])

                            # pad `label`
                            pad_label_tensor = torch.full(
                                (pad_len,), IGNORE_INDEX, dtype=sorted_labels[i].dtype, device=sorted_labels[i].device
                            )
                            sorted_labels[i] = torch.cat([sorted_labels[i], pad_label_tensor])
                    else:
                        raise ValueError(f"Invalid ring_type: {self.ring_type}")

                if False:  # num_incoming_tokens > max_seq_length:
                    print(
                        f"Warning: Skipping one packed sample with {num_incoming_tokens} tokens,\
                        please consider increase max seq len {max_seq_length}."
                    )
                    i += 1
                    continue

                if (
                    (current_num_images == 0)
                    or (current_num_images < self.sp_degree)
                    or (
                        (current_num_images + num_images <= max_num_images)
                        and (current_len + num_incoming_tokens <= max_sample_len)
                    )
                ) and (current_len + num_incoming_tokens <= max_seq_length):
                    current_num_images += num_images
                    current_len += num_incoming_tokens
                    current_num_samples += 1
                    current_position_ids = torch.cat(
                        (current_position_ids, torch.arange(start=0, end=num_incoming_tokens)), dim=0
                    )
                    current_batch = torch.cat((current_batch, sorted_ids[i]), dim=0)
                    sorted_labels[i][0] = IGNORE_INDEX
                    current_label_batch = torch.cat((current_label_batch, sorted_labels[i]), dim=0)
                    seqlens_in_batch.append(num_incoming_tokens)
                    current_batch_images.extend(sorted_images[i])
                    i += 1
                    assert current_num_images == len(current_batch_images)
                else:
                    break

            # Padding the sample with the dummy image sample, if there are no enough images
            MAX_RETRY = self.sp_degree
            num_retry = 0
            while current_num_images < self.sp_degree and current_len < max_seq_length and num_retry <= MAX_RETRY:
                current_num_images += dummy_image.size(0)
                current_len += dummy_seqlen
                current_num_samples += 1
                current_position_ids = torch.cat((current_position_ids, dummy_position_ids), dim=0)
                current_batch = torch.cat((current_batch, dummy_input_ids), dim=0)
                current_label_batch = torch.cat((current_label_batch, dummy_labels), dim=0)
                seqlens_in_batch.append(dummy_seqlen)
                current_batch_images.extend(dummy_image)
                # We pad from left side to ensure correct grad flow
                # current_batch = torch.cat((dummy_input_ids, current_batch), dim=0)
                # current_label_batch = torch.cat((dummy_labels, current_label_batch), dim=0)
                # seqlens_in_batch.insert(0, dummy_seqlen)
                # current_batch_images = torch.cat((dummy_image, current_batch_images), dim=0)
                num_retry += 1

            # Drop the samples that do not have enough images
            if current_num_images < self.sp_degree:
                print(f"Warning: Skipping one packed sample with {current_num_images} images")
                seqlens_in_batch = seqlens_in_batch[:-current_num_samples]
                continue

            max_sample_len = max(max_sample_len, current_len)
            batches.append(current_batch)
            label_batches.append(current_label_batch)
            position_ids.append(current_position_ids)
            batch_images.append(current_batch_images)

            try:
                assert current_num_images == len(torch.where(current_batch == image_token_id)[0].tolist())
            except AssertionError:
                print(f"Error num_images on {self.sp_rank}", current_num_images)
                print("current_batch", current_batch)
                print(
                    f"Error len(torch.where(batches[i] == image_token_id)[0].tolist() on {self.sp_rank}:",
                    len(torch.where(current_batch == image_token_id)[0].tolist()),
                )
                print(f"Error len(current_batch_images) on {self.sp_rank}:", len(current_batch_images))
                raise AssertionError

        # Split for sequence parallelism
        for i in range(len(batches)):
            image_token_indices = torch.where(batches[i] == image_token_id)[0].tolist()
            image_ids = torch.arange(0, len(image_token_indices), dtype=torch.int32)
            batches[i] = extract_local_input_ids(
                batches[i], image_token_indices, self.sp_rank, self.sp_degree, self.tokenizer.bos_token_id
            )
            label_batches[i] = extract_local_input_ids(
                label_batches[i], image_token_indices, self.sp_rank, self.sp_degree, self.tokenizer.bos_token_id
            )
            batch_images[i] = torch.concat(
                extract_local_from_list(batch_images[i], self.sp_rank, self.sp_degree), dim=0
            )
            H, W = batch_images[i].size(-2), batch_images[i].size(-1)
            batch_images[i] = batch_images[i].reshape(-1, 3, W, H)
            num_images = len(batch_images[i])

            try:
                assert num_images == len(torch.where(batches[i] == image_token_id)[0].tolist())
            except AssertionError:
                print(f"Error num_images on {self.sp_rank}", num_images)
                print("batches[i]", batches[i])
                print(
                    f"Error len(torch.where(batches[i] == image_token_id)[0].tolist() on {self.sp_rank}:",
                    len(torch.where(batches[i] == image_token_id)[0].tolist()),
                )
                print(f"Error batch_images[i] on {self.sp_rank}:", batch_images[i].shape)
                raise AssertionError
            position_ids[i] = extract_local_position_ids(
                position_ids[i], image_token_indices, image_ids, self.sp_rank, self.sp_degree, NUM_TOKENS_PER_IMAGE - 1
            )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            batches, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(label_batches, batch_first=True, padding_value=IGNORE_INDEX)
        seqlens_in_batch = [torch.tensor(x) for x in seqlens_in_batch]
        seqlens_in_batch = torch.stack(seqlens_in_batch, axis=0)
        seqlens_in_batch = seqlens_in_batch.flatten()
        position_ids = torch.nn.utils.rnn.pad_sequence(position_ids, batch_first=True, padding_value=-1)

        if batch_images:
            flat_batch_images = batch_images
            # batch_images = [torch.unbind(images) for images in batch_images]
            # flat_batch_images = [item for sublist in batch_images for item in sublist]
        else:
            flat_batch_images = None
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            # notice that we inject attention mask here
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            seqlens_in_batch=seqlens_in_batch,
            media={"video": flat_batch_images},
            media_config={"video": {}},
            position_ids=position_ids,
        )
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

    from .builder import build_dataset

    train_dataset = build_dataset(data_args.data_mixture, data_args, training_args, tokenizer)
    training_args.sample_lens = [len(d) for d in train_dataset.datasets]

    PROCESS_GROUP_MANAGER = get_pg_manager()
    if PROCESS_GROUP_MANAGER is None:
        data_collator = DataCollator(tokenizer=tokenizer)
    else:
        sp_degree = training_args.seq_parallel_size
        sp_rank = PROCESS_GROUP_MANAGER.sp_rank
        ring_degree = PROCESS_GROUP_MANAGER.ring_degree
        ring_type = PROCESS_GROUP_MANAGER.ring_type
        data_collator = DataCollatorForSupervisedDatasetSeqParallel(
            tokenizer=tokenizer,
            data_args=data_args,
            training_args=training_args,
            sp_degree=sp_degree,
            sp_rank=sp_rank,
            ring_degree=ring_degree,
            ring_type=ring_type,
        )

    return dict(
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
