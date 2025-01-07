import copy
import glob
import os
import random
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.data.base import BaseDataset
from llava.media import Image, Video
from llava.mm_utils import dynamic_process_images_and_prompt, process_images
from llava.train.args import DataArguments
from llava.utils import io, make_list
from llava.utils.logging import logger
from llava.utils.media import extract_media
from llava.utils.tokenizer import preprocess_conversation

from .utils import _process_image, _remove_media_tokens

__all__ = [
    "LLaVACOTDataset",
]


class LLaVACOTDataset(BaseDataset):
    def __init__(
        self, data_path: str, media_dir: Optional[str] = None, name=str, cot_relabel_path=None, is_video=False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.media_dir = media_dir
        self.instances = io.load(self.data_path)
        global_batch_size = kwargs["global_batch_size"]
        self.is_video = is_video or any(["video" in instance for instance in self.instances])
        self.enable_dynamic_res = self.data_args.image_aspect_ratio == "dynamic" and not self.is_video
        self.enable_dynamic_res_s2 = self.data_args.image_aspect_ratio == "dynamic_s2" and not self.is_video

        self.name = name
        self.cot_relabel_path = io.load(cot_relabel_path) if cot_relabel_path is not None else None

        count = 0
        avaliable_dst = set()
        for k, v in self.cot_relabel_path.items():
            if name in k:
                count += 1
        logger.info(
            f"Total number of cot relabeled conversations in {name}: {count}, original instances: {len(self.instances)}"
        )

        residual = global_batch_size - len(self.instances) % global_batch_size
        if residual != global_batch_size:
            if global_batch_size // len(self.instances) >= 2:
                self.instances = self.instances * (global_batch_size // len(self.instances))
                residual = global_batch_size - len(self.instances) % global_batch_size
            selected_elements = random.sample(range(len(self.instances)), residual)
            additional_instance = [self.instances[i] for i in selected_elements]
            self.instances.extend(additional_instance)

    def process(self, instance: Dict[str, Any], index: int) -> List[Dict[str, Any]]:
        messages = copy.deepcopy(instance["conversations"])
        uid = f"{self.name}/{index}-{0}"

        # Extract media from the instance
        medias = []
        if "image" in instance:
            for image_path in make_list(instance["image"]):
                medias.append(Image(os.path.join(self.media_dir, image_path)))

        # NOTE(ligeng): quick workaround for idefics2_sft
        if "images" in instance:
            for image_path in make_list(instance["images"]):
                medias.append(Image(os.path.join(self.media_dir, image_path)))

        if "video" in instance:
            for video_path in make_list(instance["video"]):
                medias.append(Video(os.path.join(self.media_dir, video_path)))

        if self.cot_relabel_path is not None and uid in self.cot_relabel_path and len(self.cot_relabel_path[uid]) > 0:
            logger.info(f"Overriding COT relabeled conversation '{uid}'")
            messages = self.cot_relabel_path[uid]

        # Remove media tokens from messages
        for message in messages:
            message["value"] = _remove_media_tokens(message["value"])

        # Add media to the beginning of the first message
        messages[0]["value"] = medias + [messages[0]["value"]]
        return messages

    def __getitem__(self, index: int) -> Dict[str, Any]:
        instance = self.instances[index]

        try:
            # Process instance to conversation
            conversation = self.process(instance, index)

            # Extract media from conversation
            media = extract_media(conversation, self.data_args)

            # Process media
            if "image" in media:
                if self.enable_dynamic_res and self.data_args.image_aspect_ratio == "dynamic":
                    processed_images, processed_prompt = dynamic_process_images_and_prompt(
                        media["image"], conversation[0]["value"], self.data_args
                    )
                    conversation[0]["value"] = processed_prompt
                else:
                    processed_images = _process_image(media["image"], self.data_args)

            # Prepare "input_ids" and "labels" for training
            data = preprocess_conversation(conversation, self.tokenizer, no_system_prompt=self.no_system_prompt)

            if "image" in media:
                data["image"] = processed_images

        except Exception as e:
            logger.exception(f"Error processing instance '{instance}': '{e}'.")
            raise e
            # return self.__getitem__(random.randint(0, len(self.instances) - 1))

        return data


def process_multi_img(self, instance: Dict[str, Any], index: int) -> List[Dict[str, Any]]:
    messages = copy.deepcopy(instance["conversations"])
    uid = f"{self.name}/{index}-{0}"

    # Extract media from the instance
    medias = []
    if "image" in instance:
        for image_path in make_list(instance["image"]):
            medias.append(Image(os.path.join(self.media_dir, image_path)))

    # NOTE(ligeng): quick workaround for idefics2_sft
    if "images" in instance:
        for image_path in make_list(instance["images"]):
            medias.append(Image(os.path.join(self.media_dir, image_path)))

    if "video" in instance:
        for video_path in make_list(instance["video"]):
            medias.append(Video(os.path.join(self.media_dir, video_path)))

    cot_relabel = False
    # replace all <image> tokens in the messages
    for idx1, msg in enumerate(messages):
        # value = messages[0]["value"]
        value = messages[idx1]["value"]
        img_tok_len = len(DEFAULT_IMAGE_TOKEN)
        new_value = []

        while value.find(DEFAULT_IMAGE_TOKEN) >= 0:  # still has <image>
            idx = value.find(DEFAULT_IMAGE_TOKEN)
            if idx > 0:
                new_value.append(value[:idx])
            new_value.append(medias.pop(0))
            value = value[idx + img_tok_len :]

        if self.cot_relabel_path is not None and uid in self.cot_relabel_path:
            try:
                logger.info(f"[COT] Overriding relabeled conversation '{uid}/{idx1}'")
                value = self.cot_relabel_path[uid][idx1]["value"]
            except IndexError as e:
                logger.warning(
                    f"[COT] IndexError: {uid}/{idx1} {len(self.cot_relabel_path[uid])}, using original captions"
                )

        new_value.append(value)
        messages[idx1]["value"] = new_value

    # pprint(messages); input()
    assert len(medias) == 0, f"#Num of <images> does not match the number of images in the instance. {instance}"

    return messages


