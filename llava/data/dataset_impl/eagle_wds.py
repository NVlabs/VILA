import copy
import glob
import io
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

from PIL import Image as PILImage

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.data.base import BaseDataset
from llava.data.dataset_impl.utils import _remove_media_tokens
from llava.data.simple_vila_webdataset import VILAWebDataset
from llava.media import Image, Video
from llava.utils import io, make_list

__all__ = ["EagleWDSDataset"]

"""
tabmwp_cot:
    _target_: llava.data.EagleWDSDataset
    data_path: /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/eagle/webdataset/pretrain_separate/ai2d
"""


def process_multi_img(messages, medias, imtok_type="llava") -> List[Dict[str, Any]]:
    original_medias = copy.deepcopy(medias)
    original_msg = copy.deepcopy(messages)

    plain_txt_messages = json.dumps(messages)
    # NOTE(ligeng): use <image-{i}> for eagle
    # if DEFAULT_IMAGE_TOKEN in plain_txt_messages and "<image-1>" in plain_txt_messages:
    #     raise ValueError(f"Found both {DEFAULT_IMAGE_TOKEN} and <image-1> tokens in the messages. Please use only one type of token. {original_msg}")

    # imtok_type = "llava"
    # if "<image-1>" in plain_txt_messages:
    #     imtok_type = "eagle"

    # replace all <image> tokens in the messages
    imtok_reference = defaultdict(int)
    for idx1, msg in enumerate(messages):
        # value = messages[0]["value"]
        value = messages[idx1]["value"]
        new_value = []
        # print(f"==========================")
        # print(messages[idx1]["value"])

        if imtok_type == "llava":
            assert len(medias) == 1, f"Found {len(medias)} images in the instance. {original_msg}"
            # replace all <image> tokens in the message
            while value.find(DEFAULT_IMAGE_TOKEN) >= 0:  # still has <image>
                idx = value.find(DEFAULT_IMAGE_TOKEN)
                img_tok_len = len(DEFAULT_IMAGE_TOKEN)
                if idx > 0:
                    new_value.append(value[:idx])
                new_value.append(medias[0])
                value = value[:idx] + value[idx + img_tok_len :]
                imtok_reference[DEFAULT_IMAGE_TOKEN] += 1

        elif imtok_type == "eagle":
            # replace all <image-{i}> tokens in the message
            for i in range(len(medias)):
                new_image_token = f"<image-{i+1}>"
                img_value = medias[i]
                # print(f"---------- <image-{i+1}> {len(medias)} -----------")
                # print(value)
                while value.find(new_image_token) >= 0:  # still has <image-{i}>
                    idx = value.find(new_image_token)
                    img_tok_len = len(new_image_token)
                    if idx > 0:
                        new_value.append(value[:idx])
                    new_value.append(img_value)
                    value = value[:idx] + value[idx + img_tok_len :]
                    imtok_reference[new_image_token] += 1
                #     print(f"[eagle] Found {new_image_token} {idx, img_tok_len} in the message. ")
                # input()

        new_value.append(value)
        messages[idx1]["value"] = new_value

    # pprint(messages); input()
    if imtok_type == "llava":
        # if DEFAULT_IMAGE_TOKEN in plain_txt_messages:
        #     assert (
        #         len(medias) == 0
        #     ), f"#Num of <images> does not match the number of images in the instance. {original_msg}"
        assert (
            imtok_reference[DEFAULT_IMAGE_TOKEN] > 0
        ), f"[{imtok_type}] Found no reference to {DEFAULT_IMAGE_TOKEN} in the messages. {original_msg}"
    elif imtok_type == "eagle":
        # check empty reference of <image-{i}>
        for i in range(len(medias)):
            new_image_token = f"<image-{i+1}>"
            assert (
                imtok_reference[new_image_token] > 0
            ), f"[{imtok_type}] Found no reference to {new_image_token} in the messages. {original_msg}"
    return messages


class EagleWDSDataset(BaseDataset):
    def __init__(self, data_path: str, is_video=False, **kwargs) -> None:
        # for eagle dataset, let it crash if loading fails
        kwargs["resample_on_failure"] = False
        super().__init__(**kwargs)

        self.data_path = data_path
        self.instances = VILAWebDataset(data_path)
        # print(self.instances[0])

        # for video and s2 handling
        # is_video should be set explicitly for video datasets
        self.is_video = is_video  # or any(["video" in instance for instance in self.instances])
        self.enable_dynamic_res = self.data_args.image_aspect_ratio == "dynamic" and not self.is_video
        self.enable_dynamic_res_s2 = self.data_args.image_aspect_ratio == "dynamic_s2" and not self.is_video

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            messages = copy.deepcopy(instance[".json"]["conversations"])
        except KeyError as e:
            print(f"Failed to load conversation for {instance} {instance.keys()}")
            raise e

        medias = []
        if ".img" in instance:
            # Extract media from the instance
            imtok_type = "llava"
            medias.append(PILImage.open(instance[".img"]))
            if not any(DEFAULT_IMAGE_TOKEN in msg["value"] for msg in messages):
                # If all messages do not contain DEFAULT_IMAGE_TOKEN, add it to the first message
                messages[0]["value"] = DEFAULT_IMAGE_TOKEN + messages[0]["value"]
        else:
            # handle .1.img / .2.img / .3.img like extensions
            imtok_type = "eagle"
            i = 0
            while True:
                i += 1
                key = f".{i}.img"
                if key not in instance:
                    break
                else:
                    medias.append(PILImage.open(instance[key]))
                    # replace <image-{i}> with the default image token
                    # messages = json.loads(
                    #     json.dumps(messages).replace(f"<image-{i}>", DEFAULT_IMAGE_TOKEN)
                    # )

        try:
            original_medias = copy.deepcopy(medias)
            original_msg = copy.deepcopy(messages)
            messages = process_multi_img(messages, medias, imtok_type=imtok_type)
        except Exception as e:
            print(f"Failed to process multi-image messages for {instance}")
            raise e

        # # Remove media tokens from messages
        # for message in messages:
        #     message["value"] = _remove_media_tokens(message["value"])

        # # TODO(ligeng): replace with a more general way to handle media
        # #       appending to first is wrong
        # # Add media to the beginning of the first message
        # messages[0]["value"] = medias + [messages[0]["value"]]
        return messages
