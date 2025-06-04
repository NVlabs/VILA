import copy
import glob
import io
import json
import os
import random
import tempfile
import uuid
from collections import defaultdict
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image as PILImage

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.data.base import BaseDataset
from llava.data.dataset_impl.eagle_wds import process_multi_img
from llava.data.dataset_impl.utils import _remove_media_tokens
from llava.data.simple_vila_webdataset import VILAWebDataset
from llava.media import Image, Video
from llava.utils import io, make_list

__all__ = ["EagleVideoWDSDataset"]

"""
eagle/video-cap-ego4d-video-recap:
    _target_: llava.data.EagleVideoWDSDataset
    is_video: true
    data_path: /lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/eagle-video/wds/video-cap-ego4d-video-recap
"""


class EagleVideoWDSDataset(BaseDataset):
    def __init__(self, data_path: str, is_video=True, **kwargs) -> None:
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
        messages = copy.deepcopy(instance[".json"]["conversations"])

        medias = []
        if ".mp4" in instance:
            # NOTE(ligeng): the video is a BytesIO object, current opencv does not provide a direct way to read from bytesiot thus we need to save it to a file first.
            bts = instance[".mp4"]
            medias.append(Video(bts))
        else:
            raise ValueError(f"No media found for {instance}")

        # Remove media tokens from messages
        for message in messages:
            message["value"] = _remove_media_tokens(message["value"])

        # Add media to the beginning of the first message
        messages[0]["value"] = medias + [messages[0]["value"]]

        return messages


if __name__ == "__main__":
    dataset = EagleVideoWDSDataset(
        data_path="/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/eagle-video/wds/video-cap-ego4d-video-recap"
    )
    print(dataset[0])
