import copy
import glob
import os
from typing import Any, Dict, List, Optional

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.data.base import BaseDataset
from llava.media import Image, Video
from llava.utils import io, make_list

__all__ = [
    "DummyDataset",
]

import os

from huggingface_hub import hf_hub_download


def hf_download_data(
    repo_id="Efficient-Large-Model/VILA-inference-demos",
    filename="imagenet_cat.jpg",
    cache_dir=None,
    repo_type="dataset",
):
    """
    Download dummy data from a Hugging Face repository.

    Args:
    repo_id (str): The ID of the Hugging Face repository.
    filename (str): The name of the file to download.
    cache_dir (str, optional): The directory to cache the downloaded file.

    Returns:
    str: The path to the downloaded file.
    """
    try:
        file_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            repo_type=repo_type,
        )
        return file_path
    except Exception as e:
        print(f"Error downloading file: {e}")
        return None


def _remove_media_tokens(text: str) -> str:
    for token in ["<image>", "<video>"]:
        text = text.replace(token + "\n", "").replace("\n" + token, "").replace(token, "")
    return text.strip()


class DummyDataset(BaseDataset):
    def __init__(self, num_instances: int = 1000, comments: str = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_instances = num_instances
        data = {
            "id": 0,
            "image": hf_download_data("Efficient-Large-Model/VILA-inference-demos", "imagenet_cat.jpg"),
            "conversations": [
                {"from": "human", "value": "<image>\nWhat is in the image?" * 100},
                {"from": "gpt", "value": "A cat."},
            ],
        }
        self.instances = [data] * self.num_instances
        self.media_dir = ""
        self.instances = [data] * self.num_instances
        # self.data_path = data_path
        # self.media_dir = media_dir
        # self.instances = io.load(self.data_path)
        # self.enable_dynamic_res = True

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        messages = copy.deepcopy(instance["conversations"])

        # Extract media from the instance
        medias = []
        if "image" in instance:
            for image_path in make_list(instance["image"]):
                medias.append(Image(os.path.join(self.media_dir, image_path)))
        if "video" in instance:
            for video_path in make_list(instance["video"]):
                medias.append(Video(os.path.join(self.media_dir, video_path)))

        # Remove media tokens from messages
        for message in messages:
            message["value"] = _remove_media_tokens(message["value"])

        # Add media to the beginning of the first message
        messages[0]["value"] = medias + [messages[0]["value"]]
        return messages


