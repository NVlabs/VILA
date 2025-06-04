import random
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from llava.mm_utils import (
    dynamic_process_images_and_prompt,
    dynamic_s2_process_images_and_prompt,
    get_original_image_size,
    process_images,
)

# from llava.utils.tokenizer import preprocess_conversation
from llava.remote_code.tokenizer_utils import preprocess_conversation
from llava.train.args import DataArguments
from llava.utils import io, make_list
from llava.utils.logging import logger
from llava.utils.media import extract_media

__all__ = ["BaseDataset", "local_load_or_hf_load"]


def _process_image(images: List[Any], data_args: DataArguments) -> torch.Tensor:
    return process_images(images, data_args.image_processor, data_args)


def _process_video(videos: List[Any], data_args: DataArguments) -> torch.Tensor:
    return [_process_image(video, data_args) for video in videos]


def local_load_or_hf_load(data_path: str) -> List:
    """
    Load data from either a local file or from Hugging Face Hub.

    Args:
        data_path (str): Path to the data file. Can be a local path or a Hugging Face path.
            For Hugging Face paths, use the following formats:
            - hf://repo_owner/repo_name/path/to/file.json (for model repositories)
            - hf-datasets://repo_owner/repo_name/path/to/file.json (for dataset repositories)

    Returns:
        List: The loaded data as a list / json / npy. See io.load for more details.

    Examples:
        >>> data = local_load_or_hf_load("path/to/local/file.json")
        >>> data = local_load_or_hf_load("hf-datasets://Efficient-Large-Model/thinking_data/chartqa_train_18k/instances.json")
    """
    if data_path.startswith(("hf://", "hf-hub://", "hf-dataset://", "hf-datasets://")):
        # hf://Efficient-Large-Model/thinking_data/chartqa_train_18k/instances.json
        # hf-datasets://Efficient-Large-Model/thinking_data/chartqa_train_18k/instances.json
        repo_type = "dataset" if data_path.startswith(("hf-dataset://", "hf-datasets://")) else "model"
        hf_prefix = data_path.split("://")[0]
        hf_path = data_path.split("://")[-1]  # Efficient-Large-Model/thinking_data/chartqa_train_18k/instances.json
        # hf@main://Efficient-Large-Model/thinking_data/chartqa_train_18k/instances.json
        # hf@0a5f349://Efficient-Large-Model/thinking_data/chartqa_train_18k/instances.json
        if "@" not in hf_prefix:
            revision = "main"
        else:
            revision = hf_prefix.split("@")[-1]

        segs = hf_path.split("/")
        hf_repo_id = "/".join(segs[:2])  # Efficient-Large-Model/thinking_data
        file_rpath = "/".join(segs[2:])  # chartqa_train_18k/instances.json
        from huggingface_hub import hf_hub_download

        fpath = hf_hub_download(repo_id=hf_repo_id, filename=file_rpath, repo_type=repo_type, revision=revision)
        return io.load(fpath)
    else:
        return io.load(data_path)


class BaseDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments,
        no_system_prompt: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.prepend_empty_system_prompt = no_system_prompt
        self.instances = []
        self.enable_dynamic_res = False
        self.enable_dynamic_res_s2 = False
        # global_batch_size: int,
        self.global_batch_size = kwargs.get("global_batch_size", 1)

        # by default, dataset cls will resample on failure
        self.resample_on_failure = kwargs.get("resample_on_failure", True)
        self.system_prompt = kwargs.get("system_prompt", None)

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Dict[str, Any]:
        instance = self.instances[index]

        try:
            # Process instance to conversation
            conversation = self.process(instance)

            # Extract media from conversation
            media = extract_media(conversation, self.data_args)

            block_sizes = []
            # Process media
            if "image" in media:
                if self.enable_dynamic_res_s2:
                    processed_images, block_sizes = dynamic_s2_process_images_and_prompt(
                        media["image"], conversation[0]["value"], self.data_args
                    )
                elif self.enable_dynamic_res and self.data_args.image_aspect_ratio == "dynamic":
                    processed_images, processed_prompt = dynamic_process_images_and_prompt(
                        media["image"], conversation[0]["value"], self.data_args
                    )
                    conversation[0]["value"] = processed_prompt
                else:
                    processed_images = _process_image(media["image"], self.data_args)
                original_image_sizes = [get_original_image_size(img) for img in media["image"]]

            if "video" in media:
                if self.enable_dynamic_res_s2 and self.data_args.video_max_tiles > 1:
                    processed_images, block_sizes = dynamic_s2_process_images_and_prompt(
                        media["video"][0],
                        conversation[0]["value"],
                        self.data_args,
                        max_tiles=self.data_args.video_max_tiles,
                    )
                    # For HighRes video training, we use <image> token instead of <vila/video>
                    conversation[0]["value"] = processed_prompt.replace("<vila/video>", "")
                elif (
                    self.enable_dynamic_res
                    and self.data_args.image_aspect_ratio == "dynamic"
                    and self.data_args.video_max_tiles > 1
                ):
                    processed_images, processed_prompt = dynamic_process_images_and_prompt(
                        media["video"][0],
                        conversation[0]["value"],
                        self.data_args,
                        max_tiles=self.data_args.video_max_tiles,
                    )
                    # For HighRes video training, we use <image> token instead of <vila/video>
                    conversation[0]["value"] = processed_prompt.replace("<vila/video>", "")
                else:
                    processed_images = _process_video(media["video"], self.data_args)

            # Prepare "input_ids" and "labels" for training
            if self.system_prompt is not None:
                assert (
                    self.prepend_empty_system_prompt == False
                ), "system_prompt and prepend_empty_system_prompt cannot both be set"
                if isinstance(self.system_prompt, (list, tuple)):
                    sys_prompt = random.choice(self.system_prompt)
                else:
                    sys_prompt = self.system_prompt
                conversation = [{"from": "system", "value": sys_prompt}] + conversation

            data = preprocess_conversation(
                conversation, self.tokenizer, no_system_prompt=self.prepend_empty_system_prompt
            )

            if self.enable_dynamic_res_s2 and ("image" in media or "video" in media):
                data["block_sizes"] = block_sizes

            if "image" in media:
                data["image"] = processed_images
                data["original_image_sizes"] = original_image_sizes
            if "video" in media:
                if (
                    self.enable_dynamic_res_s2 == True or self.enable_dynamic_res == True
                ) and self.data_args.video_max_tiles > 1:
                    # HighRes video training
                    data["image"] = processed_images
                else:
                    data["video"] = processed_images

        except Exception as e:
            if not self.resample_on_failure:
                raise e
            else:
                logger.exception(f"Error processing instance '{instance}': '{e}'. Resampling.")
                return self.__getitem__(random.randint(0, len(self.instances) - 1))

        return data

    def __len__(self) -> int:
        return len(self.instances)
