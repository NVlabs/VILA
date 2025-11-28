import copy
import glob
import os
import random
from typing import Any, Dict, List, Optional

import PIL
from datasets import get_dataset_config_names, load_dataset
from torch.utils.data import ConcatDataset, Dataset

from llava.constants import DEFAULT_IMAGE_TOKEN
from llava.data.base import BaseDataset, local_load_or_hf_load
from llava.data.dataset_impl.utils import _remove_media_tokens
from llava.media import Image, Video
from llava.utils import io, make_list

__all__ = ["HFParquetDataset"]


class HFParquetDataset(BaseDataset):
    def __init__(self, data_path: str, media_dir: Optional[str] = None, is_video=False, **kwargs) -> None:
        kwargs["data_path"] = data_path
        super().__init__(**kwargs)
        self.data_path = data_path
        self.media_dir = media_dir

        self.instances = load_dataset(
            "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/FineVision", name="ai2d_merged", split="train"
        )
        # self.instances = local_load_or_hf_load(self.data_path)
        global_batch_size = kwargs.get("global_batch_size", None)
        self.is_video = is_video or any(["video" in instance for instance in self.instances])

        self.enable_dynamic_res = self.data_args.image_aspect_ratio == "dynamic"
        self.enable_dynamic_res_s2 = self.data_args.image_aspect_ratio == "dynamic_s2"

        if global_batch_size is not None:
            residual = global_batch_size - len(self.instances) % global_batch_size
            if residual != global_batch_size:
                if global_batch_size // len(self.instances) >= 2:
                    self.instances = self.instances * (global_batch_size // len(self.instances))
                    residual = global_batch_size - len(self.instances) % global_batch_size
                selected_elements = random.sample(range(len(self.instances)), residual)
                additional_instance = [self.instances[i] for i in selected_elements]
                self.instances = ConcatDataset([self.instances, additional_instance])

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        raw_msgs = copy.deepcopy(instance["texts"])
        # print(raw_msgs)
        # input()

        # Extract media from the instance
        medias = []
        if "image" in instance:
            for image_path in make_list(instance["image"]):
                medias.append(Image(os.path.join(self.media_dir, image_path)))
            if self.data_args.max_num_images is not None:
                medias = medias[: min(self.data_args.max_num_images, len(medias))]

        # NOTE(ligeng): quick workaround for idefics2_sft
        if "images" in instance:
            for image_path in make_list(instance["images"]):
                if isinstance(image_path, str):
                    medias.append(Image(os.path.join(self.media_dir, image_path)))
                elif isinstance(image_path, PIL.Image.Image):
                    medias.append(image_path)
                else:
                    raise ValueError(f"Unsupported image type: {type(image_path)}")
            if self.data_args.max_num_images is not None:
                medias = medias[: min(self.data_args.max_num_images, len(medias))]

        if "video" in instance:
            for video_path in make_list(instance["video"]):
                medias.append(Video(os.path.join(self.media_dir, video_path)))

        # raw_msg: [{'user': 'Question: What do respiration and combustion give out\nChoices:\nA. Oxygen\nB. Carbon dioxide\nC. Nitrogen\nD. Heat\nAnswer with the letter.', 'assistant': 'Answer: B'}, {'user': 'Here is a diagram figure extracted from some Grade 1 - 6 science books.\nPlease first describe the content of this figure in detail, including how the knowledge visually displayed in the diagram.\nThen start with a section title "related knowledge:", briefly and concisely highlight the related domain knowledge and theories that underly this diagram. Note that you do not need to provide much detail. Simply cover the most important concepts.', 'assistant': 'The figure is a diagram that represents the carbon cycle. The diagram is in the shape of a circle, indicating the cyclical nature of the process. On the left side of the diagram, there is an image of green plants with an arrow pointing towards them labeled "photosynthesis." This arrow indicates that green plants take in carbon dioxide from the atmosphere during photosynthesis. Below the plants, there is an arrow labeled "carbon dioxide" that points to the right side of the diagram. On the right side, there are images of a rabbit, a factory with smokestacks, and dead organisms. Above the rabbit and factory, there is an arrow labeled "respiration (plants and animals)" that points up towards an arrow labeled "oxygen." This indicates that both plants and animals release oxygen into the atmosphere during respiration. Below the dead organisms, there is an arrow labeled "respiration (decomposers)" that points back towards the "carbon dioxide" arrow, completing the cycle. There is also an arrow labeled "combustion" pointing from the factory towards the "oxygen" arrow, indicating that combustion also releases oxygen into the atmosphere.\n\nRelated Knowledge:\nThe diagram represents the carbon cycle, which is a fundamental concept in ecology and environmental science. The carbon cycle is the process by which carbon moves through the Earth\'s atmosphere, oceans, and land. Key concepts related to the carbon cycle include photosynthesis, respiration, decomposition, and combustion. Photosynthesis is the process by which green plants use sunlight to convert carbon dioxide and water into glucose and oxygen. Respiration is the process by which living organisms, including plants and animals, release energy by breaking down glucose and releasing carbon dioxide and water. Decomposition is the process by which dead organisms are broken down by decomposers, such as bacteria and fungi, releasing carbon dioxide back into the atmosphere. Combustion is the process by which carbon-containing materials, such as fossil fuels, are burned, releasing carbon dioxide and other gases into the atmosphere. The carbon cycle is essential for maintaining the balance of carbon in the Earth\'s atmosphere and is closely linked to other biogeochemical cycles, such as the nitrogen and water cycles.'}]
        messages = []
        # Remove media tokens from messages
        for raw_msg in raw_msgs:
            messages += [
                {"from": "human", "value": _remove_media_tokens(raw_msg["user"])},
                {"from": "gpt", "value": _remove_media_tokens(raw_msg["assistant"])},
            ]

        # Add media to the beginning of the first message
        if messages[0]["from"] == "human":
            messages[0]["value"] = medias + [messages[0]["value"]]
        else:
            raise ValueError(f"First message is not from human: {messages}")

        # print(messages)
        # input()
        return messages
