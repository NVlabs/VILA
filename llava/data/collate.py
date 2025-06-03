from dataclasses import dataclass
from typing import Any, Dict, Sequence

import torch
from transformers import PreTrainedTokenizer

from llava.constants import IGNORE_INDEX
from llava.utils.logging import logger

__all__ = ["DataCollator"]


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizer

    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        # Gather everything from the batch
        input_ids, labels, media, block_sizes = [], [], {name: [] for name in self.tokenizer.media_tokens}, []
        for instance in instances:
            if isinstance(instance["input_ids"], torch.Tensor):
                input_ids.append(instance["input_ids"])
                labels.append(instance["labels"])
                for name in media:
                    objs = instance.get(name)
                    objs = objs if objs is not None else []
                    media[name].append([obj for obj in objs])
                if "block_sizes" in instance:
                    block_sizes.append(instance["block_sizes"])
                else:
                    block_sizes.append(
                        [None for _ in range(len(instance.get("image")))] if instance.get("image") is not None else []
                    )
            else:
                input_ids.extend(instance["input_ids"])
                labels.extend(instance["labels"])
                for name in media:
                    objs = instance.get(name)
                    objs = objs if objs is not None else [[] for _ in range(len(instance["input_ids"]))]
                    media[name].extend(objs)
                if "block_sizes" in instance:
                    block_sizes.extend(instance["block_sizes"])
                else:
                    block_sizes.extend(
                        [[None for _ in range(len(objs))] for objs in instance.get("image")]
                        if instance.get("image") is not None
                        else [[] for _ in range(len(instance["input_ids"]))]
                    )

        batch_size = len(input_ids)

        # Check if the number of media objects (or the number of block sizes) matches the number of media tokens
        for name in media:
            for k in range(batch_size):
                if name == "image" and not all([_ is None for _ in block_sizes[k]]):
                    actual = len(block_sizes[k])
                else:
                    actual = len(media[name][k])
                expected = (input_ids[k] == self.tokenizer.media_token_ids[name]).sum().item()
                if actual != expected:
                    raise ValueError(
                        f"Number mismatch between {name} objects and {name} tokens. "
                        f"There are {expected} {name} tokens but {actual} {name} objects."
                    )

        # Batchify the inputs
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Truncate media objects if necessary
        for name in media:
            objects = []
            for k in range(batch_size):
                if name == "image" and not all([_ is None for _ in block_sizes[k]]):
                    actual = len(media[name][k])
                    num_large_scale_blocks = sum([x * y for x, y in block_sizes[k]])
                    num_small_scale_blocks = actual - num_large_scale_blocks
                    num_small_scale_blocks_each_img = num_small_scale_blocks // len(block_sizes[k])
                    expected_full_image = (input_ids[k] == self.tokenizer.media_token_ids[name]).sum().item()
                    expected = (
                        sum([x * y for x, y in block_sizes[k][:expected_full_image]])
                        + num_small_scale_blocks_each_img * expected_full_image
                    )
                    if actual > expected:
                        logger.warning(f"Truncating the number of {name} objects from {actual} to {expected}")
                        media[name][k] = media[name][k][:expected]
                    objects.extend(media[name][k])
                    block_sizes[k] = block_sizes[k][:expected_full_image]
                else:
                    actual = len(media[name][k])
                    expected = (input_ids[k] == self.tokenizer.media_token_ids[name]).sum().item()
                    if actual > expected:
                        logger.warning(f"Truncating the number of {name} objects from {actual} to {expected}")
                        media[name][k] = media[name][k][:expected]
                    objects.extend(media[name][k])
                    if name == "image":
                        block_sizes[k] = block_sizes[k][:expected]
            media[name] = objects

        # Flatten block sizes from [[bls_im1_instance1, bls_im2_instance1], [bls_im1_instance2, bls_im2_instance2], ...] to [bls_im1_instance1, bls_im2_instance1, bls_im1_instance2, bls_im2_instance2, ...]
        block_sizes = sum(block_sizes, [])

        original_image_sizes = []
        for instance in instances:
            if instance.get("original_image_sizes") is not None:
                if isinstance(instance["input_ids"], list):
                    original_image_sizes.extend(instance["original_image_sizes"])
                else:
                    original_image_sizes.append(instance["original_image_sizes"])
            else:
                if isinstance(instance["input_ids"], list):
                    nums_image = (
                        [len(image) for image in instance.get("image")]
                        if instance.get("image") is not None
                        else [0] * len(instance["input_ids"])
                    )
                    original_image_sizes.extend([[None] * num_image for num_image in nums_image])
                else:
                    num_image = len(instance.get("image", []))
                    original_image_sizes.append([None] * num_image)
        original_image_sizes = sum(original_image_sizes, [])

        gt_selection_maps = []
        for instance in instances:
            gt_selection_maps.append(instance.get("gt_selection_map", None))
        assert all([m is not None for m in gt_selection_maps]) or all(
            [m is None for m in gt_selection_maps]
        )  # currently don't support mixing regular data and grounding data
        if all([m is not None for m in gt_selection_maps]):
            gt_selection_maps = torch.stack(gt_selection_maps, dim=0)
        else:
            gt_selection_maps = None

        return {
            "input_ids": input_ids,
            "media": media,
            "media_config": {
                "image": {"block_sizes": block_sizes, "original_image_sizes": original_image_sizes},
                "video": {},
            },
            "labels": labels,
            "attention_mask": attention_mask,
            "gt_selection_maps": gt_selection_maps,
        }
