import glob
import os
import random
import re
from typing import Any, Dict, List, Optional

import numpy as np

from llava.data.base import BaseDataset
from llava.media import Video
from llava.utils import io
from llava.utils.logging import logger

__all__ = ["DVCDataset", "ELDataset", "RTLDataset", "VideoQADataset"]


def _check_video_path(path: str) -> str:
    if os.path.exists(path):
        # If the path is a directory, check if all files have the same extension.
        if os.path.isdir(path):
            matches = glob.glob(os.path.join(path, "*"))
            if not matches:
                raise ValueError(f"No files found in {path}")
            exts = {os.path.splitext(f)[1] for f in matches}
            if len(exts) > 1:
                raise ValueError(f"Multiple extensions found in {path}: {exts}")
        return path

    # If the path does not exist, it might be because the extension is missing.
    # Try to find the file with the same name with an extension.
    matches = glob.glob(path + ".*")
    if not matches:
        return path
    if len(matches) > 1:
        logger.warning(f"Multiple matches found for {path}: {matches}. Using the shortest one.")
    return sorted(matches, key=len)[0]


def encode_time_token(text: str, *, duration: float, num_time_tokens: int, time_token_format: str) -> str:
    encoded = ""
    last = 0
    for m in re.finditer(r"\<(?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )\>", text, re.VERBOSE):
        start, end = m.start(0), m.end(0)
        time = float(m.group(0)[1:-1])
        t = int(np.round(time / duration * (num_time_tokens - 1)))
        encoded += text[last:start] + time_token_format.format(t=t)
        last = end
    encoded += text[last:]
    return encoded.strip()


class DVCDataset(BaseDataset):
    TASK_PROMPTS = [
        "Provide a detailed description of the given video.",
        "Describe the provided video in detail.",
        "Summarize the visual content of the video.",
        "Write a informative summary of the video.",
    ]
    TIME_PROMPTS = [
        "Each sentence should begin with the start and end timestamps.",
        "At the beginning of each sentence, include the start and end timestamps.",
        "Prepend each sentence with its start and end timestamps.",
    ]

    def __init__(self, data_path: str, video_dir: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.video_dir = video_dir

        # Load data
        data = io.load(self.data_path)
        if isinstance(data, dict):
            data = [{"id": k, **v} for k, v in data.items()]

        # Filter out videos that do not exist
        for instance in data:
            video_path = _check_video_path(os.path.join(self.video_dir, instance.pop("id")))
            if not os.path.exists(video_path):
                logger.warning(f"Video `{video_path}` not found. Excluded from dataset.")
                continue
            instance["video_path"] = video_path
            self.instances.append(instance)

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Generate question
        question = random.choice(self.TASK_PROMPTS) + " " + random.choice(self.TIME_PROMPTS)

        # Generate answer
        answer = ""
        for e, (start, end) in enumerate(instance["timestamps"]):
            event = instance["sentences"][e].strip()
            answer += encode_time_token(
                f"<{start}> <{end}>",
                duration=instance["duration"],
                num_time_tokens=self.data_args.num_time_tokens,
                time_token_format=self.data_args.time_token_format,
            )
            answer += f" {event} "
        answer = answer.strip()

        # Add video to the beginning of the question
        video = Video(instance["video_path"])
        question = [video, question]

        # Generate messages
        messages = []
        messages.append({"from": "human", "value": question})
        messages.append({"from": "gpt", "value": answer})
        return messages


class ELDataset(BaseDataset):
    TASK_PROMPTS = [
        'When does "{event}" happen in the video?',
        'At what point in the video does "{event}" happen?',
        'When is "{event}" depicted in the video?',
        'At what time in the video does "{event}" take place?',
    ]
    TIME_PROMPTS = [
        "Answer the question only using start and end timestamps.",
        "Provide a response using only start and end timestamps.",
        "Convey your answer using start and end timestamps exclusively.",
    ]

    def __init__(self, data_path: str, video_dir: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.video_dir = video_dir

        # Load data
        data = io.load(self.data_path)
        if isinstance(data, dict):
            data = [{"id": k, **v} for k, v in data.items()]

        # Filter out videos that do not exist
        for instance in data:
            video_path = _check_video_path(os.path.join(self.video_dir, instance.pop("id")))
            if not os.path.exists(video_path):
                logger.warning(f"Video `{video_path}` not found. Excluded from dataset.")
                continue
            instance["video_path"] = video_path
            self.instances.append(instance)

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Randomly select an event
        e = random.randint(0, len(instance["timestamps"]) - 1)

        # Generate question
        event = instance["sentences"][e]
        event = event.strip().rstrip(".")
        event = event[0].lower() + event[1:]
        question = random.choice(self.TASK_PROMPTS).format(event=event) + " " + random.choice(self.TIME_PROMPTS)

        # Generate answer
        start, end = instance["timestamps"][e]
        answer = encode_time_token(
            f"<{start}> <{end}>",
            duration=instance["duration"],
            num_time_tokens=self.data_args.num_time_tokens,
            time_token_format=self.data_args.time_token_format,
        )

        # Add video to the beginning of the question
        video = Video(instance["video_path"])
        question = [video, question]

        # Generate messages
        messages = []
        messages.append({"from": "human", "value": question})
        messages.append({"from": "gpt", "value": answer})
        return messages


class RTLDataset(BaseDataset):
    def __init__(self, data_path: str, video_dir: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.video_dir = video_dir
        for vid, data in io.load(self.data_path).items():
            video_path = _check_video_path(os.path.join(self.video_dir, vid))
            if not os.path.exists(video_path):
                logger.warning(f"Video `{video_path}` not found. Excluded from dataset.")
                continue
            for vqa in data["QA"]:
                self.instances.append({"video_path": video_path, "duration": data["duration"], "QA": [vqa]})

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Generate messages
        messages = []
        for vqa in instance["QA"]:
            messages.append({"from": "human", "value": vqa["q"].strip()})
            messages.append({"from": "gpt", "value": vqa["a"].strip()})

        # Encode time tokens in answers
        for message in messages:
            if message["from"] == "gpt":
                message["value"] = encode_time_token(
                    message["value"],
                    duration=instance["duration"],
                    num_time_tokens=self.data_args.num_time_tokens,
                    time_token_format=self.data_args.time_token_format,
                )

        # Add video to the first message
        video = Video(instance["video_path"])
        messages[0]["value"] = [video, messages[0]["value"]]
        return messages


class VideoQADataset(BaseDataset):
    def __init__(self, data_path: str, video_dir: str, task_prompt: Optional[str] = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.data_path = data_path
        self.video_dir = video_dir
        self.task_prompt = task_prompt
        self.instances = []
        for instance in io.load(self.data_path):
            video_path = _check_video_path(os.path.join(self.video_dir, instance.pop("video")))
            if not os.path.exists(video_path):
                logger.warning(f"Video `{video_path}` not found. Excluded from dataset.")
                continue
            instance["video_path"] = video_path
            self.instances.append(instance)

    def process(self, instance: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Generate messages
        messages = []
        for vqa in instance["QA"]:
            messages.append({"from": "human", "value": vqa["q"].strip()})
            messages.append({"from": "gpt", "value": vqa["a"].strip()})

        # Add task prompt to all human messages
        # TODO(zhijianl): If this is multi-round conversation, should we
        # add task prompt to all messages or just the first one?
        if self.task_prompt is not None:
            for message in messages:
                if message["from"] == "human":
                    message["value"] += "\n" + self.task_prompt.strip()

        # Add video to the first message
        video = Video(instance["video_path"])
        messages[0]["value"] = [video, messages[0]["value"]]
        return messages


