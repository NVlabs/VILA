import argparse
import itertools
import json
import os
import re
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

import llava
from llava.data import DATASETS
from llava.media import Video
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger


def iou(s1: Tuple[float, float], s2: Tuple[float, float]) -> float:
    i = max(min(s1[1], s2[1]) - max(s1[0], s2[0]), 0)
    u = max(s1[1] - s1[0], 0) + max(s2[1] - s2[0], 0) - i
    return i / u if u > 0 else 0


def precision(threshold: float):
    def precision_func(s1: Tuple[float, float], s2: Tuple[float, float]) -> float:
        return float(iou(s1, s2) >= threshold)

    return precision_func


def decode_time_token(text: str, *, duration: float, num_time_tokens: int, time_token_format: str) -> str:
    """Replace time tokens in text with actual timestamps."""
    for t in range(num_time_tokens):
        time_token = time_token_format.format(t=t)
        timestamp = round(t * duration / (num_time_tokens - 1), 2)
        text = text.replace(time_token, f"<{timestamp}>")

    # Handle out-of-range time tokens
    excess_pattern = re.compile(rf"<t(\d+)>")
    matches = excess_pattern.findall(text)
    for match in matches:
        t = int(match)
        if t >= num_time_tokens:
            timestamp = round(duration, 2)  # Map to the end of the video
            text = text.replace(f"<t{t}>", f"<{timestamp}>")

    return text


def parse_timestamps(text: str, duration: float, strict: bool = False) -> Tuple[float, float]:
    matches = list(re.finditer(r"\<(?: (?: \d* \.? \d+ ) | (?: \d+ \.? ) )\>", text, re.VERBOSE))
    if strict:
        assert len(matches) >= 2, "Expected at least two timestamps in the text."
    elif len(matches) < 2:
        return [0, duration]
    timestamps = []
    for match in matches[:2]:
        timestamp = float(match.group(0)[1:-1])
        timestamps.append(min(max(timestamp, 0), duration))
    return [min(timestamps), max(timestamps)]


def eval_model(args):

    # Set up distributed environment
    dist.init()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])

    # Load data and chunk it
    question_file = DATASETS[args.dataset]["data_path"]
    image_folder = DATASETS[args.dataset]["video_dir"]
    instances = io.load(question_file)[dist.rank() :: dist.size()]

    # Load model
    model = llava.load(args.model_path, model_base=args.model_base, devices=devices)

    # Set up generation config
    generation_config = model.default_generation_config
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    # Run inference
    outputs = []
    for instance in tqdm(instances, disable=not dist.is_main()):
        vid = instance["id"]
        sentence = instance["sentences"][0]
        sentence = sentence.strip().rstrip(".")
        if len(sentence) > 1:
            sentence = sentence[0].lower() + sentence[1:]
        task_prompt = 'When does "%s" happen in the video? Answer the question only using start and end timestamps.'
        question = task_prompt % sentence

        # Get GT timestamps
        answer = "<{:f}> <{:f}>".format(instance["timestamps"][0][0], instance["timestamps"][0][1])

        video_path = os.path.join(image_folder, vid)
        video = Video(video_path)
        response = model.generate_content([video, question], generation_config=generation_config)

        # Decode time tokens
        response = decode_time_token(
            response,
            duration=instance["duration"],
            num_time_tokens=model.config.num_time_tokens,
            time_token_format=model.config.time_token_format,
        )
        # print(response)

        output = {
            "vid": vid,
            "qid": vid,
            "question": question,
            "output": response,
            "target": answer,
            "duration": instance["duration"],
        }
        outputs.append(output)

    # Gather and save outputs
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = list(itertools.chain(*outputs))
    io.save(os.path.join(args.output_dir, "outputs.jsonl"), outputs)

    # Run evaluation
    metric_func = {
        "iou": iou,
        "precision@0.3": precision(0.3),
        "precision@0.5": precision(0.5),
        "precision@0.7": precision(0.7),
    }
    # metric_func = {"iou": iou, "precision@0.5": precision(0.5)}
    metrics = {name: defaultdict(list) for name in metric_func}

    for output in outputs:
        for name in metrics:
            metrics[name][output["vid"]].append(
                metric_func[name](
                    parse_timestamps(output["output"], output["duration"], strict=False),
                    parse_timestamps(output["target"], output["duration"], strict=True),
                )
            )
    for name in metrics:
        metrics[name] = np.mean([np.mean(metrics[name][vid]) for vid in metrics[name]])
    io.save(os.path.join(args.output_dir, "metrics.json"), metrics)
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--visual-data-type", type=str, default="video_frames")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--dataset", type=str, required=True)

    args = parser.parse_args()

    eval_model(args)
