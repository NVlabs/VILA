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


def decode_time_token(text: str, *, duration: float, num_time_tokens: int, time_token_format: str) -> str:
    for t in range(num_time_tokens):
        time_token = time_token_format.format(t=t)
        t = round(t * duration / (num_time_tokens - 1), 2)
        text = text.replace(time_token, f"<{t}>")
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


def iou(s1: Tuple[float, float], s2: Tuple[float, float]) -> float:
    i = max(min(s1[1], s2[1]) - max(s1[0], s2[0]), 0)
    u = max(s1[1] - s1[0], 0) + max(s2[1] - s2[0], 0) - i
    return i / u if u > 0 else 0


def precision(threshold: float):
    def precision_func(s1: Tuple[float, float], s2: Tuple[float, float]) -> float:
        return float(iou(s1, s2) >= threshold)

    return precision_func


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    # Set up distributed environment
    dist.init()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])

    # Load model
    model = llava.load(args.model_path, model_base=args.model_base, devices=devices)

    # Set up generation config
    generation_config = model.default_generation_config
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    # Load data and chunk it
    data_path = DATASETS[args.dataset]["data_path"]
    video_dir = DATASETS[args.dataset]["video_dir"]
    instances = io.load(data_path)[dist.rank() :: dist.size()]

    # Run inference
    outputs = []
    for instance in tqdm(instances, disable=not dist.is_main()):
        vid = instance["vid"]
        question = instance["question"]

        video = Video(os.path.join(video_dir, "v_" + vid))
        response = model.generate_content([video, question], generation_config=generation_config)

        # Decode time tokens
        response = decode_time_token(
            response,
            duration=instance["duration"],
            num_time_tokens=model.config.num_time_tokens,
            time_token_format=model.config.time_token_format,
        )

        output = {
            "vid": vid,
            "qid": instance["question_id"],
            "question": question,
            "output": response,
            "target": instance["answer"],
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
    metric_func = {"iou": iou, "precision@0.5": precision(0.5)}
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
    main()


