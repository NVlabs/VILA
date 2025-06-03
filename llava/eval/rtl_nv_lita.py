import argparse
import glob
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


def compute_metrics(outputs, output_dir, metric_func):
    """
    Computes and prints evaluation metrics in a structured format and saves results to a JSON file.

    Args:
        outputs (list): List of dictionaries containing evaluation results.
        output_dir (str): Directory where the metrics JSON file will be saved.
        metric_func (dict): Dictionary of metric functions (e.g., {"iou": iou, "precision@0.5": precision(0.5)}).

    Returns:
        dict: Computed overall and category-wise metrics.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize storage for metrics
    metrics = {name: defaultdict(list) for name in metric_func}
    category_metrics = defaultdict(lambda: defaultdict(list))

    # Compute metrics per output and category
    for output in outputs:
        category = output.get("category", "Other")  # Handle missing categories
        for name in metrics:
            score = metric_func[name](
                parse_timestamps(output["output"], output["duration"], strict=False),
                parse_timestamps(output["target"], output["duration"], strict=True),
            )
            metrics[name][output["vid"]].append(score)
            category_metrics[category][name].append(score)

    # Compute overall and per-category scores
    final_metrics = {}
    category_final_metrics = {}

    print("\nEvaluation Metrics:")
    print(f"{'Category':<30}{'IOU':<15}{'Precision@0.5':<15}{'Count':<10}")
    print("=" * 75)

    for category, metric_dict in sorted(category_metrics.items(), key=lambda x: -len(next(iter(x[1].values()), []))):
        category_iou = np.mean(metric_dict["iou"]) if metric_dict["iou"] else 0.0
        category_precision = np.mean(metric_dict["precision@0.5"]) if metric_dict["precision@0.5"] else 0.0
        count = len(metric_dict["iou"])

        category_final_metrics[category] = {
            "iou": category_iou,
            "precision@0.5": category_precision,
            "count": count,
        }

        print(f"{category:<30}{category_iou:<15.4f}{category_precision:<15.4f}{count:<10}")

    # Compute overall metrics
    overall_iou = np.mean([iou_value for vid in metrics["iou"] for iou_value in metrics["iou"][vid]])
    overall_precision = np.mean(
        [precision_value for vid in metrics["precision@0.5"] for precision_value in metrics["precision@0.5"][vid]]
    )
    total_items = sum(len(metrics["iou"][vid]) for vid in metrics["iou"])

    print(f"{'Overall':<30}{overall_iou:<15.4f}{overall_precision:<15.4f}{total_items:<10}")

    final_metrics["overall"] = {
        "iou": overall_iou,
        "precision@0.5": overall_precision,
        "count": total_items,
    }
    final_metrics["category_metrics"] = category_final_metrics

    # Save metrics to JSON
    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(final_metrics, f, indent=4)

    print(f"\nMetrics saved to {output_path}")

    return final_metrics


def load_and_merge_jsons(directory):
    """Load and merge all JSON files in a directory."""
    merged_data = []
    for file in glob.glob(os.path.join(directory, "*.json")):
        with open(file) as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    merged_data.extend(data)  # Append if it's a list
                elif isinstance(data, dict):
                    merged_data.append(data)  # Append dict as an entry
            except json.JSONDecodeError:
                print(f"Error loading JSON from {file}")
    return merged_data


def load_mapping_json(directory):
    """Load and merge all mapping JSON files into a single dictionary."""
    merged_mapping = {}
    for file in glob.glob(os.path.join(directory, "*.json")):
        with open(file) as f:
            try:
                data = json.load(f)
                if isinstance(data, dict):
                    merged_mapping.update(data)  # Merge dictionary
            except json.JSONDecodeError:
                print(f"Error loading JSON from {file}")
    return merged_mapping


def get_category(vid_id, mapping):
    """Find the corresponding category based on vid_id prefix after removing extensions."""
    for key in mapping:
        key_base = os.path.splitext(key)[0]  # Remove extension from key
        if vid_id.startswith(key_base):
            return mapping[key]
    return "Other"


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
    """Extract timestamps from text."""
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
    """Compute Intersection over Union (IoU) for timestamps."""
    i = max(min(s1[1], s2[1]) - max(s1[0], s2[0]), 0)
    u = max(s1[1] - s1[0], 0) + max(s2[1] - s2[0], 0) - i
    return i / u if u > 0 else 0


def precision(threshold: float):
    """Return precision function based on IoU threshold."""

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

    # Load data
    data_path = DATASETS[args.dataset]["data_path"]
    video_dir = DATASETS[args.dataset]["video_dir"]

    instances = load_and_merge_jsons(os.path.join(data_path, "metric_jsons"))[dist.rank() :: dist.size()]
    mapping = load_mapping_json(os.path.join(data_path, "category_mapping"))

    # Run inference
    outputs = []
    for instance in tqdm(instances, disable=not dist.is_main()):
        vid = instance["vid"]
        question = instance["question"]

        video = Video(os.path.join(video_dir, vid + ".mp4"))
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
            "category": get_category(vid, mapping),
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

    # Compute and print metrics
    metrics = compute_metrics(outputs, args.output_dir, metric_func)

    # Log results (optional)
    logger.info(f"Final Metrics: {metrics}")


if __name__ == "__main__":
    main()
