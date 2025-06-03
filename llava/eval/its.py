import argparse
import itertools
import json
import os
import re
import time
from collections import defaultdict
from pprint import pprint
from typing import Tuple

import numpy as np
import torch
from tqdm import tqdm

import llava
from llava.data import DATASETS
from llava.media import Image, Video
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes"}:
        return True
    elif value.lower() in {"false", "0", "no"}:
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


# Regex patterns for different tasks
TASK_PATTERNS = {
    "car_count": [r"(\d+)\s+car(?:s)?", "Count the total number of cars in the scene"],
    "lane_count": [r"total.*?lanes.*?(\d+)", "What is the total number of lanes at the intersection"],
    "car_lane_association": [r"(?i)lane\s+(\d{1,2})", "Which lane is the car with numeric ID"],
    "car_lane_binary_yes_no": [r"(?i)\b(yes|no)\b", "Is the car with numeric ID"],
    "lane_car_ids": [r"\b\d+\b", "Which cars are in lane"],
}
TASK_PATTERNS_ZS = {
    "car_count": [r"(\d+)\s+car(?:s)?", "Count the total number of cars in the scene"],
    "lane_count": [
        r"total.*?lanes.*?(\d+)",
        "What is the total number of lanes at the intersection",
    ],  # Change these 2 also if needed
    "car_lane_association": [r"(\d+)", "Which lane is the car with numeric ID"],
    "car_lane_binary_yes_no": [r"(?i)\b(yes|no)\b", "Is the car with numeric ID"],
    "lane_car_ids": [r"\b\d+\b", "Which cars are in lane"],
}

zero_shot_question_mapping = {
    "car_lane_association": " Just provide the exact lane number between 1 to n. Do not add any extra text, explanations, or comments. Only provide a single number.",
    "lane_car_ids": " You must only provide the exact numeric IDs of the cars from 1 to n. Do not add any extra text, explanations, or comments. Only provide numeric IDs of the cars from the overlayed image.",
}


def extract_count_zero_shot(text):
    """
    Extract count for cars or lanes from a given text.

    Args:
        text (str): The input text to parse.
        entity (str): The entity to extract the count for ("car" or "lane").

    Returns:
        int or None: The extracted count, or None if not found.
    """
    # Extended regex to support numbers 1-15 in both numeric and written formats
    pattern = r"(?i)(?:there\s+are\s+|there\s+are\s+a\s+total\s+of\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen)\s+(?:car|lane)"
    match = re.search(pattern, text)
    if match:
        count = match.group(1)
        # Convert written numbers to digits if necessary
        written_to_digit = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
        }
        return int(written_to_digit[count]) if count.isalpha() else int(count)
    return None


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--is_zero_shot", type=str_to_bool, default=False, help="Specify True or False for zero-shot mode"
    )
    return parser.parse_args()


def setup_environment():
    """Set up distributed environment."""
    dist.init()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])
    return devices


def load_model(args, devices):
    """Load the model."""
    model = llava.load(args.model_path, model_base=args.model_base, devices=devices)
    generation_config = model.default_generation_config
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)
    return model, generation_config


def load_dataset(args):
    """Load dataset and distribute instances across ranks."""
    data_path = DATASETS[args.dataset]["data_path"]
    image_dir = DATASETS[args.dataset]["image_dir"]
    instances = io.load(data_path)[dist.rank() :: dist.size()]
    return instances, image_dir


def process_instance(instance, image_dir, model, generation_config, task, task_ques, is_zero_shot=False):
    """Process a single instance for a given task."""
    results = []
    prompt = []
    if "image" in instance.keys() and task in ["car_count", "lane_count"]:
        image = instance["image"]
        question = instance["conversations"][0]["value"]
        # TODO: CHECK IF FOR ZERO_SHOT NEED TO ADD NEW QUESTION PROMPT
        ground_truth = instance["conversations"][1]["value"]
        image = Image(os.path.join(image_dir, image))
        if question.lower().replace("<image>\n", "").startswith(task_ques.lower()):
            prompt_with_question = [image, question]
            try:
                with torch.no_grad():
                    response = model.generate_content(prompt_with_question, generation_config=generation_config)
                    results.append(
                        {
                            "task": task,
                            "question": question,
                            "generated_response": response,
                            "ground_truth": ground_truth,
                        }
                    )
            except RuntimeError as e:
                print(f"Error generating response for task '{task}': {e}")

    elif "images" in instance.keys() and task in ["car_lane_association", "car_lane_binary_yes_no", "lane_car_ids"]:
        images = instance["images"]
        conversations = instance["conversations"]
        for i in range(len(conversations) - 1):  # Iterate over the conversations
            if conversations[i]["value"].lower().replace("<image>\n", "").startswith(task_ques.lower()):
                prompt = []
                if conversations[i]["from"] == "human":
                    for img_path in images:
                        image = Image(os.path.join(image_dir, img_path["path"]))
                        prompt.append(image)

                    question = conversations[i]["value"]
                    ground_truth = None

                    # Look for the next GPT response
                    if i + 1 < len(conversations) and conversations[i + 1]["from"] == "gpt":
                        ground_truth = conversations[i + 1]["value"]

                    # Prepend the required context to the question if not already present
                    question_prepend = (
                        "The first image is the original, and the second is an overlay. "
                        "Bright numeric IDs are labeled at the center of certain visual objects in the second image. "
                    )
                    question_ = question
                    if question_prepend not in question:
                        question_ = question_prepend + question
                    if is_zero_shot and task in zero_shot_question_mapping.keys():
                        question_ += zero_shot_question_mapping[task]
                    # Add question to the prompt
                    prompt_with_question = prompt + [question_]

                    # Generate response
                    with torch.no_grad():
                        response = model.generate_content(prompt_with_question, generation_config=generation_config)
                        # Log or store response and ground truth
                        # Store response and ground truth
                        results.append(
                            {
                                "task": task,
                                "question": question,
                                "generated_response": response,
                                "ground_truth": ground_truth,
                            }
                        )
    return results


def evaluate_outputs(outputs, is_zero_shot):
    """Evaluate outputs for each task."""
    results = {task: {"exact": 0, "tolerance_1": 0, "tolerance_2": 0, "total": 0} for task in TASK_PATTERNS}
    results["lane_car_ids"] = {"precision": 0, "recall": 0, "f1_score": 0, "total": 0, "accuracy": 0}

    for output in outputs:
        task = output["task"]
        ground_truth = output["ground_truth"]
        response = output["generated_response"]
        pattern, _ = TASK_PATTERNS[task]

        if task in ["lane_car_ids"]:
            # Extract numeric IDs
            gt_match = sorted(re.findall(pattern, ground_truth))
            response_match = sorted(re.findall(pattern, response))
            # Handle cases when no IDs are found
            if not gt_match or not response_match:
                # If both are empty, consider it a perfect match case
                continue
            # Convert to sets
            gt_set = set(gt_match)
            response_set = set(response_match)

            # Compute metrics
            true_positives = len(gt_set & response_set)  # Intersection
            false_positives = len(response_set - gt_set)  # In response but not in GT
            false_negatives = len(gt_set - response_set)  # In GT but not in response

            precision = (
                true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            )
            recall = (
                true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            )
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Compute accuracy
            total_unique_ids = len(gt_set | response_set)
            accuracy = true_positives / total_unique_ids if total_unique_ids > 0 else 0

            # Aggregate results
            results["lane_car_ids"]["precision"] += precision
            results["lane_car_ids"]["recall"] += recall
            results["lane_car_ids"]["f1_score"] += f1_score
            results["lane_car_ids"]["accuracy"] += accuracy
            results["lane_car_ids"]["total"] += 1
        else:
            # Extract counts
            gt_match = re.search(pattern, ground_truth)
            if is_zero_shot:
                pattern, _ = TASK_PATTERNS_ZS[task]
            response_match = re.search(pattern, response)
            if task in ["car_lane_binary_yes_no"]:
                gt_count = gt_match.group(1) if gt_match else None
                response_count = response_match.group(1) if response_match else None
                results[task]["total"] += 1
                if gt_count == response_count:
                    results[task]["exact"] += 1
                results[task]["tolerance_1"] = None
                results[task]["tolerance_2"] = None

            else:
                gt_count = int(gt_match.group(1)) if gt_match else None
                response_count = int(response_match.group(1)) if response_match else None
                if not response_match:
                    response_count = extract_count_zero_shot(response)
                if gt_count is None or response_count is None:
                    continue
                results[task]["total"] += 1
                if gt_count == response_count:
                    results[task]["exact"] += 1
                if abs(response_count - gt_count) <= 1:
                    results[task]["tolerance_1"] += 1
                if abs(response_count - gt_count) <= 2:
                    results[task]["tolerance_2"] += 1

    # Calculate final accuracies for each task
    final_metrics = {}
    for task, metrics in results.items():
        total = metrics["total"]
        # Compute final average metrics for 'lane_car_ids'
        if total == 0:
            continue
        if task == "lane_car_ids":
            final_metrics[task] = {
                "precision": results["lane_car_ids"]["precision"] / results["lane_car_ids"]["total"] * 100,
                "recall": results["lane_car_ids"]["recall"] / results["lane_car_ids"]["total"] * 100,
                "f1_score": results["lane_car_ids"]["f1_score"] / results["lane_car_ids"]["total"] * 100,
                "accuracy": results["lane_car_ids"]["accuracy"] / results["lane_car_ids"]["total"] * 100,
            }
        else:
            exact_accuracy = metrics["exact"] / total * 100
            if metrics["tolerance_1"]:
                tolerance_1_accuracy = metrics["tolerance_1"] / total * 100
                tolerance_2_accuracy = metrics["tolerance_2"] / total * 100
            else:
                tolerance_1_accuracy = "-"
                tolerance_2_accuracy = "-"
            final_metrics[task] = {
                "exact_accuracy": exact_accuracy,
                "tolerance_1_accuracy": tolerance_1_accuracy,
                "tolerance_2_accuracy": tolerance_2_accuracy,
            }

    return results, final_metrics


def main():
    args = parse_args()
    devices = setup_environment()
    instances, image_dir = load_dataset(args)
    model, generation_config = load_model(args, devices)

    instances = instances
    outputs = []
    c = 0
    for instance in tqdm(instances, disable=not dist.is_main()):
        c += 1
        for task, [pattern, task_ques] in TASK_PATTERNS.items():
            output = process_instance(instance, image_dir, model, generation_config, task, task_ques, args.is_zero_shot)
            if output:
                outputs.extend(output)

    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = list(itertools.chain(*outputs))
    # Save outputs
    output_path = os.path.join(args.output_dir, "outputs.jsonl")
    io.save(output_path, outputs)

    # Evaluate
    results, final_metrics = evaluate_outputs(outputs, args.is_zero_shot)

    # Save the final metrics to a JSON file
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as metrics_file:
        json.dump(final_metrics, metrics_file, indent=2)
    # Save the final count to a JSON file
    count_path = os.path.join(args.output_dir, "metrics_count.json")
    with open(count_path, "w") as count_file:
        json.dump(results, count_file, indent=2)

    # Print the final metrics
    for task, metrics in final_metrics.items():
        print(f"Task: {task}")
        print(f"Metrics:")
        pprint(metrics)


if __name__ == "__main__":
    main()
