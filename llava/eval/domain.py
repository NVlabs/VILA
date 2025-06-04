import argparse
import glob
import itertools
import json
import os
from collections import defaultdict

import torch
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.data.builder import DATASETS
from llava.eval.mmmu_utils.eval_utils import parse_choice
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger


def compute_metrics(outputs, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Compute overall accuracy
    total_correct = sum(output["pred"] == output["answer"] for output in outputs)
    total_items = len(outputs)
    overall_accuracy = total_correct / total_items if total_items > 0 else 0.0

    # Compute category-wise accuracy
    category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for output in outputs:
        category = output.get("category", "Other")  # Handle missing categories
        category_stats[category]["correct"] += int(output["pred"] == output["answer"])
        category_stats[category]["total"] += 1

    # Format results
    metrics = {"overall_accuracy": overall_accuracy}
    category_metrics = {}

    print("\nEvaluation Metrics:")
    print(f"{'Category':<30}{'Accuracy':<15}{'QA Count':<10}")
    print("=" * 55)

    for category, stats in sorted(category_stats.items(), key=lambda x: -x[1]["total"]):  # Sort by count
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        category_metrics[category] = {"accuracy": accuracy, "count": stats["total"]}

        print(f"{category:<30}{accuracy:<15.4f}{stats['total']:<10}")
    overall_acc = "Overall"
    print(f"{overall_acc:<30}{overall_accuracy:<15.4f}{total_items:<10}")
    # Save metrics
    metrics["category_metrics"] = category_metrics

    output_path = os.path.join(output_dir, "metrics.json")
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nMetrics saved to {output_path}")

    return metrics


def load_and_merge_mapping(directory):
    merged_mapping = {}

    for file in glob.glob(os.path.join(directory, "*.json")):
        try:
            with open(file) as f:
                data = json.load(f)

                if isinstance(data, dict):
                    merged_mapping.update(data)  # Merge dictionaries
                else:
                    print(f"Skipping {file}: Expected a dictionary format")

        except (json.JSONDecodeError, OSError) as e:
            print(f"Error loading JSON from {file}: {e}")

    return merged_mapping


def load_and_merge_jsons(directory):
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


def get_category(vid_id, mapping):
    """Find the corresponding category based on vid_id prefix."""
    for key in mapping:
        key_base = os.path.splitext(key)[0]  # Remove extension from key
        if vid_id.startswith(key_base):
            return mapping[key]
    return "Other"


def load_results(results_dir):
    results = []
    if os.path.exists(results_dir):
        for root, _, files in os.walk(results_dir):
            for file in files:
                if "json" in file:
                    print(os.path.join(root, file))
                    result = json.load(open(os.path.join(root, file)))
                    results.append(result)
    else:
        os.system("mkdir -p %s" % results_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--num-video-frames", type=int, default=-1)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--answer-path", type=str)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-tiles", type=int, default=12)
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()

    # Set up distributed environment
    dist.init()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])

    # TODO(zhijianl): This will be removed in the future
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    # Load model
    model = llava.load(args.model_path, model_base=args.model_base, devices=devices)

    context_length = args.num_video_frames * 512 * args.max_tiles
    model.config.model_max_length = context_length
    model.config.tokenizer_model_max_length = context_length
    model.llm.config.model_max_length = context_length
    model.llm.config.tokenizer_model_max_length = context_length
    model.tokenizer.model_max_length = context_length

    if args.num_video_frames > 0:
        model.config.num_video_frames = args.num_video_frames

    # Set up generation config
    generation_config = model.default_generation_config
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    data_path = DATASETS[args.dataset]["data_path"]
    video_dir = DATASETS[args.dataset]["video_dir"]

    # Load category mapping
    category_mapping = load_and_merge_mapping(os.path.join(data_path, "category_mapping"))

    # Load and merge all annotations
    instances = load_and_merge_jsons(os.path.join(data_path, "metric_jsons"))
    # Load data and chunk it
    # instances = json.load(open(data_path))
    instances = instances[dist.rank() :: dist.size()]

    results_dir = os.path.join(args.output_dir, "cache")
    os.makedirs(results_dir, exist_ok=True)
    # Run inference
    # TODO: Fix this
    outputs_ids = []
    outputs = []
    # outputs = load_results(results_dir)
    # outputs_ids = [output["id"] for output in outputs]

    for instance in tqdm(instances, disable=not dist.is_main()):
        uuid = instance["q_uid"]
        if uuid in outputs_ids:
            print("Finished %s" % uuid)
            continue

        video_path = instance["q_uid"]
        video = llava.Video(os.path.join(video_dir, video_path))

        question = instance["question"] + "\n"
        option_labels = ["A", "B", "C", "D"]  # Labels for options

        for i, c in enumerate(option_labels[: len(instance["options"])]):  # Only iterate over available options
            question = question + c + ". " + str(instance["options"][i]) + "\n"

        question = question + "Answer with the option's letter from the given choices directly."

        response = model.generate_content([video, question], generation_config=generation_config)
        response = response[0]
        if not response in "ABCD":
            continue

        choice = ord(parse_choice(response, ["A", "B", "C", "D"])) - ord("A")

        output = {"id": uuid, "question": question, "pred": choice}
        output["answer"] = ["A", "B", "C", "D"].index(instance["gt_option"])
        # Assign category based on vid_id prefix
        output["category"] = get_category(uuid, category_mapping)
        outputs.append(output)
        json.dump(output, open(os.path.join(results_dir, f"{os.path.basename(uuid)}.json"), "w"))

    # Gather and save outputs
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = list(itertools.chain(*outputs))
    io.save(os.path.join(args.output_dir, "outputs.jsonl"), outputs)

    # Run evaluation
    metrics = compute_metrics(outputs, args.output_dir)
    io.save(os.path.join(args.output_dir, "metrics.json"), metrics)


if __name__ == "__main__":
    main()
