import argparse
import itertools
import json
import os
import re

import torch
from PIL import Image
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.data.builder import DATASETS
from llava.utils import distributed as dist
from llava.utils import io

_LARGEST_COUNT = 15


def load_jsonl(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def _number_word_to_numeral(text: str) -> str:
    number_words = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "thirteen": "13",
        "fourteen": "14",
        "fifteen": "15",
        "sixteen": "16",
        "seventeen": "17",
        "eighteen": "18",
        "nineteen": "19",
        "twenty": "20",
    }

    # Check if text is a number word
    if text.lower() in number_words:
        return number_words[text.lower()]

    # Check if text is already a numeral
    if text.isdigit():
        return text

    # Extract numeric part from the text
    match = re.search(r"\d+", text)
    if match:
        return match.group(0)

    # Extract number word from the text
    words = text.lower().split()
    for word in words:
        if word in number_words:
            return number_words[word]

    # If no numeric part is found, return the original text (can be handled later as needed)
    return text


def eval(predictions, annotations):
    accuracies_by_type = {"all": [], "simple": [], "complex": []}
    accuracies_by_type.update({f"count_{i}": [] for i in range(_LARGEST_COUNT + 1)})

    annotation_dict = {entry["question_id"]: entry for entry in annotations}

    for prediction in predictions:
        question_id = prediction["question_id"]
        answer = prediction["text"].strip()
        answer = _number_word_to_numeral(answer)

        if question_id in annotation_dict:
            gt_entry = annotation_dict[question_id]
            gt = _number_word_to_numeral(str(gt_entry["answer"]))

            accuracies_by_type["all"].append(float(answer == gt))

            if "issimple" in gt_entry:
                if gt_entry["issimple"] == 1:
                    accuracies_by_type["simple"].append(float(answer == gt))
                elif gt_entry["issimple"] == 0:
                    accuracies_by_type["complex"].append(float(answer == gt))

            if f"count_{gt}" in accuracies_by_type:
                accuracies_by_type[f"count_{gt}"].append(float(answer == gt))

    sum_accs = {k: sum(v) for k, v in accuracies_by_type.items()}
    num_accs = {k: len(v) for k, v in accuracies_by_type.items()}

    metrics = {}
    if num_accs["all"]:
        metrics["acc"] = sum_accs["all"] / num_accs["all"]
        metrics["num"] = num_accs["all"]

    for key in sum_accs.keys():
        if key != "all" and num_accs[key]:
            metrics[f"acc/{key}"] = sum_accs[key] / num_accs[key]
            metrics[f"num/{key}"] = num_accs[key]

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--max-tiles", type=int, default=12)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--num-chunks", type=int)
    parser.add_argument("--chunk-idx", type=int)
    args = parser.parse_args()

    args.question_file = DATASETS["tallyqa_val"]["data_path"]
    args.image_folder = DATASETS["tallyqa_val"]["media_dir"]

    # Set up distributed environment
    if args.num_chunks is None:
        dist.init()
        world_size, global_rank = dist.size(), dist.rank()
        devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
        torch.cuda.set_device(devices[0])
    else:
        world_size, global_rank = args.num_chunks, args.chunk_idx

    # TODO(zhijianl): This will be removed in the future
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    # Load model
    model = llava.load(args.model_path, model_base=args.model_base, devices=devices)
    model.config.min_tiles = 1
    model.config.max_tiles = args.max_tiles
    model.llm.config.min_tiles = 1
    model.llm.config.max_tiles = args.max_tiles
    model.config.image_aspect_ratio = "dynamic"

    if args.max_tiles > 12:
        context_length = int(args.max_tiles / 12.0 * 4096)
        model.config.model_max_length = context_length
        model.config.tokenizer_model_max_length = context_length
        model.llm.config.model_max_length = context_length
        model.llm.config.tokenizer_model_max_length = context_length

    # Set up generation config
    generation_config = model.default_generation_config
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    # Load data and chunk it
    instances = io.load(args.question_file)[global_rank::world_size]

    # Run inference
    outputs = []
    for instance in tqdm(instances, disable=global_rank != 0):
        image = Image.open(os.path.join(args.image_folder, instance["image"]))
        question = instance["text"] if "text" in instance else instance["question"]
        response = model.generate_content([image, question], generation_config=generation_config)
        outputs.append(
            {
                "question_id": instance["question_id"],
                "prompt": question,
                "text": response,
                "gt_caption": instance["answer"] if "answer" in instance else "",
            }
        )

    annotations = load_jsonl(args.question_file)[0]

    # Gather outputs and save
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        metrics = eval(outputs[0], annotations)
        for key, value in metrics.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
