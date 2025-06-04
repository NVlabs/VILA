import argparse
import itertools
import json
import os

import torch
from PIL import Image
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from rouge import Rouge
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.data.builder import DATASETS
from llava.utils import distributed as dist
from llava.utils import io


def load_predictions(predictions_file):
    predictions = []
    with open(predictions_file) as f:
        for line in f:
            predictions.append(json.loads(line))
    return predictions


def prepare_references(predictions):
    gts = {}
    res = {}

    for pred in predictions:
        unique_id = f"{pred['question_id']}"
        text = pred["text"]
        gt_captions = pred["gt_caption"]

        if unique_id not in gts:
            gts[unique_id] = [gt_captions]
        else:
            gts[unique_id].extend(gt_captions)

        res[unique_id] = [text]

    return gts, res


def compute_cider(gts, res):
    scorer = Cider()
    score, scores = scorer.compute_score(gts, res)
    return score, scores


def compute_bleu(gts, res):
    scorer = Bleu(4)
    score, scores = scorer.compute_score(gts, res)
    return score, scores


def compute_rouge(gts, res):
    rouge = Rouge()
    rouge_scores = {"rouge-1": [], "rouge-2": [], "rouge-l": []}

    for unique_id in gts:
        gt_text = " ".join(gts[unique_id])  # Join ground truth captions
        res_text = " ".join(res[unique_id])  # Join predicted captions
        if res_text == "":
            print(f"Empty prediction for unique ID: {unique_id}")
            res_text = "dummy text"
        score = rouge.get_scores(res_text, gt_text)[0]
        rouge_scores["rouge-1"].append(score["rouge-1"]["f"])
        rouge_scores["rouge-2"].append(score["rouge-2"]["f"])
        rouge_scores["rouge-l"].append(score["rouge-l"]["f"])

    return rouge_scores


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

    question_file = DATASETS["nuscenes_val"]["data_path"]
    image_folder = DATASETS["nuscenes_val"]["media_dir"]

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
    model.config.image_aspect_ratio = "resize"

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
    instances = io.load(question_file)[global_rank::world_size]

    # Run inference
    gts = {}
    res = {}

    print("Preparing references and predictions...")

    for instance in tqdm(instances, disable=global_rank != 0):
        input = []
        text_parts = instance["text"].split("<image>\n")
        images = instance["image"] if isinstance(instance["image"], list) else [instance["image"]]
        for i, text_part in enumerate(text_parts):
            # Append text part if it's non-empty
            if text_part.strip():
                input.append(text_part.strip())
            # If there's a corresponding image, add it after the text part
            if i < len(images):
                image = Image.open(os.path.join(image_folder, images[i]))
                input.append(image)
        # for img in instance["image"]:
        #     image = Image.open(os.path.join(args.image_folder, img))
        #     input.append(image)
        # # image = Image.open(os.path.join(args.image_folder, instance["image"]))
        # question = instance["text"].replace("<image>\n", "")
        # input.append(question)

        unique_id = instance["question_id"]
        gt_caption = instance["answer"] if "answer" in instance else ""
        if unique_id not in gts:
            gts[unique_id] = [gt_caption]
        else:
            gts[unique_id].extend(gt_caption)

        response = model.generate_content(input, generation_config=generation_config)
        res[unique_id] = [response]

    print("Computing CIDEr score...")
    overall_cider_score, cider_scores = compute_cider(gts, res)

    print("Computing BLEU score...")
    overall_bleu_score, bleu_scores = compute_bleu(gts, res)

    print("Computing ROUGE score...")
    rouge_scores = compute_rouge(gts, res)

    # for unique_id, score in zip(gts.keys(), scores):
    #     print(f"Unique ID: {unique_id}, CIDEr score: {score}")
    print(f"Overall CIDEr score: {overall_cider_score}")
    print(f"Overall BLEU-4 score: {overall_bleu_score[0]}")  # BLEU-1 score
    print(f"Overall BLEU-4 score: {overall_bleu_score[1]}")  # BLEU-2 score
    print(f"Overall BLEU-4 score: {overall_bleu_score[2]}")  # BLEU-3 score
    print(f"Overall BLEU-4 score: {overall_bleu_score[3]}")  # BLEU-4 score
    print(f"Overall ROUGE-1 score (F1): {sum(rouge_scores['rouge-1']) / len(rouge_scores['rouge-1'])}")
    print(f"Overall ROUGE-2 score (F1): {sum(rouge_scores['rouge-2']) / len(rouge_scores['rouge-2'])}")
    print(f"Overall ROUGE-L score (F1): {sum(rouge_scores['rouge-l']) / len(rouge_scores['rouge-l'])}")


if __name__ == "__main__":
    main()
