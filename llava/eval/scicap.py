import argparse
import itertools
import json
import os

import torch
from PIL import Image
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
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


def eval_main(predictions):

    print("Preparing references and predictions...")
    gts, res = prepare_references(predictions)

    print("Computing CIDEr score...")
    overall_cider_score, cider_scores = compute_cider(gts, res)

    print("Computing BLEU score...")
    overall_bleu_score, bleu_scores = compute_bleu(gts, res)

    # for unique_id, score in zip(gts.keys(), scores):
    #     print(f"Unique ID: {unique_id}, CIDEr score: {score}")
    print(f"Overall CIDEr score: {overall_cider_score}")
    print(f"Overall BLEU-4 score: {overall_bleu_score[3]}")  # BLEU-4 score


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

    args.image_folder = DATASETS["scicap_val"]["media_dir"]
    args.question_file = DATASETS["scicap_val"]["data_path"]

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

    # Gather outputs and save
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        eval_main(outputs[0])


if __name__ == "__main__":
    main()
