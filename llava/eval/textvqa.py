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
from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger


def prompt_processor(prompt):
    if prompt.startswith("OCR tokens: "):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif "Reference OCR token: " in prompt and len(prompt.split("\n")) == 3:
        if prompt.startswith("Reference OCR token:"):
            question = prompt.split("\n")[1]
        else:
            question = prompt.split("\n")[0]
    elif len(prompt.split("\n")) == 2:
        question = prompt.split("\n")[0]
    else:
        assert False

    return question.lower()


def eval_single(outputs, answers):
    answers = answers["data"]
    answers = {(annotation["image_id"], annotation["question"].lower()): annotation for annotation in answers}

    pred_list = []
    for result in outputs:
        annotation = answers[(result["question_id"], prompt_processor(result["prompt"]))]
        pred_list.append(
            {
                "pred_answer": result["text"],
                "gt_answers": annotation["answers"],
            }
        )

    evaluator = TextVQAAccuracyEvaluator()
    return {"accuracy": evaluator.eval_pred_list(pred_list)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--max-tiles", type=int)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--answer-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()

    # Set up distributed environment
    dist.init()
    devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
    torch.cuda.set_device(devices[0])

    # TODO(zhijianl): This will be removed in the future
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    # Load model
    model = llava.load(args.model_path, model_base=args.model_base, devices=devices)
    model.config.min_tiles = 1
    model.config.max_tiles = args.max_tiles
    model.llm.config.min_tiles = 1
    model.llm.config.max_tiles = args.max_tiles

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
    instances = io.load(args.data_path)[dist.rank() :: dist.size()]

    # Run inference
    outputs = []
    for instance in tqdm(instances, disable=not dist.is_main()):
        image = Image.open(os.path.join(args.image_dir, instance["image"]))
        question = instance["text"]
        response = model.generate_content([image, question], generation_config=generation_config)
        outputs.append({"question_id": instance["question_id"], "prompt": question, "text": response})

    # Gather and save outputs
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = list(itertools.chain(*outputs))
    io.save(os.path.join(args.output_dir, "outputs.jsonl"), outputs)

    # Run evaluation
    answers = io.load(args.answer_path)
    metrics = eval_single(outputs, answers)
    io.save(os.path.join(args.output_dir, "metrics.json"), metrics)
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()


