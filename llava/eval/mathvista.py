import argparse
import itertools
import json
import os

import torch
from datasets import load_dataset
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.eval.mathvista_utils.calculate_score import simple_calculate_score
from llava.eval.mathvista_utils.extract_answer import extract_answer
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--max-tiles", type=int)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--split", type=str, required=True)
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
    data = load_dataset("AI4Math/MathVista")[args.split]
    instances = data.select(range(dist.rank(), len(data), dist.size()))

    # Run inference
    outputs = {}
    for instance in tqdm(instances, disable=not dist.is_main()):
        image = instance.pop("decoded_image")
        question = instance["query"]
        response = model.generate_content([image, question], generation_config=generation_config)

        instance["extraction"] = extract_answer(response, instance)
        outputs[instance["pid"]] = instance

    # Gather and save outputs
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = dict(itertools.chain(*[output.items() for output in outputs]))
    io.save(os.path.join(args.output_dir, "outputs.jsonl"), outputs)

    # Run evaluation
    if args.split == "testmini":
        metrics = simple_calculate_score(outputs)
        io.save(os.path.join(args.output_dir, "metrics.json"), metrics)
        logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()


