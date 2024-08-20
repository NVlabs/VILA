# This file is originated from the official MMMU codebase:
# https://github.com/MMMU-Benchmark/MMMU
import itertools
import json
import random
from argparse import ArgumentParser

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.eval.mmmu_utils.data_utils import CAT_SHORT2LONG, construct_prompt, process_single_sample
from llava.eval.mmmu_utils.eval_utils import parse_choice
from llava.utils import distributed as dist
from llava.utils import io

CONFIG = {
    "task_instructions": "",
    "multi_choice_example_format": """{}

{}

Answer with the option's letter from the given choices directly.""",
    "short_ans_example_format": """{}

Answer the question using a single word or phrase.""",
    "temperature": 0,
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--num-chunks", type=int)
    parser.add_argument("--chunk-idx", type=int)
    args = parser.parse_args()

    # TODO(zhijianl): Is this necessary?
    set_seed(42)

    # Set up distributed environment
    if args.num_chunks is None:
        dist.init()
        torch.cuda.set_device(dist.local_rank())
        world_size, global_rank = dist.size(), dist.rank()
    else:
        world_size, global_rank = args.num_chunks, args.chunk_idx

    # TODO(zhijianl): This will be removed in the future
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    # Load model
    model = llava.load(args.model_path, model_base=args.model_base)

    # Set up generation config
    generation_config = model.generation_config
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    # Load data and chunk it
    data = concatenate_datasets(
        [load_dataset(args.data_path, name, split=args.split) for name in CAT_SHORT2LONG.values()]
    )
    instances = data.select(range(global_rank, len(data), world_size))

    # Run inference
    outputs = {}
    for instance in tqdm(instances, disable=global_rank != 0):
        instance = process_single_sample(instance)
        instance = construct_prompt(instance, CONFIG)

        images = instance["image"]
        prompt = instance["final_input_prompt"].replace("<image>", "").strip()

        response = model.generate_content(images + [prompt], generation_config=generation_config)
        if instance["question_type"] == "multiple-choice":
            response = parse_choice(response, instance["all_choices"], instance["index2ans"])
        outputs[instance["id"]] = response

    # Gather outputs and save
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = dict(itertools.chain(*[output.items() for output in outputs]))
    io.save(args.output_path, outputs)


if __name__ == "__main__":
    main()
