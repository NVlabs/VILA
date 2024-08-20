import argparse
import itertools
import json

import torch
from datasets import load_dataset
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.eval.mathvista_utils.extract_answer import extract_answer
from llava.utils import distributed as dist
from llava.utils import io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--num-chunks", type=int)
    parser.add_argument("--chunk-idx", type=int)
    args = parser.parse_args()

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
    data = load_dataset("AI4Math/MathVista")[args.split]
    instances = data.select(range(global_rank, len(data), world_size))

    # Run inference
    outputs = {}
    for instance in tqdm(instances, disable=global_rank != 0):
        image = instance["decoded_image"]
        question = instance["query"]
        response = model.generate_content([image, question], generation_config=generation_config)

        del instance["decoded_image"]
        instance["extraction"] = extract_answer(response, instance)
        outputs[instance["pid"]] = instance

    # Gather outputs and save
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = dict(itertools.chain(*[output.items() for output in outputs]))
    io.save(args.answers_file, outputs)


if __name__ == "__main__":
    main()
