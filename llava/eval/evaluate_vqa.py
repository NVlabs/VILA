import argparse
import itertools
import json
import os

import torch
from PIL import Image
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.utils import distributed as dist
from llava.utils import io


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
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
    instances = io.load(args.data_path)[global_rank::world_size]

    # Run inference
    outputs = []
    for instance in tqdm(instances, disable=global_rank != 0):
        question = instance["question"]
        image = Image.open(os.path.join(args.image_folder, instance["image"]))
        response = model.generate_content([image, question], generation_config=generation_config)
        if args.dataset.startswith("ocrvqa"):
            output = {"questionId": instance["question_id"], "answer": response, "annotation": instance["answer"]}
        elif args.dataset.startswith("ai2diagram"):
            output = {"image": instance["question_id"], "answer": response, "annotation": instance["answer"]}
        elif args.dataset.startswith("chartqa"):
            output = {"answer": response, "annotation": instance["answer"]}
        elif args.dataset.startswith("docvqa"):
            output = {"questionId": instance["question_id"], "answer": response}
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        outputs.append(output)

    # Gather outputs and save
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = list(itertools.chain(*outputs))
    io.save(args.answers_file, outputs)


if __name__ == "__main__":
    main()
