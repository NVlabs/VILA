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
    parser.add_argument("--max-tiles", type=int, default=12)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--answers-file", type=str, required=True)
    parser.add_argument("--num-chunks", type=int)
    parser.add_argument("--chunk-idx", type=int)
    args = parser.parse_args()

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
        input = []
        text_parts = instance["text"].split("<image>\n")
        images = instance["image"] if isinstance(instance["image"], list) else [instance["image"]]
        for i, text_part in enumerate(text_parts):
            # Append text part if it's non-empty
            if text_part.strip():
                input.append(text_part.strip())
            # If there's a corresponding image, add it after the text part
            if i < len(images):
                image = Image.open(os.path.join(args.image_folder, images[i]))
                input.append(image)
        # for img in instance["image"]:
        #     image = Image.open(os.path.join(args.image_folder, img))
        #     input.append(image)
        # # image = Image.open(os.path.join(args.image_folder, instance["image"]))
        # question = instance["text"].replace("<image>\n", "")
        # input.append(question)
        response = model.generate_content(input, generation_config=generation_config)
        outputs.append(
            {
                "question_id": instance["question_id"],
                "prompt": instance["text"],
                "text": response,
                "gt_caption": instance["answer"] if "answer" in instance else "",
            }
        )

    # Gather outputs and save
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = list(itertools.chain(*outputs))
    io.save(args.answers_file, outputs)


if __name__ == "__main__":
    main()
