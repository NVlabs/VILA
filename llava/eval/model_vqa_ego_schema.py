import argparse
import itertools
import json
import os

import torch
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.eval.mmmu_utils.eval_utils import parse_choice
from llava.utils import distributed as dist
from llava.utils import io


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--question-file", type=str, required=True)
    parser.add_argument("--video-folder", type=str, required=True)
    parser.add_argument("--gt-answers-file", type=str)
    parser.add_argument("--split", type=str, choices=["validation", "test"], default="validation")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--output_name", type=str)
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
    instances = io.load(args.question_file)
    if args.split == "validation":
        answers = io.load(args.gt_answers_file)
        instances = [instance for instance in instances if instance["q_uid"] in answers]
    instances = instances[global_rank::world_size]

    # Run inference
    outputs = []
    for instance in tqdm(instances, disable=global_rank != 0):
        uuid = instance["q_uid"]
        video = llava.Video(os.path.join(args.video_folder, f"{uuid}.mp4"))

        question = instance["question"] + "\n"
        for i, c in enumerate(["A", "B", "C", "D", "E"]):
            question = question + c + ". " + instance[f"option {i}"] + "\n"
        question = "Watching the video and answer with the option's letter from the given choices directly." + question

        response = model.generate_content([video, question], generation_config=generation_config)
        choice = ord(parse_choice(response, ["A", "B", "C", "D", "E"])) - ord("A")

        output = {"id": uuid, "question": question, "pred": choice}
        if args.split == "validation":
            output["answer"] = answers[uuid]
        outputs.append(output)

    # Gather outputs and save
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = list(itertools.chain(*outputs))
    io.save(os.path.join(args.output_dir, f"{args.output_name}.json"), outputs)

    # Run evaluation
    if args.split == "validation":
        print(sum(output["pred"] == output["answer"] for output in outputs) / len(outputs) * 100)


if __name__ == "__main__":
    main()
