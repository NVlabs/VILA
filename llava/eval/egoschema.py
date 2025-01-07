import argparse
import csv
import itertools
import json
import os

import torch
from pygments.lexer import default
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.data.builder import DATASETS
from llava.eval.mmmu_utils.eval_utils import parse_choice
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--num-video-frames", type=int, default=-1)
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

    if args.num_video_frames > 0:
        model.config.num_video_frames = args.num_video_frames

    # Set up generation config
    generation_config = model.default_generation_config
    if args.generation_config is not None:
        generation_config.update(**args.generation_config)

    data_path = DATASETS["egoschema"]["data_path"]
    answer_path = DATASETS["egoschema"]["answer_path"]
    video_dir = DATASETS["egoschema"]["media_dir"]

    # Load data and chunk it
    instances = io.load(data_path)
    if args.split == "val":
        answers = io.load(answer_path)
        instances = [instance for instance in instances if instance["q_uid"] in answers]
    instances = instances[dist.rank() :: dist.size()]

    # Run inference
    outputs = []
    for instance in tqdm(instances, disable=not dist.is_main()):
        uuid = instance["q_uid"]
        video = llava.Video(os.path.join(video_dir, f"{uuid}.mp4"))

        question = instance["question"] + "\n"
        for i, c in enumerate(["A", "B", "C", "D", "E"]):
            question = question + c + ". " + instance[f"option {i}"] + "\n"
        question = "Watching the video and answer with the option's letter from the given choices directly." + question

        response = model.generate_content([video, question], generation_config=generation_config)
        choice = ord(parse_choice(response, ["A", "B", "C", "D", "E"])) - ord("A")

        output = {"id": uuid, "question": question, "pred": choice}
        if args.split == "val":
            output["answer"] = answers[uuid]
        outputs.append(output)

    # Gather and save outputs
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = list(itertools.chain(*outputs))
    io.save(os.path.join(args.output_dir, "outputs.jsonl"), outputs)

    # Run evaluation
    if args.split == "val":
        metrics = {"accuracy": sum(output["pred"] == output["answer"] for output in outputs) / len(outputs)}
        io.save(os.path.join(args.output_dir, "metrics.json"), metrics)
        logger.info(f"Metrics: {metrics}")

    # Prepare for submission
    if args.split == "test":
        rows = []
        for output in outputs:
            q_uid, answer = output["id"], output["pred"]
            if str(answer) not in ["0", "1", "2", "3", "4"]:
                answer = 0
            rows.append({"q_uid": q_uid, "answer": answer})
        with open(os.path.join(args.output_dir, "submission.csv"), "w", newline="") as fd:
            writer = csv.DictWriter(fd, fieldnames=["q_uid", "answer"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()


