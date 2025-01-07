import argparse
import csv
import itertools
import json
import os

import torch
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.data.builder import DATASETS
from llava.eval.mmmu_utils.eval_utils import parse_choice
from llava.utils import distributed as dist
from llava.utils import io
from llava.utils.logging import logger


def load_results(results_dir):
    results = []
    if os.path.exists(results_dir):
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if "json" in file:
                    result = json.load(open(os.path.join(root, file)))
                    results.append(result)
    else:
        os.system("mkdir -p %s" % results_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--conv-mode", type=str, required=True)
    parser.add_argument("--num-video-frames", type=int, default=-1)
    parser.add_argument("--generation-config", type=json.loads)
    parser.add_argument("--answer-path", type=str)
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

    # Load data and chunk it
    instances = json.load(open(DATASETS["vnbench"]["data_path"]))
    instances = instances[dist.rank() :: dist.size()]

    results_dir = os.path.join(args.output_dir, "cache")
    # Run inference
    outputs = load_results(results_dir)

    outputs_ids = [output["id"] for output in outputs]
    video_dir = DATASETS["vnbench"]["media_dir"]

    for instance in tqdm(instances, disable=not dist.is_main()):
        uuid = instance["video"].split("/")[-1]
        if uuid in outputs_ids:
            print("Finished %s" % uuid)
            continue

        video_path = instance["video"]
        video = llava.Video(os.path.join(video_dir, video_path))

        question = instance["question"] + "\n"
        for i, c in enumerate(["A", "B", "C", "D"]):
            question = question + c + ". " + str(instance["options"][i]) + "\n"
        question = question + "Answer with the option's letter from the given choices directly."

        response = model.generate_content([video, question], generation_config=generation_config)
        response = response[0]
        if not response in "ABCD":
            continue

        print("response", response)
        choice = ord(parse_choice(response, ["A", "B", "C", "D"])) - ord("A")

        output = {"id": uuid, "question": question, "pred": choice}
        output["answer"] = ["A", "B", "C", "D"].index(instance["gt_option"])
        outputs.append(output)
        json.dump(output, open(os.path.join(results_dir, "%s.json") % uuid, "w"))

    # Gather and save outputs
    if dist.size() > 1:
        outputs = dist.gather(outputs, dst=0)
        if not dist.is_main():
            return
        outputs = list(itertools.chain(*outputs))
    io.save(os.path.join(args.output_dir, "outputs.jsonl"), outputs)

    # Run evaluation
    metrics = {"accuracy": sum(output["pred"] == output["answer"] for output in outputs) / len(outputs)}
    io.save(os.path.join(args.output_dir, "metrics.json"), metrics)
    logger.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()


