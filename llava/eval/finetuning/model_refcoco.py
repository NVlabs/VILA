import argparse
import itertools
import json
import os
import re

import torch
import wandb
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

import llava
from llava import conversation as conversation_lib
from llava.utils import distributed as dist
from llava.utils import io


def print_rank0(rank, *args, **kwargs):
    if rank == 0:
        print(*args, **kwargs)


def extract_all_bounding_boxes(text, w, h):
    pattern = r"\[([\d.]+),\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\]"
    matches = re.findall(pattern, text)
    bboxes = [[float("0." + b) for b in box] for box in matches]
    bboxes = [[box[0] * w, box[1] * h, box[2] * w, box[3] * h] for box in bboxes]
    return bboxes


def draw_bounding_boxes(image, bboxes, labels, color="green", width=2, fontcolor=None):
    image_copy = image.copy()

    # Create a drawing context
    draw = ImageDraw.Draw(image_copy)

    # Load a font (you may need to specify the font file path)
    font = ImageFont.load_default()

    fontcolor = color if fontcolor is None else fontcolor
    for box, class_name in zip(bboxes, labels):
        x_min, y_min, x_max, y_max = box

        if x_min < x_max and y_min < y_max:
            # Draw the bounding box
            draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)

            # Draw the class name above the bounding box
            text_x = x_min
            text_y = y_min - width - 1

            # Ensure the text doesn't go out of the image bounds
            if text_y < 0:
                text_y = 0

            draw.text((text_x, text_y), class_name, fill=fontcolor, font=font)

    return image_copy


def postprocess_2d_grounding(output, h, w, box_format="llava"):
    h = float(h)
    w = float(w)
    output = output.strip()
    if output.endswith("."):
        output = output[:-1]
    output = output.strip()
    output = output[1:-1]  # remove [ and ]
    output = output.split(",")
    assert len(output) == 4, f"output should have 4 elements, but got {output}"
    if "standard" in box_format:
        x1, y1, x2, y2 = (float(x.strip()) for x in output)
    elif "000" in box_format:
        x1, y1, x2, y2 = (float("0." + x.strip()) for x in output)
    else:
        # we don't know the format. try both
        try:
            x1, y1, x2, y2 = (float("0." + x.strip()) for x in output)
        except:
            x1, y1, x2, y2 = (float(x.strip()) for x in output)

    if "llava" in box_format:
        # NOTE: llava box format is on image coordinate AFTER square pad.

        # 1. convert to image coordinate
        n = max(w, h)
        x1 = x1 * n
        y1 = y1 * n
        x2 = x2 * n
        y2 = y2 * n

        # 2. convert to coordinate before padding
        x1 = x1 - (n - w) / 2
        y1 = y1 - (n - h) / 2
        x2 = x2 - (n - w) / 2
        y2 = y2 - (n - h) / 2

        return [x1, y1, x2, y2]
    else:
        return [x1 * w, y1 * h, x2 * w, y2 * h]


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
    parser.add_argument("--use-wandb", type=str, default="False")
    parser.add_argument("--group", type=str, default="")
    parser.add_argument("--project", type=str, default="VILA-RefCOCO-Eval")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--vis_interval", type=int, default=100)
    parser.add_argument("--box_format", type=str, default=None)
    parser.add_argument("--short-eval", type=str, default="False")
    parser.add_argument("--num-resample", type=int, default=0)
    args = parser.parse_args()

    if args.short_eval.lower() == "true":
        eval_dict = {"refcoco": ["val"]}
    else:
        eval_dict = {
            "refcoco": ["val", "testA", "testB"],
            "refcoco+": ["val", "testA", "testB"],
            "refcocog": ["val", "test"],
        }

    # Set up distributed environment
    if args.num_chunks is None:
        dist.init()
        world_size, global_rank = dist.size(), dist.rank()
        devices = range(dist.local_rank(), torch.cuda.device_count(), dist.local_size())
        torch.cuda.set_device(devices[0])
    else:
        world_size, global_rank = args.num_chunks, args.chunk_idx

    if global_rank == 0 and args.use_wandb.lower() == "true":
        print_rank0(
            global_rank,
            f"wandb inititialize with group: {args.group}, project: {args.project}, name: {args.run_name}\n",
            flush=True,
        )
        wandb.init(group=args.group, project=args.project, name=args.run_name)

    # TODO(zhijianl): This will be removed in the future
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode].copy()

    # Load model
    model = llava.load(args.model_path, model_base=args.model_base, devices=devices)
    box_format = getattr(model.config, "box_format", "llava") if args.box_format is None else args.box_format

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

    for dataset in eval_dict.keys():
        for split in eval_dict[dataset]:

            # Load data and chunk it
            instances = io.load(os.path.join(args.question_file, f"{dataset}_{split}.json"))[global_rank::world_size]

            # Run inference
            answer_all = []
            resamples = []
            count = 0
            for instance in tqdm(instances, disable=global_rank != 0):
                image = Image.open(
                    os.path.join(args.image_folder, "_".join(instance["img_id"].split("_")[:-1]) + ".jpg")
                )
                sents = instance["sents"]
                question = f"Please provide the bounding box coordinate of the region this sentence describes: {sents}."
                response = model.generate_content([image, question], generation_config=generation_config)

                gt_bbox = instance["bbox"]
                h = instance["height"]
                w = instance["width"]
                image_file = instance["img_id"]
                ans = None
                try:
                    pred_bbox = postprocess_2d_grounding(response, h, w, box_format)
                    ans = {
                        "img_id": image_file,
                        "text": response,
                        "bbox": pred_bbox,
                        "gt_bbox": gt_bbox,
                        "sents": sents,
                    }
                    answer_all.append(ans)
                except Exception as e:
                    print(response, e, flush=True)
                    resamples.append(instance)
                if count < 20 and ans is not None:
                    print(f"answer: {ans}\n", flush=True)
                count += 1

                if (args.use_wandb.lower() == "true") and global_rank == 0 and count % args.vis_interval == 0:
                    # gt is in x, y, w, h format
                    x, y, w, h = gt_bbox
                    gt_bbox = [x, y, x + w, y + h]
                    image_vis = draw_bounding_boxes(image.convert("RGB"), [gt_bbox], [sents], color="green", width=3)
                    image_vis = draw_bounding_boxes(image_vis, [pred_bbox], [sents], color="red", width=3)

                    wandb.log(
                        {
                            f"AR prediction": [
                                wandb.Image(image_vis, caption=sents),
                            ],
                        },
                        commit=True,
                    )

            for i in range(args.num_resample):
                print(f"Remaining {len(resamples)} images. Resampling {i+1} time.\n")
                instances = resamples
                resamples = []
                for instance in instances:
                    image = Image.open(
                        os.path.join(args.image_folder, "_".join(instance["img_id"].split("_")[:-1]) + ".jpg")
                    )
                    sents = instance["sents"]
                    question = (
                        f"Please provide the bounding box coordinate of the region this sentence describes: {sents}."
                    )
                    response = model.generate_content([image, question], generation_config=generation_config)

                    gt_bbox = instance["bbox"]
                    h = instance["height"]
                    w = instance["width"]
                    image_file = instance["img_id"]
                    try:
                        pred_bbox = postprocess_2d_grounding(response, h, w, box_format)
                        ans = {
                            "img_id": image_file,
                            "text": response,
                            "bbox": pred_bbox,
                            "gt_bbox": gt_bbox,
                            "sents": sents,
                        }
                        answer_all.append(ans)
                    except Exception as e:
                        print(response, e, flush=True)
                        resamples.append(instance)

                    if len(resamples) == 0:
                        break

            # Gather outputs and save
            answer_all = sorted(answer_all, key=lambda x: x["img_id"])
            if dist.size() > 1:
                answer_all = dist.gather(answer_all, dst=0)
                if not dist.is_main():
                    continue
                answer_all = list(itertools.chain(*answer_all))
            answer_file = os.path.join(args.answers_file, dataset, split, "merge.jsonl")
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            io.save(answer_file, answer_all)


if __name__ == "__main__":
    main()
