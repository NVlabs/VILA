import argparse
import json
import math
import os
import os.path as osp
import re
from collections import defaultdict

import numpy as np
import shortuuid
import torch
import wandb
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    is_gemma_tokenizer,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


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


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class RefCOCODataset(Dataset):
    def __init__(self, loaded_data, image_folder, tokenizer, image_processor, model_config, conv_mode):
        self.loaded_data = loaded_data
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.conv_mode = conv_mode

    def __getitem__(self, index):
        line = self.loaded_data[index]
        image_file = line["img_id"]

        sents = line["sents"]
        qs = f"Please provide the bounding box coordinate of the region this sentence describes: {sents}."
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image = Image.open(os.path.join(self.image_folder, "_".join(image_file.split("_")[:-1]) + ".jpg")).convert(
            "RGB"
        )
        image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        input_ids = tokenizer_image_token(prompt, self.tokenizer, return_tensors="pt")

        return input_ids, image_tensor, image_file, line, image

    def __len__(self):
        return len(self.loaded_data)


def collate_fn(data):
    input_ids, image_tensor, image_file, raw_data, image = zip(*data)
    input_ids = torch.stack(input_ids, dim=0)
    image_tensor = torch.stack(image_tensor, dim=0)
    return input_ids, image_tensor, image_file, raw_data[0], image[0]


# DataLoader
def create_data_loader(
    loaded_data, image_folder, tokenizer, image_processor, model_config, conv_mode, batch_size=1, num_workers=8
):
    assert batch_size == 1, "batch_size must be 1"
    dataset = RefCOCODataset(loaded_data, image_folder, tokenizer, image_processor, model_config, conv_mode)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn, shuffle=False
    )
    return data_loader


def eval_model(args):
    if args.short_eval.lower() == "true":
        eval_dict = {"refcoco": ["val"]}
    else:
        eval_dict = {
            "refcoco": ["val", "testA", "testB"],
            "refcoco+": ["val", "testA", "testB"],
            "refcocog": ["val", "test"],
        }
    if args.chunk_idx == 0 and args.use_wandb.lower() == "true":
        print_rank0(
            args.chunk_idx,
            f"wandb inititialize with group: {args.group}, project: {args.project}, name: {args.run_name}\n",
            flush=True,
        )
        wandb.init(group=args.group, project=args.project, name=args.run_name)

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    print_rank0(f"model path: {model_path}\nmodel name: {model_name}\nmodel base: {args.model_base}\n", flush=True)

    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, model_name, args.model_base)
    print_rank0(args.chunk_idx, "model config: \n", model.config, flush=True)

    conv = conv_templates[args.conv_mode]
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [conv.sep]
    stopping_criteria = (
        [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)]
        if args.conv_mode == "v0" or is_gemma_tokenizer(tokenizer)
        else None
    )

    box_format = getattr(model.config, "box_format", "llava") if args.box_format is None else args.box_format
    for dataset in eval_dict.keys():
        for split in eval_dict[dataset]:
            count = 0
            # make directory for saving answers
            answers_file = osp.join(args.answers_path, dataset, split, args.answers_file_name)
            os.makedirs(os.path.dirname(answers_file), exist_ok=True)
            ans_file = open(answers_file, "w")

            with open(osp.join(args.data_path, f"{dataset}_{split}.json")) as f:
                loaded_data_full = json.load(f)

            loaded_data = get_chunk(loaded_data_full, args.num_chunks, args.chunk_idx)
            data_loader = create_data_loader(
                loaded_data, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode
            )
            print_rank0(args.chunk_idx, f"dataset size: {len(loaded_data)}/{len(loaded_data_full)}\n")

            answer_all = []
            resamples = []
            for input_ids, image_tensor, image_file, raw_data, raw_image in tqdm(data_loader, total=len(loaded_data)):
                sents = raw_data["sents"]
                gt_bbox = raw_data["bbox"]
                h = raw_data["height"]
                w = raw_data["width"]

                input_ids = input_ids.to(device="cuda", non_blocking=True)
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                        do_sample=True if args.temperature > 0 else False,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=args.top_k,
                        num_beams=args.num_beams,
                        max_new_tokens=args.max_new_tokens,
                        use_cache=True,
                        stopping_criteria=stopping_criteria,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                outputs = outputs.strip()
                ans = None
                try:
                    # pred_bbox = extract_all_bounding_boxes(outputs, w, h)[0]
                    pred_bbox = postprocess_2d_grounding(outputs, h, w, box_format)
                    ans = {
                        "img_id": image_file,
                        "text": outputs,
                        "bbox": pred_bbox,
                        "gt_bbox": gt_bbox,
                        "sents": sents,
                    }
                    answer_all.append(ans)
                except Exception as e:
                    print(outputs, e, flush=True)
                    resamples.append(raw_data)
                if count < 20 and ans is not None:
                    print(f"answer: {ans}\nimage size: {image_tensor.shape}\n", flush=True)
                count += 1

                if (args.use_wandb.lower() == "true") and args.chunk_idx == 0 and count % args.vis_interval == 0:
                    # gt is in x, y, w, h format
                    x, y, w, h = gt_bbox
                    gt_bbox = [x, y, x + w, y + h]
                    image_vis = draw_bounding_boxes(raw_image, [gt_bbox], [sents], color="green", width=3)
                    image_vis = draw_bounding_boxes(image_vis, [pred_bbox], [sents], color="red", width=3)

                    wandb.log(
                        {
                            f"AR prediction": [
                                wandb.Image(image_vis, caption=sents),
                            ],
                        },
                        commit=True,
                    )

            # process remaining questions
            for i in range(args.num_resample):
                print(f"Remaining {len(resamples)} images. Resampling {i+1} time.\n")
                data_loader = create_data_loader(
                    resamples, args.image_folder, tokenizer, image_processor, model.config, args.conv_mode
                )
                resamples = []
                for input_ids, image_tensor, image_file, raw_data, raw_image in data_loader:
                    gt_bbox = raw_data["bbox"]

                    h = raw_data["height"]
                    w = raw_data["width"]
                    sents = raw_data["sents"]

                    input_ids = input_ids.to(device="cuda", non_blocking=True)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                            do_sample=True if args.temperature > 0 else False,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            num_beams=args.num_beams,
                            max_new_tokens=args.max_new_tokens,
                            use_cache=True,
                            stopping_criteria=stopping_criteria,
                        )

                    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                    outputs = outputs.strip()
                    try:
                        pred_bbox = postprocess_2d_grounding(outputs, h, w, box_format)
                        ans = {
                            "img_id": image_file,
                            "text": outputs,
                            "bbox": pred_bbox,
                            "sents": sents,
                            "gt_bbox": gt_bbox,
                        }
                        answer_all.append(ans)
                    except:
                        print(raw_data, flush=True)
                        resamples.append(raw_data)

                if len(resamples) == 0:
                    break

            print(f"Finished resampling. Remaining {len(resamples)} images.\n")
            answer_all = sorted(answer_all, key=lambda x: x["img_id"])
            for ans in answer_all:
                ans_file.write(json.dumps(ans) + "\n")
            ans_file.close()
            print(f"answers saved at {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--data-path", type=str, default="data/annotations/finetune_refcoco_testA.json")
    parser.add_argument("--answers-path", type=str, default="")
    parser.add_argument("--answers-file-name", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-resample", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--use-wandb", type=str, default="False")
    parser.add_argument("--group", type=str, default="")
    parser.add_argument("--project", type=str, default="VILA-RefCOCO-Eval")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--vis_interval", type=int, default=100)
    parser.add_argument("--box_format", type=str, default=None)
    parser.add_argument("--short-eval", type=str, default="False")
    args = parser.parse_args()

    eval_model(args)


