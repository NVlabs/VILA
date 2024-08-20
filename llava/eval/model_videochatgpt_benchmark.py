import argparse
import json
import math
import os

import shortuuid
import torch
from decord import VideoReader
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from llava import conversation as conversation_lib
from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.data.dataset import LazySupervisedDataset
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    is_gemma_tokenizer,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config):
        self.questions = questions
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        if hasattr(model_config, "num_video_frames") and model_config.num_video_frames is not None:
            self.num_video_frames = model_config.num_video_frames
        else:
            self.num_video_frames = 8

        if hasattr(model_config, "fps") and model_config.fps is not None:
            self.fps = model_config.fps
        else:
            self.fps = 0.0

    def __getitem__(self, index):
        line = self.questions[index]

        # load visual
        video_name = line["video_name"]
        video_formats = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
        prepend = ["", "v_"]
        video_path = None
        for fmt in video_formats:
            for pre in prepend:
                temp_path = os.path.join(self.image_folder, f"{pre}{video_name}{fmt}")
                if os.path.exists(temp_path):
                    video_path = temp_path
                    break
            if video_path is not None:
                break

        images, frames_loaded = LazySupervisedDataset._load_video(video_path, self.num_video_frames, self.fps, args)
        image_tensor = process_images(images, self.image_processor, self.model_config)
        num_frames_loaded_successfully = len(images)

        if "Q" in line:
            questions = [line["Q"]]
        elif "Q1" in line:
            questions = [line["Q1"], line["Q2"]]

        input_ids_list = []
        for qs in questions:
            qs = qs.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            qs = qs.replace("<video>\n", "").replace("\n<video>", "").replace("<video>", "")
            qs = "<image>\n" * num_frames_loaded_successfully + qs

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            input_ids_list.append(input_ids)

        return input_ids_list, image_tensor

    def __len__(self):
        return len(self.questions)


def collate_fn(batch):
    input_ids, image_tensors = zip(*batch)
    input_ids = list(input_ids)
    image_tensors = torch.stack(image_tensors, dim=0)
    return input_ids, image_tensors


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn
    )
    return data_loader


def get_key(sample_set):
    question = sample_set["Q"] if "Q" in sample_set else (sample_set["Q1"] + sample_set["Q2"])
    k = question + sample_set["A"] + sample_set["video_name"]
    return k


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)
    args.image_processor = image_processor

    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode]

    gt_questions = json.load(open(args.gt_file))
    gt_questions = get_chunk(gt_questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.join(args.output_dir, f"{args.output_name}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(answers_file):
        with open(answers_file) as f:
            cache_ans = f.readlines()
            cache_set = list(json.loads(line) for line in cache_ans)
            cache_set = {get_key(line) for line in cache_set}
    else:
        cache_set = set()

    ans_file = open(answers_file, "a")
    data_loader = create_data_loader(gt_questions, args.image_folder, tokenizer, image_processor, model.config)

    for (input_ids_list, image_tensor), sample_q in tqdm(zip(data_loader, gt_questions), total=len(gt_questions)):
        input_ids_list = input_ids_list[0]
        sample_set = sample_q
        if get_key(sample_set) in cache_set:
            print(f"skip exist answer")
            continue
        outputs_list = []
        for input_ids in input_ids_list:
            input_ids = input_ids.to(device="cuda", non_blocking=True).unsqueeze(0)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            outputs_list.append(outputs)

        if len(outputs_list) == 1:
            sample_set["pred"] = outputs_list[0]
        elif len(outputs_list) == 2:
            sample_set["pred1"] = outputs_list[0]
            sample_set["pred2"] = outputs_list[1]

        ans_file.write(json.dumps(sample_set) + "\n")
        ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument(
        "--gt_file", help="Path to the ground truth file containing question and answer.", required=True
    )
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    args = parser.parse_args()

    eval_model(args)
