# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import json
import math
import os
import signal

import numpy as np
import shortuuid
import torch
from PIL import Image
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Resize
from tqdm import tqdm

from llava import conversation as conversation_lib
from llava.conversation import SeparatorStyle, conv_templates
from llava.data.dataset import LazySupervisedDataset
from llava.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_image,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


# This function will be called when the timeout is reached
def handler(signum, frame):
    raise TimeoutError()


# Set the signal handler
signal.signal(signal.SIGALRM, handler)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def get_model_output(model, image_processor, tokenizer, video_path, qs, args):
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_mode]
    if hasattr(model.config, "num_video_frames") and model.config.num_video_frames is not None:
        num_video_frames = model.config.num_video_frames
    else:
        num_video_frames = 8

    if hasattr(model.config, "fps") and model.config.fps is not None:
        fps = model.config.fps
    else:
        fps = 0.0

    # print(fps)
    images, frames_loaded = LazySupervisedDataset._load_video(video_path, num_video_frames, fps, args)
    # image_tensor = process_images(images, image_processor, model.config)
    image_tensor = torch.stack([process_image(image, args, None) for image in images])
    num_frames_loaded_successfully = len(images)
    # print(f"Number of frames loaded successfully: {num_frames_loaded_successfully}")
    qs = qs.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
    qs = qs.replace("<video>\n", "").replace("\n<video>", "").replace("<video>", "")
    qs = "<image>\n" * num_frames_loaded_successfully + qs

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        return_tensors="pt",
    )
    input_ids = torch.unsqueeze(input_ids, 0)
    input_ids = torch.as_tensor(input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.half().cuda(),
            do_sample=True,
            temperature=0.2,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    input_token_len = input_ids.shape[1]
    # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    # if n_diff_input_output > 0:
    #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    # outputs_2 = tokenizer.batch_decode(output_ids)[0]
    print("========================================")
    print("Input:", qs)
    # print("num frames loaded successfully:", num_frames_loaded_successfully)
    print("Output 1:", outputs)
    # print("raw output 2:", outputs_2)
    # print("batched output 2," , tokenizer.batch_decode(output_ids))
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs


def parse_caption_template(template_type):
    if "bin" in template_type:
        short_bin = template_type.split("_")[1]
        long_bin = template_type.split("_")[2]
        short_question = "Elaborate on the visual and narrative elements of the video in detail."
        long_question = f"Summarize the visual content of the following video. Please write the caption with no more than {long_bin} words."
    else:
        raise ValueError(f"Invalid template type: {template_type}")

    return short_question, long_question


def eval_model(args):
    # Model
    disable_torch_init()

    # List video files
    video_formats = [".mp4", ".avi", ".mov", ".mkv"]
    video_files = os.listdir(args.video_dir)
    video_files = [f for f in video_files if os.path.splitext(f)[1] in video_formats]
    short_q, long_q = parse_caption_template(args.prompt_template)
    gt_questions = []
    for i, video_name in enumerate(video_files):
        gt_questions.append(
            {
                "video_name": video_name,
                "short_q": short_q,
                "long_q": long_q,
            }
        )

    # Create the output directory if it doesn't exist

    args.output_dir = os.path.expanduser(args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f"Output directory: {args.output_dir}")
    args.video_dir = os.path.expanduser(args.video_dir)
    print(f"Video directory: {args.video_dir}")
    short_caption_file = os.path.join(args.output_dir, f"detailed_captions.txt")
    short_caption_file = open(short_caption_file, "w")
    long_caption_file = os.path.join(args.output_dir, f"video_captions.txt")
    long_caption_file = open(long_caption_file, "w")

    # List to store the output results
    output_list = []

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)
    args.image_processor = image_processor

    if hasattr(model.config, "image_aspect_ratio") and model.config.image_aspect_ratio is not None:
        args.image_aspect_ratio = model.config.image_aspect_ratio
    else:
        raise ValueError("image_aspect_ratio is not found in the model config")

    # Iterate over each sample in the ground truth file
    index = 0
    for sample in tqdm(gt_questions):
        video_name = sample["video_name"]
        short_question = sample["short_q"]
        long_question = sample["long_q"]
        index += 1

        # Load the video file
        for question_type in ["short", "long"]:
            temp_path = os.path.join(args.video_dir, f"{video_name}")
            print(f"Processing video: {temp_path}")
            if os.path.exists(temp_path):
                video_path = temp_path
                question = short_question if question_type == "short" else long_question
                output = get_model_output(model, image_processor, tokenizer, video_path, question, args)
                print("question:", question)
                print("output:", output)
                if question_type == "short":
                    short_caption_file.write(f"{video_name}: {output}\n")
                elif question_type == "long":
                    long_caption_file.write(f"{video_name}: {output}\n")
                else:
                    raise ValueError(f"Invalid question type: {question_type}")
    short_caption_file.close()
    long_caption_file.close()
    print(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--video_dir", help="Directory containing video files.", required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--prompt_template", type=str, help="The template of video caption question.", required=True)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)


