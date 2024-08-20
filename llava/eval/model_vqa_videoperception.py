# This file is modified from https://github.com/haotian-liu/LLaVA/

import argparse
import json
import math
import os
import signal

import torch
from tqdm import tqdm

from llava import conversation as conversation_lib
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.data import preprocess
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


def get_model_output(model, image_processor, tokenizer, video_path, qs, options, args):

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
    images = process_images(images, image_processor, model.config)

    num_frames_loaded_successfully = len(images)
    # print(f"Number of frames loaded successfully: {num_frames_loaded_successfully}")

    qs = qs.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
    qs = qs.replace("<video>\n", "").replace("\n<video>", "").replace("<video>", "")
    qs = "Watching the video and answer with the option's letter from the given choices directly." + qs
    qs = "<image>\n" * num_frames_loaded_successfully + qs

    loss_list = []

    for i, option in enumerate(options):
        qs = qs + chr(ord("A") + i) + ". " + option + "\n"

    conversation = [
        {"from": "human", "value": qs},
        {"from": "gpt", "value": None},
    ]

    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = (
        [KeywordsStoppingCriteria(keywords, tokenizer, input_ids)]
        if conv.version == "v0" or is_gemma_tokenizer(tokenizer)
        else None
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images.half().cuda(),
            # images=images.bfloat16().cuda(),
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )

    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    # out_samples[q_uid] = output_text
    output_text = output_text.strip()
    if output_text.endswith(stop_str):
        output_text = output_text[: -len(stop_str)]
    output_text = output_text.strip()
    print(output_text)

    if len(output_text) >= 1:
        pred_text = output_text[0]
    else:
        pred_text = ""

    return pred_text
    # sources = [conversation]

    # data_dict = preprocess(
    #     sources,
    #     tokenizer,
    #     has_image=True,
    # )
    # input_ids = data_dict["input_ids"]
    # targets = data_dict["labels"]
    # # Remove last ending token
    # targets[0][-1] = IGNORE_INDEX

    # with torch.inference_mode():
    #     outputs = model(
    #         input_ids=input_ids.cuda(),
    #         labels=targets.cuda(),
    #         images=image_tensor.half().cuda(),
    #         attention_mask=input_ids.cuda().ne(tokenizer.pad_token_id),
    #     )
    #     loss = outputs.loss.item()
    #     loss_list.append(loss)

    # # Get index of the minimum loss
    # min_loss_index = loss_list.index(min(loss_list))
    # return min_loss_index


def eval_model(args):
    # Model
    disable_torch_init()

    gt_questions = json.load(open(os.path.expanduser(args.gt_file)))
    # Convert the gt_questions dict to list
    gt_questions_list = []
    for key in gt_questions.keys():
        gt_questions_list.append(gt_questions[key])

    gt_questions = get_chunk(gt_questions_list, args.num_chunks, args.chunk_idx)
    # gt_answers = json.load(open(os.path.expanduser(args.gt_file_answers), "r"))
    # gt_answers = get_chunk(gt_answers, args.num_chunks, args.chunk_idx)

    args.output_dir = os.path.expanduser(args.output_dir)
    print(f"Output directory: {args.output_dir}")
    args.video_dir = os.path.expanduser(args.video_dir)
    print(f"Video directory: {args.video_dir}")
    answers_file = os.path.join(args.output_dir, f"{args.output_name}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    # Read cache answer file, each line is a json object
    if os.path.exists(answers_file):
        cache_ans_file = open(answers_file)
        cache_ans = cache_ans_file.readlines()
        cache_ans_file.close()
    else:
        cache_ans = []

    # Get cached video ids
    cache_set = {json.loads(line)["video_name_question_id"] for line in cache_ans}

    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)
    args.image_processor = image_processor

    # Iterate over each sample in the ground truth file
    # index = 0
    for sample in tqdm(gt_questions):
        video_name = sample["metadata"]["video_id"]
        questions = sample["mc_question"]
        # Check if the video is in the cache

        for question_dict in questions:
            # total += 1
            question = question_dict["question"]
            options = question_dict["options"]
            question_id = question_dict["id"]
            answer_id = question_dict["answer_id"]

            if f"{video_name}_{question_id}" in cache_set:
                print(f"Skipping {video_name}_{question_id} because it is in the cache")
                continue
            response = get_model_output(
                model,
                image_processor,
                tokenizer,
                os.path.join(args.video_dir, f"{video_name}.mp4"),
                question,
                options,
                args,
            )
            sample_set = {
                "video_name_question_id": f"{video_name}_{question_id}",
                "question": question,
                "answer_id": answer_id,
                "prediction": response,
                "correct": response == chr(ord("A") + answer_id),
            }
            with open(answers_file, "a") as f:
                f.write(json.dumps(sample_set) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model_max_length", type=int, required=False, default=2048)
    parser.add_argument("--video_dir", help="Directory containing video files.", required=True)
    parser.add_argument("--gt_file", help="Path to the ground truth file containing question.", required=True)
    # parser.add_argument('--gt_file_answers', help='Path to the ground truth file containing answers.', required=True)
    parser.add_argument("--output_dir", help="Directory to save the model results JSON.", required=True)
    parser.add_argument("--output_name", help="Name of the file for storing results JSON.", required=True)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
