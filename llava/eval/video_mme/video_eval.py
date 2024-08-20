# Developed by Ligeng Zhu (@lyken17)

import argparse
import glob
import json
import math
import os
import os.path as osp
import shutil
import signal

import numpy as np
import shortuuid
import torch
from filelock import FileLock
from PIL import Image
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Resize
from tqdm import tqdm

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import KeywordsStoppingCriteria, get_model_name_from_path, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


# This function will be called when the timeout is reached
def handler(signum, frame):
    raise TimeoutError()


# Set the signal handler
signal.signal(signal.SIGALRM, handler)


def safely_merge_info(out_fpath, info):
    out_folder = osp.dirname(out_fpath)
    if len(out_folder) > 2:
        os.makedirs(out_folder, exist_ok=True)
    with FileLock(out_fpath + ".lock"):
        if osp.exists(out_fpath):
            try:
                new_info = json.load(
                    open(out_fpath, "r+"),
                )
                info.update(new_info)
            except json.decoder.JSONDecodeError:
                pass
        json.dump(info, open(out_fpath + ".meta", "w+"), indent=2)
        shutil.move(out_fpath + ".meta", out_fpath)
    return info


def get_model_output(
    model,
    image_processor,
    tokenizer,
    video_path,
    qs,
    conv_mode="vicuna_v1",
    num_video_frames=8,
    temperature=0.2,
    num_beams=1,
):
    from llava.mm_utils import opencv_extract_frames

    imgs, num_frames = opencv_extract_frames(video_path, num_video_frames)
    num_frames_loaded = len(imgs)
    # print(imgs)

    image_tensor = [
        # processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in torch.unbind(image_tensor)
        image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        for image in imgs
    ]
    image_tensor = torch.stack(image_tensor)

    qs = "<image>\n" * num_frames_loaded + qs
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt,
        tokenizer,
        image_token_index=IMAGE_TOKEN_INDEX,
        return_tensors="pt",
    )
    input_ids = torch.unsqueeze(input_ids, 0)
    input_ids = torch.as_tensor(input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    do_sample = True
    if temperature == 0:
        do_sample = False
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.half().cuda(),
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=1024,
            num_beams=num_beams,
            use_cache=True,
            stopping_criteria=[stopping_criteria],
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    return outputs


template = r""" Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{question}
The best answer is:
"""

template_wsub = r""" This video's subtitles are listed below:
{subtitle}
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
{question}
The best answer is:
"""


def get_path(root_path):
    from huggingface_hub import repo_exists, snapshot_download
    from huggingface_hub.utils import HFValidationError, validate_repo_id

    if root_path is not None and not osp.exists(root_path):
        try:
            valid_hf_repo = repo_exists(root_path)
        except HFValidationError as e:
            valid_hf_repo = False
        if valid_hf_repo:
            root_path = snapshot_download(root_path)
    return root_path


def eval_model(args):
    from pprint import pprint

    pprint(vars(args))
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    if args.output_name is None:
        args.output_name = (
            osp.basename(args.model_path) + f"tmp={args.temperature}_beams={args.num_beams}-" + "video_mme.json"
        )
    if not args.output_name.endswith(".json"):
        args.output_name += ".json"

    if args.num_video_frames is None or args.num_video_frames < 0:
        root_path = osp.join(get_path(model_path))
        args.num_video_frames = json.load(open(osp.join(root_path, "config.json")))["num_video_frames"]
        print(
            f"detecing video frames None, using {args.num_video_frames} frames, loading from {root_path}/config: ",
        )
    else:
        print(f"using num_video_frames from args: ", args.num_video_frames)

    answers_file = os.path.join(args.output_dir, args.output_name)
    os.makedirs(args.output_dir, exist_ok=True)

    output_json = []
    labeled_key = {}
    if osp.exists(answers_file):
        labeled_key = json.load(open(answers_file))
    print(f"[{answers_file}] already answered ", len(labeled_key.keys()))

    # videomme v1, released in june 1  2024
    jinfo = json.load(open("/home/ligengz/workspace/video-mme/Video-MME.json"))
    folder = "/home/ligengz/workspace/video-mme/ytb_videos"

    # videomme v2, updated in june 20 2024
    jinfo = json.load(open("/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/Video-MME/qa_old_format.json"))
    folder = "/lustre/fsw/portfolios/nvr/projects/nvr_elm_llm/dataset/Video-MME/videos"
    subtitle_folder = "/home/ligengz/nvr_elm_llm/dataset/Video-MME/subtitle"

    if args.convert:
        for vmeta in jinfo:
            for question in vmeta["questions"]:
                qid = question["question_id"]
                if qid in labeled_key:
                    # question["response"] = labeled_key[qid]["response"]
                    question["response_w/o_sub"] = labeled_key[qid]["response_w/o_sub"]
                    question["response_w/_sub"] = labeled_key[qid]["response_w/_sub"]
                else:
                    # if not answered, using "C" as the default answer.
                    print("missing", qid)
                    question["response_w/o_sub"] = "C"
                    question["response_w/_sub"] = "C"
        with open(answers_file.replace(".json", "_converted.json"), "w") as fp:
            json.dump(jinfo, fp, indent=2)
        return 0

    begin_idx = 0
    end_idx = len(jinfo)
    if args.total > 0:
        chunk = len(jinfo) / float(args.total)
        begin_idx = int(chunk * args.shard)
        end_idx = min(int(chunk * (args.shard + 1)), len(jinfo))
        print(f"labeling btw {begin_idx}-{end_idx}, total {end_idx - begin_idx} videos from {len(jinfo)}")
    # print("debug", begin_idx, end_idx)

    # Model
    disable_torch_init()

    print(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, model_name, args.model_base)

    for idx, vmeta in tqdm(enumerate(jinfo), total=len(jinfo)):
        if not (idx >= begin_idx and idx < end_idx):
            continue
        url = vmeta["url"]
        video_id = vmeta["video_id"]
        uid = osp.basename(url).split("?v=")[-1]

        vpath = osp.join(folder, f"{uid}.mp4")
        subpath = osp.join(subtitle_folder, f"{uid}.srt")

        from llava.eval.video_mme.w_sub_eval import slice_frames

        video_frames, video_subtitles = slice_frames(vpath, subpath, num_frames=args.num_video_frames)
        if not osp.exists(vpath):
            print("[video not downloaded] Skip", vpath)
            continue

        for questions in vmeta["questions"]:
            qid = questions["question_id"]
            if qid in labeled_key:
                print("[question id answered] Skip", qid, url)
                continue
            qa = (
                questions["question"]
                + "\n"
                + "Answer the question by only outputing the choice.\n"
                + "\n".join(questions["choices"])
            )

            qs = template.format(question=qa)
            output = get_model_output(
                model,
                image_processor,
                tokenizer,
                vpath,
                qs,
                conv_mode=args.conv_mode,
                temperature=args.temperature,
                num_beams=args.num_beams,
                num_video_frames=args.num_video_frames,
            )
            questions["response_w/o_sub"] = output

            qs = template_wsub.format(question=qa, subtitle=video_subtitles)
            output = get_model_output(
                model,
                image_processor,
                tokenizer,
                vpath,
                qs,
                conv_mode=args.conv_mode,
                temperature=args.temperature,
                num_beams=args.num_beams,
                num_video_frames=args.num_video_frames,
            )
            questions["response_w/_sub"] = output
            labeled_key[questions["question_id"]] = questions
        # break
        # output_json.append(vmeta)
        safely_merge_info(answers_file, labeled_key)
        # with open(output_name, "w") as fp:
        #     json.dump(output_json, fp, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("-c", "--convert", action="store_true")
    parser.add_argument("--with-sub", action="store_true")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--total", type=int, default=-1)
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-3b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model_max_length", type=int, required=False, default=5120)
    parser.add_argument("--num_video_frames", default=None, type=int)
    parser.add_argument(
        "--output_dir",
        help="Directory to save the model results JSON.",
        default=".",
        type=str,
    )
    parser.add_argument(
        "--output_name",
        help="Name of the file for storing results JSON.",
        default=None,
        type=str,
    )
    parser.add_argument("--conv-mode", type=str, default="vicuna_v1")
    parser.add_argument("--num-chunks", type=int, default=1)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    args = parser.parse_args()

    eval_model(args)
