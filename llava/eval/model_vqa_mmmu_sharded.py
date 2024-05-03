# This file is originated from the official MMMU codebase:
# https://github.com/MMMU-Benchmark/MMMU
import torch
import os
import random

import math
import numpy as np
from tqdm import tqdm

from datasets import load_dataset, concatenate_datasets
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.mm_utils import process_images

from argparse import ArgumentParser

from llava.eval.mmmu_utils.data_utils import (
    load_yaml,
    construct_prompt,
    save_json,
    process_single_sample,
    CAT_SHORT2LONG,
)
from llava.eval.mmmu_utils.model_utils import (
    call_llava_engine_df,
    llava_image_processor,
)
from llava.eval.mmmu_utils.eval_utils import (
    parse_multi_choice_response,
    parse_open_response,
)


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def run_model(
    args,
    samples,
    model,
    call_model_engine_fn=None,
    tokenizer=None,
    processor=None,
    conv_version=None,
):
    out_samples = dict()
    with torch.no_grad():
        for sample in tqdm(samples):
            response = call_model_engine_fn(args, sample, model, tokenizer, processor)

            if sample["question_type"] == "multiple-choice":
                pred_ans = parse_multi_choice_response(
                    response, sample["all_choices"], sample["index2ans"]
                )
            else:  # open question
                pred_ans = response
            out_samples[sample["id"]] = pred_ans
    return out_samples


def set_seed(seed_value):
    """
    Set the seed for PyTorch (both CPU and CUDA), Python, and NumPy for reproducible results.

    :param seed_value: An integer value to be used as the seed.
    """
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # For multi-GPU setups
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--output_path",
        type=str,
        default="llava1.5_13b_val.json",
        help="name of saved json",
    )
    parser.add_argument(
        "--config_path", type=str, default="llava/eval/mmmu_utils/configs/llava1.5.yaml"
    )
    parser.add_argument(
        "--data_path", type=str, default="playground/data/eval/MMMU"
    )  # hf dataset path.
    parser.add_argument("--model_path", type=str, default="liuhaotian/llava-v1.5-13b")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    print("llava_initializing...")
    processor = None
    call_model_engine = call_llava_engine_df
    vis_process_func = llava_image_processor

    # load config and process to one value
    args.config = load_yaml(args.config_path)
    for key, value in args.config.items():
        if key != "eval_params" and type(value) == list:
            assert len(value) == 1, "key {} has more than one value".format(key)
            args.config[key] = value[0]

    # run for each subject
    sub_dataset_list = []
    print(len(list(CAT_SHORT2LONG.values())))
    for subject in CAT_SHORT2LONG.values():
        sub_dataset = load_dataset(args.data_path, subject, split=args.split)
        sub_dataset_list.append(sub_dataset)

    # merge all dataset
    dataset = concatenate_datasets(sub_dataset_list)
    dataset = list(dataset)
    dataset = get_chunk(dataset, args.num_chunks, args.chunk_idx)

    # load model
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, vis_processors, _ = load_pretrained_model(
        args.model_path, model_name, None
    )

    samples = []
    for sample in dataset:
        sample = process_single_sample(sample)

        sample = construct_prompt(sample, args.config)
        if sample["image"]:
            sample['image'] = process_images([image.convert("RGB") for image in sample['image']], vis_processors, model.config)
        samples.append(sample)

    # run ex
    # TODO (kentang-mit@): for other backbones such as mistral, we may still need this.
    # conv_version = args.conv_mode
    out_samples = run_model(
        args, samples, model, call_model_engine, tokenizer, processor
    )
    os.makedirs("/".join(args.output_path.split("/")[:-1]), exist_ok=True)
    output_path = args.output_path.replace(".json", f"-{args.chunk_idx}.json")
    save_json(output_path, out_samples)


if __name__ == "__main__":
    main()
