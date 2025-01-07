# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/EvolvingLMMs-Lab/LongVA

import argparse
import gc
import glob
import json
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from accelerate import Accelerator
from datasets import load_dataset
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM

from llava.eval.vision_niah_vila.zigzag_ring_attn.modeling_qwen2 import Qwen2ForCausalLM_RingAttn
from llava.eval.vision_niah_vila.zigzag_ring_attn.monkey_patch import apply_zigzag_ring_attn_monkey_patch_llama
from llava.eval.vision_niah_vila.zigzag_ring_attn.prepare_inputs import prepare_zigzag_ring_attn_inputs
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model

apply_zigzag_ring_attn_monkey_patch_llama()

SEED = 24242424
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

prompt_templates = {
    "mistral": {"preprompt": "<s>[INST]", "postprompt": " [/INST]"},
    "vicuna": {
        "preprompt": "<s>A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER:",
        "postprompt": "ASSISTANT:",
    },
    "llama3": {
        "preprompt": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "postprompt": "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    },
    "qwen2": {
        "preprompt": "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
        "postprompt": "<|im_end|>\n<|im_start|>assistant\n",
    },
    "yi": {
        "preprompt": "<|im_start|>system\nAnswer the questions.<|im_end|>\n<|im_start|>user\n",
        "postprompt": "<|im_end|>\n<|im_start|>assistant\n",
    },
}
# \nAnswer the question using a single word or phrase.
# The color of the bottle cap is
# answer = "Yellow"


def safe_tokenize(tokenizer, text):
    tokenized = tokenizer.encode(text, return_tensors="pt")
    if tokenizer.bos_token != None and len(tokenized) > 0 and tokenized[0, 0] == tokenizer.bos_token_id:
        tokenized = tokenized[:, 1:]
    return tokenized


# answer = "more bet"
def eval_forward(accelerator, model, input_embeds, answer_embeds, pad_id, answer_ids, tokenizer):
    # first append answer_embeds to input_embeds
    prompt_length = input_embeds.shape[1]
    labels_length = answer_embeds.shape[1]
    input_embeds = torch.cat([input_embeds, answer_embeds], dim=1)
    # second pad input_embeds to the multiple of accelerator.num_processes
    pad_tensor = (
        torch.tensor(
            [pad_id] * ((accelerator.num_processes * 2) - input_embeds.shape[1] % (accelerator.num_processes * 2))
        )
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(-1, -1, input_embeds.shape[-1])
    )  # .to(accelerator.device)
    input_embeds = torch.cat([input_embeds, pad_tensor], dim=1)
    position_ids = (
        torch.arange(input_embeds.shape[1]).unsqueeze(0).expand(input_embeds.shape[0], -1)
    )  # .to(accelerator.device)
    accelerator.print(input_embeds.shape)
    prepared = prepare_zigzag_ring_attn_inputs(
        input_embeds,
        position_ids,
        None,
        accelerator.process_index,
        accelerator.num_processes,
        accelerator.device,
    )
    local_input_embeds = prepared["local_input_ids"]
    local_position_ids = prepared["local_position_ids"]
    with torch.inference_mode():
        logits = model(
            inputs_embeds=local_input_embeds,
            position_ids=local_position_ids,
            use_cache=False,
        ).logits
        pred = logits.argmax(dim=-1)

    # gather all logits using accelerator.gather
    def undo_extract_local(gathered_value, world_size, dim=1):
        value_chunks = gathered_value.chunk(2 * world_size, dim=dim)
        reordered_chunks = [None] * (2 * world_size)
        for i in range(world_size):
            reordered_chunks[i] = value_chunks[i * 2]
            reordered_chunks[2 * world_size - i - 1] = value_chunks[i * 2 + 1]
        return torch.cat(reordered_chunks, dim=dim)

    correct = False

    gathered_logits = accelerator.gather(pred.squeeze(0)).unsqueeze(0)
    # undo extract local on the gathered logits
    pred = undo_extract_local(gathered_logits, accelerator.num_processes)

    pred = pred[:, prompt_length - 1 : prompt_length + labels_length - 1]
    # check if the logits are correct, extract argmax id
    # compare the predicted_ids with the labels
    pred_text = tokenizer.decode(pred.squeeze().tolist())
    answer_text = tokenizer.decode(answer_ids.squeeze().tolist())
    correct = pred_text.replace(" ", "").lower() == answer_text.replace(" ", "").lower()
    if accelerator.is_main_process:
        print(
            "Predicted: ",
            pred_text,
            "Answer: ",
            answer_text,
        )
        # print id as well
        print(
            "Predicted: ",
            pred.squeeze().tolist(),
            "Answer: ",
            answer_ids.squeeze().tolist(),
        )
    return int(correct)


def load_haystack(args, accelerator):
    haystack_embeddings = torch.load(f"{args.haystack_dir}/video_embeddings.pt").to(torch.bfloat16)
    return haystack_embeddings


def load_text_embeddings(str, tokenizer, model, accelerator, replace_double_newline=False):
    token_ids = safe_tokenize(tokenizer, str)

    def replace_double_newline_func(token_ids):
        double_newline_loc = (token_ids == 271).nonzero()[:, 1]
        double_newline_loc += torch.arange(len(double_newline_loc))
        if len(double_newline_loc) > 0:
            for loc in double_newline_loc:
                token_ids = torch.cat(
                    [
                        token_ids[:, :loc],
                        torch.tensor([[198, 198]]),
                        token_ids[:, loc + 1 :],
                    ],
                    dim=1,
                )
        return token_ids

    if replace_double_newline:
        token_ids = replace_double_newline_func(token_ids)
    with torch.inference_mode():
        embeddings = model.model.embed_tokens(token_ids)
    return embeddings.to(torch.bfloat16)


def get_model_name(model):
    model_split = [name for name in model.split("/") if len(name) > 0]
    model_name = f"{model_split[-2]}_{model_split[-1]}"
    return model_name


def load_results(results_dir):
    results = []
    if os.path.exists(results_dir):
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                if "json" in file:
                    print("file", file)
                    results.append(json.load(open(os.path.join(root, file))))
    else:
        os.system("mkdir -p %s" % results_dir)
    return results


def inference(args):
    model = args.model
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(args.model, "llm"),
        model_max_length=sys.maxsize,
        trust_remote_code=True,
    )

    tokenizer.pad_token = tokenizer.eos_token

    accelerator = Accelerator(
        mixed_precision="bf16",
    )
    kwargs = {"rope_theta": args.rope_theta} if args.rope_theta is not None else {}
    if "qwen2" in args.model.lower() or "longva" in args.model.lower():
        model = Qwen2ForCausalLM_RingAttn.from_pretrained(
            os.path.join(args.model, "llm"),
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
            **kwargs,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            os.path.join(args.model, "llm"),
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
            device_map=accelerator.device,
            **kwargs,
        )
    tokenizer.pad_token = tokenizer.eos_token
    # remember to remove <s>
    accelerator.print("Preparing Haystack...")
    haystack_embeddings = load_haystack(args, accelerator)
    assert (
        len(haystack_embeddings) >= args.max_frame_num
    ), f"Haystack embeddings are not enough. Max frame {args.max_frame_num} is not found. Currently only {len(haystack_embeddings)} frames."
    haystack_embeddings = haystack_embeddings[: args.max_frame_num]
    prompt = prompt_templates[args.prompt_template]
    preprompt_embeddings = load_text_embeddings(
        prompt["preprompt"], tokenizer, model, accelerator, args.replace_double_newline
    )
    postprompt_embeddings = load_text_embeddings(
        prompt["postprompt"], tokenizer, model, accelerator, args.replace_double_newline
    )

    needle_dataset = load_dataset(args.needle_dataset)["test"]
    answer_embedding_list = []
    answer_id_list = []
    needle_embedding_list = []
    question_embeding_list = []
    for index, instance in enumerate(needle_dataset):
        answer = instance["answer"]
        question = instance["question"]
        needle_embedding_list.append(
            torch.load(args.needle_embedding_dir + f"/{index}.pt", map_location="cpu").to(torch.bfloat16)
        )  # .to(accelerator.device))
        answer_embedding_list.append(load_text_embeddings(answer, tokenizer, model, accelerator))
        answer_id_list.append(safe_tokenize(tokenizer, answer))
        question_embeding_list.append(load_text_embeddings(question, tokenizer, model, accelerator))

    accelerator.print("Starting Evaluation...")
    model = accelerator.prepare(model)
    model.gradient_checkpointing_enable()

    model_name = get_model_name(args.model)
    results_dir = "results/%s" % model_name
    all_accuries = load_results(results_dir)

    for num_frames in tqdm(range(args.min_frame_num, args.max_frame_num + 1, args.frame_interval)):
        context_depths = [result["Frame Depth"] for result in all_accuries if result["Num. Frame"] == num_frames]
        for depth in np.arange(0, 1 + args.depth_interval, args.depth_interval):
            if round(depth * 100, -1) in context_depths:
                print("Context %d, depth %d already done." % (num_frames, round(depth * 100, -1)))
                continue
            accuracies = []
            for (question_embedding, needle_embedding, answer_embedding, answer_id,) in zip(
                question_embeding_list,
                needle_embedding_list,
                answer_embedding_list,
                answer_id_list,
            ):
                query_frame_idx = int(depth * num_frames)
                input_frames = (
                    torch.cat(
                        [
                            haystack_embeddings[:query_frame_idx],
                            needle_embedding.unsqueeze(0),
                            haystack_embeddings[query_frame_idx:num_frames],
                        ],
                        dim=0,
                    )
                    .view(-1, haystack_embeddings.shape[-1])
                    .unsqueeze(0)
                )
                input_emebds = torch.cat(
                    [
                        preprompt_embeddings,
                        input_frames,
                        question_embedding,
                        postprompt_embeddings,
                    ],
                    dim=1,
                )
                correct = eval_forward(
                    accelerator,
                    model,
                    input_emebds,
                    answer_embedding,
                    tokenizer.pad_token_id,
                    answer_id,
                    tokenizer,
                )
                gc.collect()
                torch.cuda.empty_cache()
                if accelerator.is_main_process:
                    accuracies.append(correct)

            if accelerator.is_main_process:
                result = {
                    "Num. Frame": num_frames,
                    "Frame Depth": round(depth * 100, -1),
                    "Score": sum(accuracies) / len(accuracies),
                }
                accelerator.print(result)
                all_accuries.append(result)
                json.dump(
                    result,
                    open(
                        os.path.join(
                            results_dir,
                            "frame_%d_depth_%d.json" % (num_frames, int(depth * 100)),
                        ),
                        "w",
                    ),
                )

    if accelerator.is_main_process:
        model_name = args.model.split("/")[-1]
        os.makedirs(f"{args.output_path}/{model_name}", exist_ok=True)
        # save all_accuries as json
        with open(f"{args.output_path}/{model_name}/all_accuracies.json", "w") as f:
            json.dump(all_accuries, f, indent=4)
    return all_accuries, accelerator


def plot(args, all_accuries):
    df = pd.DataFrame(all_accuries)
    cmap = LinearSegmentedColormap.from_list("custom_cmap", ["#F0496E", "#EBB839", "#9ad5b3"])

    pivot_table = pd.pivot_table(
        df,
        values="Score",
        index=["Frame Depth", "Num. Frame"],
        aggfunc="mean",
    ).reset_index()  # This will aggregate
    pivot_table = pivot_table.pivot(index="Frame Depth", columns="Num. Frame", values="Score")
    # Create the heatmap with better aesthetics
    plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    ax = sns.heatmap(
        pivot_table,
        # annot=True,
        fmt="g",
        vmin=0,
        vmax=1,
        linecolor="white",
        linewidths=1.5,
        cmap=cmap,
        cbar_kws={"label": "Score"},
    )
    font_size = 24

    # Set the color bar label font size
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(font_size)
    cbar.ax.tick_params(labelsize=font_size)

    # Define the formatter function
    def thousands_formatter(x, pos):
        if x >= 1000:
            return f"{x/1000:.1f}K"
        return f"{x}"

    context_lengths = pivot_table.columns
    formatted_context_lengths = [thousands_formatter(x, None) for x in context_lengths]

    # More aesthetics
    plt.xlabel("Num. of Frames", fontsize=font_size)  # X-axis label
    plt.ylabel("Depth Percent", fontsize=font_size)  # Y-axis label
    plt.xticks(
        ticks=[i + 0.5 for i in range(len(context_lengths))],
        labels=formatted_context_lengths,
        rotation=45,
        fontsize=font_size,
    )
    # plt.xticks(rotation=45, fontsize=14)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0, fontsize=font_size)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area
    # save
    model_name = args.model.split("/")[-1]

    plt.savefig(f"{args.output_path}/{model_name}/heatmap.png")
    # calculate average accuracy
    average_accuracy = df["Score"].mean()
    print(f"Average Accuracy: {average_accuracy}")
    # save as txt
    with open(f"{args.output_path}/{model_name}/avg_accuracy.txt", "w") as f:
        f.write(f"Average Accuracy: {average_accuracy}\n")


def main(args):
    if args.plot_only:
        # load all_accuracies from json
        model_name = args.model.split("/")[-1]
        with open(f"{args.output_path}/{model_name}/all_accuracies.json") as f:
            all_accuracies = json.load(f)
        plot(args, all_accuracies)
    else:
        all_accuracies, accelerator = inference(args)
        if accelerator.is_main_process:
            plot(args, all_accuracies)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model", type=str, default="output/LLaVA-NeXT-Video-7B-32K")
    args.add_argument("--max_frame_num", type=int, default=300)
    args.add_argument("--needle_dataset", type=str, default="lmms-lab/v_niah_needles")
    args.add_argument("--min_frame_num", type=int, default=20)
    args.add_argument("--frame_interval", type=int, default=20)
    args.add_argument("--output_path", type=str, default="vision_niah/niah_output")
    args.add_argument("--depth_interval", type=float, default=0.1)
    args.add_argument("--num_samples", type=int, default=1)
    args.add_argument("--rope_theta", type=float, default=None)
    args.add_argument(
        "--haystack_dir",
        type=str,
        default="video_needle_haystack/data/haystack_embeddings",
    )
    args.add_argument("--needle_embedding_dir", type=str, default="vision_niah/data/needle_embeddings")
    args.add_argument("--prompt_template", type=str)
    args.add_argument("--replace_double_newline", action="store_true")
    args.add_argument("--plot_only", action="store_true")

    main(args.parse_args())


