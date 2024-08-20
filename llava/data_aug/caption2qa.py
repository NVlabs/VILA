# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import json
import os
import os.path as osp
import shutil
import sys

import torch
import torch.distributed as dist
from filelock import FileLock, Timeout
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

task2prompt = {
    "cap2qa": """Below is an image description. Please propose 3 questions and answers based on the context. Each line should start with either "question" or "answer" and there should be only single linebreak between question and answer.\n\n""",
    "rephrase": """Below is an image description. Please rephrease the sentences make the writing more professional.\n\n""",
}


def safely_merge_info(out_fpath, info):
    os.makedirs(osp.dirname(out_fpath), exist_ok=True)
    with FileLock(out_fpath.replace(".json", ".lock")):
        if osp.exists(out_fpath):
            new_info = json.load(
                open(out_fpath, "r+"),
            )
            info.update(new_info)
        json.dump(info, open(out_fpath + ".meta", "w+"), indent=2)
        shutil.move(out_fpath + ".meta", out_fpath)
    return info


def process_caption(msg, task):
    # msg = v['output']
    segs = []
    d = set()
    for seg in msg.split("."):
        # repeatition detect
        if seg.lower() in d:
            break
        d.add(seg.lower())
        segs.append(seg)
    caption = ".".join(segs)
    return task2prompt[task] + caption


class Cap2QADataset(Dataset):
    def __init__(self, data_path="captioner/coyo-25m-recap/coyo25m-0-000000.tar.json", task="cap2qa") -> None:
        caption_json = json.load(open(data_path))
        self.captions = list(caption_json.items())
        self.task = task

    def __getitem__(self, index):
        k, v = self.captions[index]
        v["cap2llm"] = process_caption(v["output"], task=self.task)
        return k, v

    def __len__(self):
        return len(self.captions)


generation_config = {
    "temperature": 0.2,
    "top_p": 0.6,
    "do_sample": True,
    "max_new_tokens": 1024,
}


def main(
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    data_path="captioner/coyo-25m-recap/coyo25m-0-000000.tar.json",
    load_in_4bit=False,
    task="cap2qa",
):
    dist.init_process_group()

    # from llava.train.slurm_utils import (get_local_rank, get_rank,
    #                                      get_world_size)
    # local_rank, rank, world_size = get_local_rank(), get_rank(), get_world_size()
    # print(local_rank, rank, world_size, flush=True)
    local_rank = dist.get_rank()

    dst = Cap2QADataset(data_path=data_path, task=task)
    dloader = DataLoader(dst, batch_size=2, sampler=DistributedSampler(dst))

    output_json = {}

    save_folder = "captioner_bk"
    save_folder = osp.join(save_folder, task, model_id.replace("/", "--"))
    # output_path = osp.join(save_folder, data_path.replace(".json", f"-{rank}-of-{world_size}.json"))
    output_path = osp.join(save_folder, osp.basename(data_path))
    print("[DEBUG] ", data_path, "==>", output_path, flush=True)
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    # print("[DEBUG]", output_path, output_json, flush=True)
    output_json = safely_merge_info(output_path, output_json)
    # return 0

    pipe = pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={
            "torch_dtype": torch.float16,
            "load_in_4bit": load_in_4bit,
            "device_map": f"cuda:{local_rank}",
        },  # "device_map": "auto"},
        return_full_text=False,
        repetition_penalty=1.0,
    )

    for idx, (k, v) in enumerate(dloader):
        input_msg = v["cap2llm"]

        if all([url in output_json for url in k]):
            print(f"[{idx}-of-{len(dloader)}] already labeled, skip")
            continue

        result = pipe(input_msg, **generation_config)
        print("---" * 20, f" {idx}-of-{len(dloader)} ", flush=True)
        # print(input_msg)
        # print("***" * 40)
        # print(result)
        for url, inp, out in zip(k, input_msg, result):
            print(url, inp, out[0]["generated_text"])
            output_json[url] = {
                "caption": inp,
                "QA": out[0]["generated_text"].strip(),
            }

        if idx % 20 == 0:
            output_json = safely_merge_info(output_path, output_json)

    # with open(output_path, "w") as fp:
    #     json.dump(output_json, fp, indent=2)
    output_json = safely_merge_info(output_path, output_json)


if __name__ == "__main__":
    import fire

    fire.Fire(main)

"""
srun --label -A llmservice_nlp_fm -N 1 \
    -p batch_block1,batch_block2,batch_block3 -t 4:00:00 \
    -J llmservice_nlp_fm:test2 --gpus-per-node 8 --exclusive \
    --pty torchrun --nproc-per-node 8  llava/data_aug/caption2qa.py --model_id=NousResearch/Llama-2-13b-chat-hf


JOBS_LIMIT=64  # Set your limit here
model_id=NousResearch/Llama-2-13b-chat-hf
for f in captioner/*.json; do
  while [ $(jobs -rp | wc -l) -ge $JOBS_LIMIT ]; do
    sleep 1
  done

  fname=$(echo $f | cut -d "/" -f 2)
  model=$(echo $model_id | cut -d "/" -f 2)

  # Replace this with your actual command
  echo "Processing task $f and running jobs $(jobs -rp | wc -l)"; \
  srun --label -A llmservice_nlp_fm -N 1 \
    -p batch_block1,batch_block2,batch_block3 -t 4:00:00 \
    -J llmservice_nlp_fm-dev:cap2qa-$fname-$model --gpus-per-node 8 --exclusive \
    -e slurm-logs/dev/$fname-$model-$j.err -o slurm-logs/dev/$fname-$model-$j.out \
    torchrun --nproc-per-node 8  llava/data_aug/caption2qa.py --data_path=$f --model_id=$model_id &
done
wait
"""
