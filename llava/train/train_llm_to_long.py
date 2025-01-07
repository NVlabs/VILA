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
# This file is modified from https://github.com/dvlab-research/LongLoRA

import copy
import math
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import datasets
import torch
import torch.distributed as dist
import transformers
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, PeftModel, get_peft_model
from ring_flash_attn.zigzag_ring_flash_attn import zigzag_ring_flash_attn_func
from torch.distributed import barrier
from torch.utils.data import DataLoader, Dataset
from transformers import DataCollatorForLanguageModeling, LlamaForCausalLM, Qwen2ForCausalLM, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available


def get_train_dataloader(self) -> DataLoader:
    """
    Returns the training [`~torch.utils.data.DataLoader`].

    Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
    training if necessary) otherwise.

    Subclass and override this method if you want to inject some custom behavior.
    """
    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")

    train_dataset = self.train_dataset
    data_collator = self.data_collator
    if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
        train_dataset = self._remove_unused_columns(train_dataset, description="training")
    else:
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

    dataloader_params = {
        "batch_size": self._train_batch_size,
        "collate_fn": data_collator,
        "num_workers": self.args.dataloader_num_workers,
        "pin_memory": self.args.dataloader_pin_memory,
        "persistent_workers": self.args.dataloader_persistent_workers,
    }

    if not isinstance(train_dataset, torch.utils.data.IterableDataset):
        dataloader_params["sampler"] = self._get_train_sampler()
        dataloader_params["drop_last"] = self.args.dataloader_drop_last
        dataloader_params["worker_init_fn"] = seed_worker

    return DataLoader(train_dataset, **dataloader_params)


Trainer.get_train_dataloader = get_train_dataloader


def extract_local(value, rank, world_size, device, dim=1):
    value_chunks = value.chunk(2 * world_size, dim=dim)
    local_value = torch.cat([value_chunks[rank], value_chunks[2 * world_size - rank - 1]], dim=dim)
    return local_value.to(device)


def ring_flash_attention_forward(
    self,
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    dropout=0.0,
    softmax_scale=None,
    seqlens_in_batch=None,
):
    attn_output = zigzag_ring_flash_attn_func(
        query_states, key_states, value_states, dropout, softmax_scale=softmax_scale, causal=self.is_causal
    )
    return attn_output


transformers.modeling_flash_attention_utils._flash_attention_forward = ring_flash_attention_forward

forward_qwen2_embed_ori = copy.deepcopy(Qwen2RotaryEmbedding.forward)


def forward_qwen2_embed(self, x, seq_len=None):
    seq_len = seq_len * dist.get_world_size()
    return forward_qwen2_embed_ori(self, x, seq_len)


Qwen2RotaryEmbedding.forward = forward_qwen2_embed


def judge_dir(resume_dir):
    is_checkpoint_dir = False
    if os.path.exists(resume_dir) == False:
        return False
    for _dir in os.listdir(resume_dir):
        if "checkpoint" in _dir:
            is_checkpoint_dir = True
        if "pth" in _dir:
            is_checkpoint_dir = True
    return is_checkpoint_dir


IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-1.4b-deduped")
    model_type: Optional[str] = field(default="llama")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=8192 * 4,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    data_max_length: int = field(
        default=80000,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    low_rank_training: bool = field(
        default=True,
        metadata={"help": "Whether use low rank adaptation for training."},
    )
    trainable_params: str = field(
        default="embed,norm",
        metadata={"help": "Additional trainable parameters except LoRA weights, if low rank training."},
    )
    resume_from_checkpoint: bool = field(
        default=True,
        metadata={"help": "Whether use flash attention for training."},
    )
    scaling_type: str = field(
        default="linear",
        metadata={"help": "Whether use flash attention for training."},
    )
    scaling_factor: int = field(
        default=1.0,
        metadata={"help": "Whether use flash attention for training."},
    )
    rope_theta: int = field(
        default=500000.0,
        metadata={"help": "Whether use flash attention for training."},
    )
    data_file: str = field(default="linear", metadata={"help": "Whether use flash attention for training."})
    peft_model: str = field(default=None, metadata={"help": "Whether use flash attention for training."})


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    print("len(tokenizer)", len(tokenizer))
    # model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def tokenize_fn(tokenizer, example):
    context_length = tokenizer.model_max_length
    outputs = tokenizer(
        tokenizer.eos_token.join(example["text"]),
        truncation=True,
        return_tensors="pt",
        pad_to_multiple_of=context_length,
        padding=True,
    )
    return {"input_ids": outputs["input_ids"].view(-1, context_length)}


def chunk_fn(tokenizer, example):
    input_ids = torch.tensor(example["text"], dtype=torch.int64)
    # world_size = 8 # dist.get_world_size()
    # input_ids = input_ids.unsqueeze(0).repeat(world_size, 1, 1).permute(1, 0, 2).reshape(-1, input_ids.shape[-1])
    return {"input_ids": input_ids}


def train():
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    architectures = config.architectures[0].lower()
    if "llama" in architectures:
        llm_model = LlamaForCausalLM
    elif "qwen" in architectures:
        llm_model = Qwen2ForCausalLM
    else:
        raise ValueError("Unsupported architecture %s." % architectures)

    forward_llm_ori = copy.deepcopy(llm_model.forward)

    def forward_llm(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        seq_len = input_ids.shape[-1]
        rank = dist.get_rank()
        num_processes = dist.get_world_size()
        input_ids = extract_local(input_ids, rank, num_processes, input_ids.device)
        labels = extract_local(labels, rank, num_processes, labels.device)
        position_ids = (
            torch.arange(seq_len, device=input_ids.device, dtype=torch.long).unsqueeze(0).expand(input_ids.shape[0], -1)
        )
        position_ids = extract_local(position_ids, rank, num_processes, position_ids.device)

        return forward_llm_ori(
            self=self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

    llm_model.forward = forward_llm
    config.rope_theta = training_args.rope_theta
    config.max_position_embeddings = training_args.model_max_length

    dataset = load_dataset("json", data_files=training_args.data_file, cache_dir=training_args.cache_dir)
    dataset = dataset.map(partial(chunk_fn, None), batched=True, num_proc=1, remove_columns=["text"])

    # Load model and tokenizer
    model = llm_model.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.data_max_length,
        padding_side="right",
        use_fast=True,
    )

    special_tokens_dict = dict()
    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN

    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=special_tokens_dict,
        tokenizer=tokenizer,
        model=model,
    )

    rank = int(os.environ.get("RANK", -1))
    if rank > 0:
        barrier()

    # dataset = load_dataset("yaofu/slimpajama-per-source-length-upsample", cache_dir=training_args.cache_dir)
    # dataset = dataset.map(partial(chunk_fn,tokenizer),batched=True, num_proc=16, remove_columns=["labels", "source"])
    # dataset = load_dataset('json', data_files=training_args.data_file, cache_dir=training_args.cache_dir)
    # dataset = dataset.map(partial(chunk_fn,tokenizer),batched=True, num_proc=2, remove_columns=["text"])
    # from IPython import embed; embed()

    if rank == 0:
        barrier()

    print(dataset)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if training_args.low_rank_training:
        if training_args.peft_model is None:
            if model_args.model_type == "gpt-neox":
                # added `dense` to match with llama as the basic LoRA would only target 'query_key_value'
                targets = ["query_key_value", "dense"]
            else:
                targets = ["q_proj", "k_proj", "v_proj", "o_proj"]

            config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=targets,
                lora_dropout=0,
                bias="none",
                modules_to_save=training_args.trainable_params.split(","),
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)
            # enable trainable params
            # [p.requires_grad_() for n, p in model.named_parameters() if any([k in n for k in training_args.trainable_params.split(",")])]
        else:
            model = PeftModel.from_pretrained(
                model,
                training_args.peft_model,
                torch_dtype=torch.bfloat16,
            )
            for n, p in model.named_parameters():
                if "lora" in n or any([k in n for k in training_args.trainable_params.split(",")]):
                    if not "original_module" in n:
                        p.requires_grad_()
                if p.requires_grad:
                    print(n)

    model.config.use_cache = False  # required for gradient checkpointing
    model.enable_input_require_grads()  # required for gradient checkpointing
    model.gradient_checkpointing_enable()  # enable gradient checkpointing
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=None,
        data_collator=data_collator,
    )

    if training_args.resume_from_checkpoint and judge_dir(training_args.output_dir):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()


