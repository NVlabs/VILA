# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
from typing import Dict, Optional, Sequence, List
import warnings

import torch
import transformers

from transformers import HfArgumentParser, AutoTokenizer, AutoConfig, LlamaForCausalLM
from transformers.modeling_utils import unwrap_model
from transformers import set_seed

from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer
from llava.train.args import TrainingArguments, ModelArguments, DataArguments
from llava.train.callbacks.autoresume_callback import AutoResumeCallback

from llava import conversation as conversation_lib
from llava.data import make_supervised_data_module
from llava.model import *
from llava.train.utils import (
    get_checkpoint_path,
    prepare_config_for_training,
    vision_resolution_elevation,
    unit_test_rope_scaling,
    mprint,
)


local_rank = None

if "WANDB_PROJECT" not in os.environ:
    # Default to WANDB project "VILA".
    os.environ["WANDB_PROJECT"] = "VILA"


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir, _internal_call=True)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.run_name = training_args.output_dir.split("/")[-1]
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["mm_projector"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    set_seed(training_args.seed)

    resume_path, continue_training = get_checkpoint_path(training_args.output_dir)

    if not continue_training:
        print(f"Models has been ready under {training_args.output_dir}. Skipp training")
        exit(0)

    if resume_path:
        resume_from_checkpoint = True
        config = AutoConfig.from_pretrained(resume_path, trust_remote_code=True)
        config.resume_path = resume_path
        model_cls = eval(config.architectures[0])
    else:
        ## first time training
        resume_from_checkpoint = False
        if "mpt" in model_args.model_name_or_path:
            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path, trust_remote_code=True
            )
            config.attn_config["attn_impl"] = training_args.mpt_attn_impl
            model_cls = LlavaMPTForCausalLM
        elif "mistral" in model_args.model_name_or_path.lower():
            config = LlavaMistralConfig.from_pretrained(model_args.model_name_or_path)
            config._attn_implementation = "flash_attention_2"
            model_cls = LlavaMistralForCausalLM
        elif "mixtral" in model_args.model_name_or_path.lower():
            config = LlavaMixtralConfig.from_pretrained(model_args.model_name_or_path)
            config._attn_implementation = "flash_attention_2"
            model_cls = LlavaMixtralForCausalLM
        elif "gemma" in model_args.model_name_or_path.lower():
            config = LlavaGemmaConfig.from_pretrained(model_args.model_name_or_path)
            config._attn_implementation = "flash_attention_2"
            model_cls = LlavaGemmaForCausalLM
        else:
            ## llm and default multimodal model
            model_cls = LlavaLlamaModel
            config = LlavaLlamaConfig.from_pretrained(
                model_args.model_name_or_path,
                resume=resume_from_checkpoint
            )
        if getattr(config, "resume_path", None) is not None:
            config.resume_path = model_args.model_name_or_path
    
    ## extra configurations
    prepare_config_for_training(config, model_args, training_args, data_args)

    model = model_cls(
        config=config,
        attn_implementation="flash_attention_2",
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args,
    )

    vision_resolution_elevation(model, config)
    # This is an empty func.
    # It would be overwritten by unit test script.
    if unit_test_rope_scaling(model, model.llm.config, training_args):
        return

    # Take a look on model architecture.
    mprint(model)

    model.llm.config.use_cache = False
    ## set tunnable parameters
    logging.warning(
        "You are setting tunable parameters for the model. Previous args include 'freeze_backbone' and 'tune_mm_mlp_adapter' are deprecated.\n Notice: default value of tune_xxx is False, which means you would not tune this part."
    )
    model.get_llm().requires_grad_(training_args.tune_language_model)
    mprint(f"Tunable parameters:\nlanguage model {training_args.tune_language_model}")
    if model.get_vision_tower():
        model.get_vision_tower().requires_grad_(training_args.tune_vision_tower)
        model.get_mm_projector().requires_grad_(training_args.tune_mm_projector)
        mprint(f"vision tower {training_args.tune_vision_tower}")
        mprint(f"mm projector {training_args.tune_mm_projector}")
    if not any([training_args.tune_language_model, training_args.tune_vision_tower, training_args.tune_mm_projector]):
        logging.warning(
            "You are not tuning any part of the model. Please check if this is intended."
        )
    def need_to_modify_do_sample(generation_config):
        if generation_config.do_sample is False:
            if (
                generation_config.temperature is not None
                and generation_config.temperature != 1.0
            ):
                return True
            if generation_config.top_p is not None and generation_config.top_p != 1.0:
                return True
        return False

    if need_to_modify_do_sample(model.llm.generation_config):
        model.llm.generation_config.do_sample = True

    ## quantize training @yunhao: be careful here
    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training

        model.llm.config.torch_dtype = (
            torch.float32
            if training_args.fp16
            else (torch.bfloat16 if training_args.bf16 else torch.float32)
        )
        model.llm = prepare_model_for_kbit_training(
            model.llm, use_gradient_checkpointing=training_args.gradient_checkpointing
        )

    if training_args.gradient_checkpointing:
        if hasattr(model.llm, "enable_input_require_grads"):
            model.llm.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        mprint("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
    # @yunhao: tokenizer instantiation is moved into build_llm
    tokenizer = model.tokenizer
    # @yunhao: may move this block into method "build_llm"
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model.llm,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model.llm,
            )
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]

    # kentang-mit@: It will be useful in on-the-fly packing
    model.llm.pad_token_id = tokenizer.pad_token_id
    model.llm.config.tokenizer_padding_side = tokenizer.padding_side
    model.llm.config.tokenizer_model_max_length = tokenizer.model_max_length
    if training_args.lora_enable:
        model.base_model.model.llm.pad_token_id = tokenizer.pad_token_id

    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.num_video_frames = data_args.num_video_frames
        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    ## TODO pay attention to quantize
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer

        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.float32)
            if "lm_head" in name or "embed_tokens" in name:
                if hasattr(module, "weight"):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    # Add a training step_end callback to check whether to autosuspend.
    callbacks = [AutoResumeCallback()]

    trainer = LLaVATrainer(
        model=model, tokenizer=tokenizer, args=training_args,
        callbacks=callbacks, **data_module
    )
    print(
        "length of dataloader:",
        len(trainer.get_train_dataloader()),
        len(trainer.train_dataset),
        flush=True,
    )
    print(
        "[GPU memory] before trainer",
        torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
        flush=True,
    )

    # print(resume_from_checkpoint); input("DEBUG")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_state()

    model.llm.config.use_cache = True
    model.config.resume_path = model.config._name_or_path = training_args.output_dir
    ## TODO handle lora for new initialization
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(
            trainer=trainer, output_dir=training_args.output_dir
        )


if __name__ == "__main__":
    train()
