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

import copy
import logging
import math
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, LlamaForCausalLM, set_seed
from transformers.modeling_utils import unwrap_model

import llava.data.dataset as dataset
import llava.data.datasets_mixture as datasets_mixture
from llava import conversation as conversation_lib
from llava.constants import IGNORE_INDEX
from llava.data import make_supervised_data_module
from llava.mm_utils import process_image
from llava.model import LlavaLlamaConfig, LlavaLlamaModel
from llava.model.language_model.fp8linearqwen2 import Qwen2ForCausalLM  # We need this line to register AutoConfig
from llava.model.language_model.qllava_qllama import QLlavaLlamaModel, quantize_args_to_model_class
from llava.train.args import DataArguments, ModelArguments, TrainingArguments
from llava.train.callbacks.autoresume_callback import AutoResumeCallback
from llava.train.llava_trainer import LLaVATrainer, VILADPOTrainer
from llava.train.sequence_parallel import set_pg_manager
from llava.train.slurm_utils import TimeoutTerminateCallback
from llava.train.utils import (
    get_checkpoint_path,
    mprint,
    prepare_config_for_training,
    unit_test_rope_scaling,
    vision_resolution_elevation,
)
from llava.trl.trainer.utils import DPODataCollatorWithPadding

local_rank = None

if "WANDB_PROJECT" not in os.environ:
    # Default to WANDB project "VILA".
    os.environ["WANDB_PROJECT"] = "VILA"


def get_nb_trainable_parameters(model) -> tuple[int, int]:
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        # Due to the design of 4bit linear layers from bitsandbytes
        # one needs to multiply the number of parameters by 2 to get
        # the correct number of parameters
        if param.__class__.__name__ == "Params4bit":
            if hasattr(param, "element_size"):
                num_bytes = param.element_size()
            elif not hasattr(param, "quant_storage"):
                num_bytes = 1
            else:
                num_bytes = param.quant_storage.itemsize
            num_params = num_params * 2 * num_bytes

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params

    return trainable_params, all_param


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
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
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model, lora_llm, lora_vt):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_resampler"]
    assert lora_llm or lora_vt, "Not applying LoRA to any of the modules..."

    if not lora_llm:
        multimodal_keywords += ["llm"]
    if not lora_vt:
        multimodal_keywords += ["vision_tower"]

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            if not "lm_head" in name:
                lora_module_names.add(name)
            # names = name.split(".")
            # lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # if "lm_head" in lora_module_names:  # needed for 16-bit
    #     lora_module_names.remove("lm_head")
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

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def make_conv(prompt, answer):
    return [
        {
            "from": "human",
            "value": prompt,
        },
        {
            "from": "gpt",
            "value": answer,
        },
    ]


@dataclass
class DPODataCollator(DPODataCollatorWithPadding):
    tokenizer: Any = None

    def collate(self, batch):
        # first, pad everything to the same length
        # input_ids, labels = tuple([instance[key] for instance in instances]
        #                           for key in ("input_ids", "labels"))
        # input_ids = torch.nn.utils.rnn.pad_sequence(
        #     input_ids,
        #     batch_first=True,
        #     padding_value=self.tokenizer.pad_token_id)
        # labels = torch.nn.utils.rnn.pad_sequence(labels,
        #                                          batch_first=True,
        #                                          padding_value=IGNORE_INDEX)
        # input_ids = input_ids[:, :self.tokenizer.model_max_length]
        # labels = labels[:, :self.tokenizer.model_max_length]
        # batch = dict(
        #     input_ids=input_ids,
        #     labels=labels,
        #     attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        # )
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith("_input_ids") or k.endswith("_attention_mask") or k.endswith("_labels"):
                # if "prompt" in k:
                #     to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                # else:
                to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                if k.endswith("_input_ids"):
                    padding_value = self.pad_token_id
                elif k.endswith("_labels"):
                    padding_value = self.label_pad_token_id
                else:
                    continue
                # elif k.endswith("_attention_mask"):
                #     padding_value = self.padding_value
                # else:
                #     raise ValueError(f"Unexpected key in batch '{k}'")

                padded_batch[k] = torch.nn.utils.rnn.pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                # for the prompt, flip back so padding is on left side
                # if "prompt" in k:
                #     padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]
        for k in ["chosen_input_ids", "rejected_input_ids"]:
            attn_k = k.replace("input_ids", "attention_mask")
            padded_batch[attn_k] = padded_batch[k].ne(self.pad_token_id)
        return padded_batch

    def tokenize_batch_element(self, prompt: str, chosen: str, rejected: str) -> Dict:
        """Tokenize a single batch element.

        At this stage, we don't convert to PyTorch tensors yet; we just handle the truncation
            in case the prompt + chosen or prompt + rejected responses is/are too long. First
            we truncate the prompt; if we're still too long, we truncate the chosen/rejected.

        We also create the labels for the chosen/rejected responses, which are of length equal to
            the sum of the length of the prompt and the chosen/rejected response, with
            label_pad_token_id  for the prompt tokens.
        """
        # import pdb; pdb.set_trace()
        batch = {}

        chosen_sources = make_conv(prompt, chosen)
        rejected_sources = make_conv(prompt, rejected)
        chosen_data_dict = dataset.preprocess([chosen_sources], self.tokenizer, has_image=True)
        # chosen_data_dict['attention_mask'] = chosen_data_dict["input_ids"].ne(self.tokenizer.pad_token_id)

        rejected_data_dict = dataset.preprocess([rejected_sources], self.tokenizer, has_image=True)
        # rejected_data_dict['attention_mask'] = rejected_data_dict["input_ids"].ne(self.tokenizer.pad_token_id)

        chosen_data_dict = {k: v[0] for k, v in chosen_data_dict.items()}
        rejected_data_dict = {k: v[0] for k, v in rejected_data_dict.items()}

        for k, toks in {
            "chosen": chosen_data_dict,
            "rejected": rejected_data_dict,
        }.items():
            for type_key, tokens in toks.items():
                if type_key == "token_type_ids":
                    continue
                batch[f"{k}_{type_key}"] = tokens
        return batch

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokenized_batch = []
        Xs, keys = [], []
        for feature in features:
            prompt = feature["prompt"]
            chosen = feature["chosen"]
            rejected = feature["rejected"]

            batch_element = self.tokenize_batch_element(prompt, chosen, rejected)
            batch_element["images"] = feature["images"]
            tokenized_batch.append(batch_element)

        # return collated batch
        padded_batch = self.collate(tokenized_batch)
        return padded_batch


import json


def load_jsonl(save_path):
    with open(save_path) as f:
        data = [json.loads(line) for line in f.readlines()]
    return data


def load_json(path):
    with open(path) as f:
        data = json.load(f)
    return data


def load_data(data_path):
    if "jsonl" in data_path:
        data_list = load_jsonl(data_path)
    else:
        data_list = load_json(data_path)
    return data_list


class DPODataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_mixture: str, tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments):
        super(Dataset, self).__init__()
        data_path = datasets_mixture.DATASETS_LEGACY[data_mixture].data_path
        list_data_dict = load_data(data_path)
        # if data_args.num_sample is not None:
        #     list_data_dict = list_data_dict[:data_args.num_sample]

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.image_folder = datasets_mixture.DATASETS_LEGACY[data_mixture].image_path

    def __len__(self):
        # return 20
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(sum(len(conv["value"].split()) for conv in sample["conversations"]) + img_tokens)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        {
            'prompt': 'Is there a snowman wearing a green scarf and hat in the background?',
            'chosen': 'No, there is no snowman wearing a green scarf and hat in the background of the image. The image features a person ...',
            'rejected': 'No, there is no snowman in the background.',
            'image_path': '/mnt/bn/liangkeg/data/ruohongz/dpo_data/dpo_images/LRVInstruction-000000009569.jpg',
            'image_name': 'LRVInstruction-000000009569.jpg'
        }
        """
        # sources = self.list_data_dict[i]
        # if isinstance(i, int):
        #     sources = [sources]
        # assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        data_dict = copy.deepcopy(self.list_data_dict[i])  # inplace modification following

        video_file = data_dict["video"] + ".mp4"
        video_folder = self.image_folder
        video_path = os.path.join(video_folder, video_file)
        num_video_frames = self.data_args.num_video_frames if hasattr(self.data_args, "num_video_frames") else 8
        loader_fps = self.data_args.fps if hasattr(self.data_args, "fps") else 0.0

        fps = None
        frame_count = None

        images, frames_loaded = dataset.LazySupervisedDataset._load_video(
            video_path, num_video_frames, loader_fps, self.data_args, fps=fps, frame_count=frame_count
        )

        image_tensor = torch.stack([process_image(image, self.data_args, None) for image in images])
        image_tensor = torch.stack([process_image(image, self.data_args, None) for image in images])

        data_dict["images"] = image_tensor

        prompt = data_dict["prompt"]
        prompt = prompt.replace("<video>", "").strip()
        prompt = "<image>\n" * frames_loaded + prompt
        data_dict["prompt"] = prompt

        return data_dict


def train():
    global local_rank

    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # FIXME(zhijianl): This should be deprecated when we move to the new scripts.
    if os.getenv("RUN_NAME") is not None:
        training_args.run_name = os.getenv("RUN_NAME")
    else:
        training_args.run_name = training_args.output_dir.split("/")[-1]

    local_rank = training_args.local_rank
    compute_dtype = torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig

        bnb_model_from_pretrained_args.update(
            dict(
                device_map={"": training_args.device},
                # load_in_4bit=training_args.bits == 4,
                # load_in_8bit=training_args.bits == 8,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=training_args.bits == 4,
                    load_in_8bit=training_args.bits == 8,
                    llm_int8_skip_modules=["lm_head"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=compute_dtype,
                    bnb_4bit_use_double_quant=training_args.double_quant,
                    bnb_4bit_quant_type=training_args.quant_type,  # {'fp4', 'nf4'}
                ),
            )
        )

    set_seed(training_args.seed)

    sp_degree = training_args.seq_parallel_size
    ring_degree = training_args.seq_parallel_ring_size
    if sp_degree > 1:
        set_pg_manager(sp_degree, ring_degree, ring_type=training_args.seq_parallel_ring_type)
        print(f"Sequence parallelism is enabled, SP = {sp_degree}")

    resume_path, continue_training = get_checkpoint_path(training_args.output_dir)

    if not continue_training:
        print(f"Models has been ready under {training_args.output_dir}. Skipp training")
        exit(0)

    if resume_path:
        resume_from_checkpoint = True
        if training_args.lora_enable:
            model_cls = LlavaLlamaModel
            config = LlavaLlamaConfig.from_pretrained(model_args.model_name_or_path, resume=resume_from_checkpoint)
            config.resume_path = model_args.model_name_or_path
        else:
            config = AutoConfig.from_pretrained(resume_path, trust_remote_code=True)
            config.resume_path = resume_path
            model_cls = eval(config.architectures[0])
    else:
        ## first time training
        resume_from_checkpoint = False
        ## llm and default multimodal model
        if (
            model_args.quantize_model in quantize_args_to_model_class.keys()
        ):  # However, qmem should not used currently becuase I haven't merge the memory reduction version into VILA
            from llava.model.language_model.qllava_qllama import QLlavaLlamaModel

            model_cls = QLlavaLlamaModel
        else:
            assert (
                model_args.quantize_model == "false"
            ), f"{model_args.quantize_model} for model_args.quantize_model is not supported"
            model_cls = LlavaLlamaModel
        config = LlavaLlamaConfig.from_pretrained(model_args.model_name_or_path, resume=resume_from_checkpoint)

        if getattr(config, "resume_path", None) is not None:
            config.resume_path = model_args.model_name_or_path

    ## extra configurations
    prepare_config_for_training(config, model_args, training_args, data_args)
    if model_args.quantize_model in quantize_args_to_model_class.keys():
        model = model_cls(
            config=config,
            model_args=model_args,
            attn_implementation="flash_attention_2",
            model_max_length=training_args.model_max_length,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args,
        )
    else:
        model = model_cls(
            config=config,
            attn_implementation="flash_attention_2",
            model_max_length=training_args.model_max_length,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args,
        )

    if not resume_path or training_args.lora_enable:
        if model_args.mlp_path is not None:
            state_dict = torch.load(model_args.mlp_path, map_location="cpu")
            state_dict_new = {}
            for k, v in state_dict.items():
                if k == "0.weight":
                    state_dict_new["layers.1.weight"] = v
                if k == "0.bias":
                    state_dict_new["layers.1.bias"] = v
                if k == "1.weight":
                    state_dict_new["layers.2.weight"] = v
                if k == "1.bias":
                    state_dict_new["layers.2.bias"] = v
                if k == "3.weight":
                    state_dict_new["layers.4.weight"] = v
                if k == "3.bias":
                    state_dict_new["layers.4.bias"] = v
            model.get_mm_projector().load_state_dict(state_dict_new)

    vision_resolution_elevation(model, config)
    # This is an empty func.
    # It would be overwritten by unit test script.
    if unit_test_rope_scaling(model, model.llm.config, training_args):
        return

    # Take a look on model architecture.
    mprint(model)

    model.llm.config.use_cache = False

    ## set tunnable parameters
    # logging.warning(
    #     "You are setting tunable parameters for the model. Previous args include 'freeze_backbone' and 'tune_mm_mlp_adapter' are deprecated.\n Notice: default value of tune_xxx is False, which means you would not tune this part."
    # )

    def need_to_modify_do_sample(generation_config):
        if generation_config is None:
            warnings.warn("generation config is None, skip do sample modification")
            return False
        if generation_config.do_sample is False:
            if generation_config.temperature is not None and generation_config.temperature != 1.0:
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
            torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)
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
        from peft import LoraConfig, PeftModel, get_peft_model

        lora_config = LoraConfig(
            use_dora=training_args.use_dora,
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model, training_args.lora_llm, training_args.lora_vt),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        if resume_from_checkpoint:
            # load non-lora weights
            if os.path.exists(os.path.join(resume_path, "non_lora_trainables.bin")):
                non_lora_trainables = torch.load(
                    os.path.join(resume_path, "non_lora_trainables.bin"),
                    map_location="cpu",
                )
                non_lora_trainables = {
                    (k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()
                }
                if any(k.startswith("model.model.") for k in non_lora_trainables):
                    non_lora_trainables = {
                        (k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()
                    }
                model.load_state_dict(non_lora_trainables, strict=False)

            mprint("Resume from checkpoint...", resume_path)
            model = PeftModel.from_pretrained(model, resume_path, is_trainable=True)
        else:
            mprint("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)
        mprint(model)
        model.print_trainable_parameters()

    # currently assume fft for mm projector
    if training_args.lora_enable:
        if not training_args.lora_llm:
            model.get_llm().requires_grad_(training_args.tune_language_model)
        if model.get_vision_tower():
            if training_args.lora_vt:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_vision_tower().vision_tower.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )
            elif training_args.tune_vision_tower:
                model.get_vision_tower().requires_grad_(training_args.tune_vision_tower)
            model.get_mm_projector().requires_grad_(training_args.tune_mm_projector)
            mprint(f"mm projector {training_args.tune_mm_projector}")
            model.print_trainable_parameters()
    else:
        model.get_llm().requires_grad_(training_args.tune_language_model)
        mprint(f"Tunable parameters:\nlanguage model {training_args.tune_language_model}")
        if model.get_vision_tower():
            model.get_vision_tower().requires_grad_(training_args.tune_vision_tower)
            model.get_mm_projector().requires_grad_(training_args.tune_mm_projector)
            mprint(f"vision tower {training_args.tune_vision_tower}")
            mprint(f"mm projector {training_args.tune_mm_projector}")
            trainable_params, all_param = get_nb_trainable_parameters(model)
            print(
                f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
            )

        if not any(
            [training_args.tune_language_model, training_args.tune_vision_tower, training_args.tune_mm_projector]
        ):
            logging.warning("You are not tuning any part of the model. Please check if this is intended.")

    # @yunhao: tokenizer instantiation is moved into build_llm
    tokenizer = model.tokenizer

    if tokenizer.bos_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(bos_token="[BOS]"),
            tokenizer=tokenizer,
            model=model.llm,
        )

    # @yunhao: may move this block into method "build_llm"
    tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token="[PAD]"),
            tokenizer=tokenizer,
            model=model.llm,
        )
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    vision_tower = model.get_vision_tower()
    if vision_tower is not None:
        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        if hasattr(data_args, "num_video_frames") and data_args.num_video_frames != None:
            model.config.num_video_frames = data_args.num_video_frames
        else:
            model.config.num_video_frames = 8

        if hasattr(data_args, "fps"):
            model.config.fps = data_args.fps
        else:
            model.config.fps = 0.0

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.vision_tower_lr = training_args.vision_tower_lr
        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        assert not model_args.mm_use_im_patch_token

        model.config.num_time_tokens = data_args.num_time_tokens = model_args.num_time_tokens
        model.config.time_token_format = data_args.time_token_format = model_args.time_token_format
        if model_args.num_time_tokens > 0:
            time_tokens = [model.config.time_token_format.format(t=t) for t in range(model.config.num_time_tokens)]
            num_new_tokens = tokenizer.add_tokens(time_tokens)
            assert len(time_tokens) == num_new_tokens or num_new_tokens == 0
            model.resize_token_embeddings(len(tokenizer))
            model.config.time_token_ids = tokenizer.convert_tokens_to_ids(time_tokens)
        else:
            model.config.time_token_ids = []
        model.config.soft_ce_std = model_args.soft_ce_std

        num_patches = model.get_vision_tower().num_patches
        downsample_rate = model.get_mm_projector().downsample_rate
        num_image_tokens = math.ceil(num_patches**0.5 / downsample_rate) ** 2
        data_args.num_image_tokens = num_image_tokens

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

    data_args.s2_scales = list(map(int, model_args.s2_scales.split(",")))

    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )

    # Add a training step_end callback to check whether to autosuspend.
    callbacks = [AutoResumeCallback(), TimeoutTerminateCallback()]

    if training_args.dpo:
        ref_model = model_cls(
            config=config,
            attn_implementation="flash_attention_2",
            model_max_length=training_args.model_max_length,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args,
        )

        train_dataset = DPODataset(tokenizer=tokenizer, data_mixture=data_args.data_mixture, data_args=data_args)

        data_collator = DPODataCollator(
            tokenizer=tokenizer,
            label_pad_token_id=IGNORE_INDEX,
            pad_token_id=tokenizer.pad_token_id,
        )
        extra_info = []
        extra_info.append(len(train_dataset))
        training_args.sample_lens = extra_info

        trainer = VILADPOTrainer(
            model=model,
            dpo_alpha=1.0,
            gamma=0,
            ref_model=ref_model,
            tokenizer=tokenizer,
            args=training_args,
            beta=training_args.dpo_beta,
            callbacks=callbacks,
            train_dataset=train_dataset,
            data_collator=data_collator,
        )
    else:
        trainer = LLaVATrainer(model=model, tokenizer=tokenizer, args=training_args, callbacks=callbacks, **data_module)

        if model_args.quantize_model in ["fp8Activation_qwen2", "fp8ActivationResidual_qwen2"]:
            from llava.model.coat.fp8_trainer import CoatFP8Trainer

            trainer._inner_training_loop = CoatFP8Trainer._inner_training_loop.__get__(
                trainer, LLaVATrainer
            )  # GPT told me to do this

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

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    if training_args.debug_e2e:
        exit()

    trainer.save_state()

    model.llm.config.use_cache = True
    model.config.resume_path = model.config._name_or_path = training_args.output_dir
    ## TODO handle lora for new initialization
    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(
                non_lora_state_dict,
                os.path.join(training_args.output_dir, "non_lora_trainables.bin"),
            )
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()


