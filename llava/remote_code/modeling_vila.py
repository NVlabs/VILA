import copy
import json
import logging
import math
import os
import os.path
import os.path as osp
import shutil
import warnings
from abc import ABC
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from itertools import chain
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from einops import rearrange
from PIL import Image
from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessor,
    PretrainedConfig,
    PreTrainedModel,
    Qwen2Config,
    Qwen2ForCausalLM,
    Qwen2PreTrainedModel,
    TextIteratorStreamer,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import ContextManagers, no_init_weights

from .auto_processor import VILAProcessor
from .base_projector import MultimodalProjector, MultimodalProjectorConfig
from .builder import build_llm_and_tokenizer
from .configuration_vila import VILAConfig
from .constants import *
from .conversation import SeparatorStyle, default_conversation
from .distributed import all_gather as vila_all_gather
from .loss import soft_cross_entropy
from .media import extract_media
from .media_encoder import BasicImageEncoder, BasicVideoEncoder
from .mm_utils import process_image, process_images
from .model_utils_packing import set_seqlens_in_batch
from .siglip_encoder import SiglipVisionTower, SiglipVisionTowerDynamicS2, SiglipVisionTowerS2
from .tokenizer_utils import tokenize_conversation
from .utils import get_model_config, load_tokenizer_then_handle_media_tokens_and_chat_template

# from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, NUM_EXTRA_TOKENS

# ease debugging
python_input = input


# quick hack for remote code
def get_pg_manager():
    return None


def get_model_weights_dtype(model: nn.Module):
    pass


def build_mm_projector(model_type_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    if model_type_or_path is None:
        return None
    ## load from pretrained model
    if config.resume_path:
        assert os.path.exists(model_type_or_path), f"Resume mm projector path {model_type_or_path} does not exist!"
        return MultimodalProjector.from_pretrained(model_type_or_path, config)
    ## build from scratch
    else:
        mm_projector_cfg = MultimodalProjectorConfig(model_type_or_path)
        mm_projector = MultimodalProjector(mm_projector_cfg, config)
        return mm_projector


def check_dot_in_model_path(model_path: str):
    """Check if the model path contains dot, which will affect the remote code loading."""
    if osp.isdir(model_path):  # local model
        if "." in osp.abspath(model_path):
            return True
    else:  # remote model
        if "." in model_path:
            return True
    return False


def get_vila_version(model_path: str) -> str:
    VERSIONS = ["vila1.5", "vila-u", "longvila", "nvila", "vila-m3"]
    for version in VERSIONS:
        if version in model_path.lower():
            return version
    return None


def generate_jinja_template(conv_mode: str) -> str:
    if conv_mode == "vicuna_v1":
        return """{% set system_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. " %}
{% set roles = ["user", "assistant"] %}
{% set sep = " " %}

{{ system_prompt }}

{% for message in messages %}
    {% if message['role'] == roles[0] %}
        {{ "USER: " }}{{ sep }}{{ message['content'] }}{{ sep }}
    {% else %}
        {{ "ASSISTANT: " }}{{ sep }}{{ message['content'] }}{{ sep }}
    {% endif %}
{% endfor %}
{% if messages[-1]['role'] == 'user' %}
    {{ "ASSISTANT:" }}
{% endif %}
"""
    elif conv_mode == "llama_3":
        return """{% set system_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\nYou are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.<|eot_id|>" %}
{% set roles = ["<|start_header_id|>user<|end_header_id|>\\n\\n", "<|start_header_id|>assistant<|end_header_id|>\\n\\n"]%}
{% set sep = "<|eot_id|>" %}

{{ system_prompt }}
{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ roles[0] }}{{ message['content'] }}{{ sep }}
    {% else %}
        {{ roles[1] }}{{ message['content'] }}{{ sep }}
    {% endif %}
{% endfor %}
{% if messages[-1]['role'] == 'user' %}
    {{ roles[1] }}
{% endif %}
"""
    elif conv_mode == "hermes_2":
        return """{% set system_prompt = "<|im_start|>system\nAnswer the questions." %}
{% set roles = ["<|im_start|>user\n", "<|im_start|>assistant\n"] %}
{% set sep = "<|im_end|>" %}

{{ system_prompt }}{{ sep }}

{% for message in messages %}
    {% if message['role'] == 'user' %}
        {{ roles[0] }}{{ message['content'] }}{{ sep }}
    {% else %}
        {{ roles[1] }}{{ message['content'] }}{{ sep }}
    {% endif %}
{% endfor %}"""
    else:
        raise NotImplementedError(f"Jinja template generation is not implemented for {conv_mode}.")


def build_vision_tower(model_name_or_path: str, config: PretrainedConfig) -> PreTrainedModel:
    ## skip vision tower instantiation
    if model_name_or_path is None:
        return None

    vision_tower_arch = None
    if config.resume_path and "radio" not in model_name_or_path:
        assert os.path.exists(model_name_or_path), f"Resume vision tower path {model_name_or_path} does not exist!"
        vision_tower_cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        vision_tower_arch = vision_tower_cfg.architectures[0].lower()
    vision_tower_name = vision_tower_arch if vision_tower_arch is not None else model_name_or_path

    use_s2 = getattr(config, "s2", False)
    use_dynamic_s2 = getattr(config, "dynamic_s2", False)

    if "siglip" in vision_tower_name:
        if use_dynamic_s2:
            vision_tower = SiglipVisionTowerDynamicS2(model_name_or_path, config)
        elif use_s2:
            vision_tower = SiglipVisionTowerS2(model_name_or_path, config)
        else:
            vision_tower = SiglipVisionTower(model_name_or_path, config)
    else:
        raise NotImplementedError(f"Unknown vision tower: {model_name_or_path}")

    config.mm_hidden_size = (
        vision_tower.config.hidden_size if not (use_s2 or use_dynamic_s2) else vision_tower.hidden_size
    )
    return vision_tower


class VILAPretrainedModel(PreTrainedModel):
    config_class = VILAConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True
    _no_split_modules = ["Qwen2DecoderLayer", "SiglipEncoderLayer"]

    def __init__(self, config: VILAConfig, *args, **kwargs):
        super().__init__(config)
        self.config = config
        cfgs = get_model_config(config)
        if len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
        else:
            raise ValueError("`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config.")

        # loading on auto by default
        device_map = kwargs.get("device_map", "auto")
        self.mm_projector = build_mm_projector(mm_projector_cfg, config)
        self.vision_tower = build_vision_tower(vision_tower_cfg, config)
        if device_map in ["auto", "cuda"]:
            self.mm_projector = self.mm_projector.cuda()
            self.vision_tower = self.vision_tower.cuda()
        # set device_map auto can autoamtically shard llm to different devices
        self.llm, self.tokenizer = self.init_llm(llm_cfg, config, device_map=device_map)

        # NOTE(ligeng): hard code to set padding_side to left
        self.tokenizer.padding_side = "left"
        # TODO(ligeng): need to add other decoders from config
        self.encoders = {"image": BasicImageEncoder(self), "video": BasicVideoEncoder(self)}

        self.post_config()
        self.is_loaded = True

        assert (
            self.llm is not None or self.vision_tower is not None or self.mm_projector is not None
        ), "At least one of the components must be instantiated."

    @classmethod
    def convert_vila_dev_ckpt_to_remote(
        self,
        model_path: str,
        output_dir: str = None,
        vila_version: str | None = None,
        conv_mode: str | None = None,
        copy: bool = False,
        copy_weights: bool = True,
        copy_code: bool = True,
        *model_args,
        **kwargs,
    ):
        # assert type(self) == VILAForCasualLM, "This method is only available for VILAForCasualLM."
        assert model_path != output_dir, "model_path and output_dir cannot be the same"
        if os.path.isdir(model_path):
            model_path = model_path
        else:
            from huggingface_hub import HfApi, snapshot_download

            model_path = snapshot_download(model_path)
            print("downloading HF model to", model_path)

        if check_dot_in_model_path(model_path) and output_dir is None:
            raise ValueError(
                f"Model path {model_path} contains a dot, which will affect the remote code loading. Please specify the output directory without dot in the path to fix this issue."
            )
        if output_dir is not None and "." in output_dir:
            raise ValueError(
                f"Output directory {output_dir} contains a dot, which will affect the remote code loading. Please specify a valid output directory without dots."
            )

        if copy:
            print("copy is set to True, copying weights and code to output_dir")
            copy_weights = copy_code = True
        # copy weights and code to output_dir
        self.copy_or_symlink_directory(model_path, output_dir, copy=copy_weights)
        self.copy_remote_py_files(output_dir, copy=copy_code)

        if vila_version is None:
            vila_version = get_vila_version(output_dir)

        cfg_path = os.path.join(output_dir, "config.json")
        config = json.load(open(cfg_path))
        config["version"] = "2.0"  # nvila tag
        config["architectures"] = ["VILAForCausalLM"]
        config["auto_map"] = {
            "AutoProcessor": "auto_processor.VILAProcessor",
            "AutoConfig": "modeling_vila.VILAConfig",
            "AutoModel": "modeling_vila.VILAForCausalLM",
            "AutoModelForCausalLM": "modeling_vila.VILAForCausalLM",
        }
        # vila1.5 legacy support
        config["model_type"] = "vila"
        if vila_version in ["vila1.5", "vila-m3"]:
            if conv_mode is None:
                raise ValueError(f"Please specify the conversation mode for {output_dir}.")
            config["chat_template"] = conv_mode
            jinja_template = generate_jinja_template(conv_mode)
            jinja_path = os.path.join(output_dir, f"{conv_mode}.jinja")
            with open(jinja_path, "w") as f:
                f.write(jinja_template)
        json.dump(config, open(cfg_path, "w"), indent=2)

        ##########################################################################################
        config = AutoConfig.from_pretrained(output_dir, trust_remote_code=True)
        tokenizer = load_tokenizer_then_handle_media_tokens_and_chat_template(output_dir, config)
        tokenizer.save_pretrained(osp.join(output_dir, "llm"))
        ##########################################################################################

    @classmethod
    def copy_or_symlink_directory(cls, model_path, output_dir, copy=True):
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        # Create symlinks for all files in model_path to output_dir
        for item in os.listdir(model_path):
            src_path = os.path.join(model_path, item)
            dst_path = os.path.join(output_dir, item)

            # Remove existing file/directory at destination if it exists
            if os.path.exists(dst_path):
                if os.path.islink(dst_path):
                    os.unlink(dst_path)
                elif os.path.isdir(dst_path):
                    shutil.rmtree(dst_path)
                else:
                    os.remove(dst_path)

            # Create symlink
            if copy:
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path)
                else:
                    shutil.copy2(src_path, dst_path)
                print(f"Copied {src_path} to {dst_path}")
            else:
                os.symlink(src_path, dst_path)
                print(f"Created symlink from {src_path} to {dst_path}")

    @classmethod
    def copy_remote_py_files(cls, output_dir, copy=True):
        ## copy .py and REAMDE for next loading remote code
        current_file_path = os.path.abspath(__file__)
        current_folder = os.path.dirname(current_file_path)
        for file_name in os.listdir(current_folder):
            if file_name == "INSTRUCTIONS.md":
                src_fname = os.path.join(current_folder, file_name)
                dst_fname = os.path.join(output_dir, "README.md")
                if os.path.exists(dst_fname):
                    old_reamde = open(dst_fname).read()
                else:
                    old_reamde = ""
                with open(src_fname) as src, open(dst_fname, "w") as dst:
                    dst.write(src.read())
                    dst.write(old_reamde)
                print("[HF remote code] REAMDE ", src_fname, "to", dst_fname)
            if file_name.endswith(".py") or file_name.endswith(".jinja"):
                full_file_name = os.path.join(current_folder, file_name)
                if os.path.isfile(full_file_name):
                    if copy:
                        shutil.copy(full_file_name, output_dir)
                        print("[HF remote code] copying", full_file_name, "to", output_dir)
                    else:
                        # symlink to ease development
                        if os.path.exists(os.path.join(output_dir, file_name)):
                            os.remove(os.path.join(output_dir, file_name))
                        os.symlink(full_file_name, os.path.join(output_dir, file_name))
                        print("[HF remote code] linking", full_file_name, "to", output_dir)

    def save_pretrained(self, output_dir, state_dict=None, **kwargs):
        if state_dict is None:
            # other wise fetch from deepspeed
            # state_dict = accelerator.get_state_dict(is_deepspeed_enabled)
            state_dict = self.state_dict()

        if getattr(self, "tokenizer", None):
            self.tokenizer.save_pretrained(osp.join(output_dir, "llm"))

        if self.get_llm():
            print(f"saving llm to {osp.join(output_dir, 'llm')}")
            self.llm.config._name_or_path = osp.join(output_dir, "llm")
            llm_state_dict = OrderedDict({k.split("llm.")[-1]: v for k, v in state_dict.items() if "llm" in k})
            self.llm.save_pretrained(os.path.join(output_dir, "llm"), state_dict=llm_state_dict)
            self.config.llm_cfg = self.llm.config

        if self.get_vision_tower():
            print(f"saving vision_tower to {osp.join(output_dir, 'vision_tower')}")
            self.vision_tower.config._name_or_path = osp.join(output_dir, "vision_tower")
            vision_tower_state_dict = OrderedDict(
                {k.split("vision_tower.vision_tower.")[-1]: v for k, v in state_dict.items() if "vision_tower" in k}
            )
            self.vision_tower.vision_tower.save_pretrained(
                os.path.join(output_dir, "vision_tower"),
                state_dict=vision_tower_state_dict,
            )
            self.vision_tower.image_processor.save_pretrained(os.path.join(output_dir, "vision_tower"))
            self.config.vision_tower_cfg = self.vision_tower.config
            if hasattr(self.config.vision_tower_cfg, "auto_map"):
                if "radio" not in self.get_vision_tower().__class__.__name__.lower():
                    delattr(self.config.vision_tower_cfg, "auto_map")

        if self.get_mm_projector():
            print(f"saving mm_projector to {osp.join(output_dir, 'mm_projector')}")
            self.mm_projector.config._name_or_path = osp.join(output_dir, "mm_projector")
            mm_projector_state_dict = OrderedDict(
                {k.split("mm_projector.")[-1]: v for k, v in state_dict.items() if "mm_projector" in k}
            )
            self.mm_projector.save_pretrained(
                os.path.join(output_dir, "mm_projector"),
                state_dict=mm_projector_state_dict,
            )
            self.config.mm_projector_cfg = self.mm_projector.config

        ## update and save top-level config
        self.config._name_or_path = output_dir
        self.config.architectures = [self.__class__.__name__]
        self.config.save_pretrained(output_dir)

        ## copy .py and REAMDE for next loading remote code
        self.copy_remote_py_files(output_dir)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[str] = None,
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ):
        # print("DEBUG2", kwargs); input()
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
        return cls._from_config(config, **kwargs)

    def init_llm(self, llm_config, config, *args, **kwargs):
        self.llm, self.tokenizer = build_llm_and_tokenizer(llm_config, config, *args, **kwargs)
        # hard coded for NVILA
        # variables for XGrammar
        # print("DEBUG", len(self.tokenizer.added_tokens_encoder.keys()), self.tokenizer.added_tokens_encoder.keys())
        NUM_EXTRA_TOKENS = len(self.tokenizer.added_tokens_encoder.keys())

        self.pad_token_list = (
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.tokenize("<|endoftext|>")[0],  # for qwen
        )

        # TODO: SENTINEL_TOKEN is not added, need to check with Zhijian
        self.vocab_size = self.tokenizer.vocab_size + NUM_EXTRA_TOKENS
        # XGrammar tokenizer and grammar compiler
        # lazy init only when specified json output during inference
        self.grammar_compiler = None
        self.llm.resize_token_embeddings(len(self.tokenizer))
        return self.llm, self.tokenizer

    def post_config(self):
        ######################################################################
        # TODO: need to check dtype with jason
        self.llm = self.llm.to(torch.float16)
        self.mm_projector = self.mm_projector.to(torch.float16)
        self.vision_tower = self.vision_tower.to(torch.float16)
        ######################################################################
        self.training = self.llm.training
        if self.training:
            self.train()
        else:
            self.eval()
        ## configuration
        if getattr(self.config, "llm_cfg", None) is None:
            self.config.llm_cfg = self.llm.config
        if getattr(self.config, "vision_tower_cfg", None) is None:
            self.config.vision_tower_cfg = self.vision_tower.config
        if getattr(self.config, "mm_projector_cfg", None) is None:
            self.config.mm_projector_cfg = self.mm_projector.config

    def get_llm(self):
        llm = getattr(self, "llm", None)
        if type(llm) is list:
            llm = llm[0]
        return llm

    def get_lm_head(self):
        lm_head = getattr(self.get_llm(), "lm_head", None)
        return lm_head

    def get_vision_tower(self):
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_mm_projector(self):
        mm_projector = getattr(self, "mm_projector", None)
        if type(mm_projector) is list:
            mm_projector = mm_projector[0]
        return mm_projector

    def freezed_module_patch(self):
        """
        Huggingface will call model.train() at each training_step. To ensure the expected behaviors for modules like dropout, batchnorm, etc., we need to call model.eval() for the freezed modules.
        """
        if self.training:
            if self.get_llm() and not getattr(self.config, "tune_language_model", False):
                pass
                # logging.warning("Caution: Your LLM is currently in training mode, ensuring accurate gradient computation. Please be vigilant, particularly regarding BatchNorm and Dropout operations.")
            if self.get_vision_tower() and not getattr(self.config, "tune_vision_tower", False):
                self.get_vision_tower().eval()
            if self.get_mm_projector() and not getattr(self.config, "tune_mm_projector", False):
                self.get_mm_projector().eval()


class VILAForCausalLM(VILAPretrainedModel):
    def __init__(self, config: VILAConfig, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def merge_features_for_dynamic_s2(self, image_features, block_sizes):
        scales = self.get_vision_tower().scales
        resize_output_to_scale_idx = self.get_vision_tower().resize_output_to_scale_idx

        image_features_each_image = []
        new_block_sizes = []
        block_cnt = 0
        for block_size_each_image in block_sizes:
            if block_size_each_image is None:
                cur_features = image_features[block_cnt : block_cnt + 1]
                cur_features = rearrange(cur_features, "1 (h w) c -> 1 c h w", h=int(cur_features.shape[1] ** 0.5))
                cur_features = cur_features.repeat(1, len(scales), 1, 1)
                image_features_each_image.append(cur_features)
                new_block_sizes.append((1, 1))
                block_cnt += 1
            else:
                cur_features_each_scale = []
                for scale in scales[:-1]:
                    num_blocks_this_scale = (scale // scales[0]) ** 2
                    cur_features_each_scale.append(
                        self.merge_chessboard(
                            image_features[block_cnt : block_cnt + num_blocks_this_scale],
                            num_split_h=scale // scales[0],
                            num_split_w=scale // scales[0],
                        )
                    )  # 1 * C * H * W
                    block_cnt += num_blocks_this_scale
                num_blocks_last_scale = block_size_each_image[0] * block_size_each_image[1]
                cur_features_each_scale.append(
                    self.merge_chessboard(
                        image_features[block_cnt : block_cnt + num_blocks_last_scale],
                        num_split_h=block_size_each_image[0],
                        num_split_w=block_size_each_image[1],
                    )
                )  # 1 * C * H * W
                block_cnt += num_blocks_last_scale

                # resize and concat features from different scales
                output_size = cur_features_each_scale[resize_output_to_scale_idx].shape[-2:]
                cur_features = torch.cat(
                    [
                        F.interpolate(cur_features_each_scale[i].to(torch.float32), size=output_size, mode="area").to(
                            cur_features_each_scale[i].dtype
                        )
                        for i in range(len(cur_features_each_scale))
                    ],
                    dim=1,
                )
                # cur_features = rearrange(cur_features, "1 c h w -> (h w) c")

                image_features_each_image.append(cur_features)

                if resize_output_to_scale_idx == len(scales) - 1 or resize_output_to_scale_idx == -1:
                    new_block_sizes.append(block_size_each_image)
                else:
                    new_block_sizes.append(
                        (
                            scales[resize_output_to_scale_idx] // scales[0],
                            scales[resize_output_to_scale_idx] // scales[0],
                        )
                    )

        assert block_cnt == len(image_features)

        return image_features_each_image, new_block_sizes

    def encode_images(self, images, block_sizes: Optional[Optional[Tuple[int, ...]]] = None):
        if block_sizes is None:
            block_sizes = [None] * len(images)
        if getattr(self.config, "dynamic_s2", False):
            image_features = self.get_vision_tower()(images)
            image_features, new_block_sizes = self.merge_features_for_dynamic_s2(image_features, block_sizes)

            image_features = [
                self.split_chessboard(x, block_size[0], block_size[1])
                for x, block_size in zip(image_features, new_block_sizes)
            ]  # list of B * C * H * W tensors
            image_features = torch.cat(
                [rearrange(x, "b c h w -> b (h w) c") for x in image_features], dim=0
            )  # B * N * C
            image_features = self.get_mm_projector()(image_features)
            image_features = list(
                image_features.split([block_size[0] * block_size[1] for block_size in new_block_sizes], dim=0)
            )
            image_features = [
                self.merge_chessboard(x, block_size[0], block_size[1])
                for x, block_size in zip(image_features, new_block_sizes)
            ]  # list of 1 * C * H * W tensors
            image_features = [rearrange(x, "1 c h w -> (h w) c") for x in image_features]  # list of N * C tensors
            if all([feature.shape[0] == image_features[0].shape[0] for feature in image_features]):
                image_features = torch.stack(image_features, dim=0)
        else:
            image_features = self.get_vision_tower()(images)
            image_features = self.get_mm_projector()(image_features)
        return image_features

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def _embed(
        self,
        input_ids: torch.Tensor,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NOTE(ligeng): deep copy to avoid modifying the original media and media_config
        media = copy.deepcopy(media)
        media_config = copy.deepcopy(media_config)

        labels = labels if labels is not None else torch.full_like(input_ids, IGNORE_INDEX)
        attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)

        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            for name in media:
                self.encoders[name].end_tokens = None

        # Extract text and media embeddings
        text_embeds = self.llm.model.embed_tokens(input_ids)
        if media is not None:
            media_embeds = self.__embed_media_tokens(media, media_config)
        else:
            # no media was provided, so we just return an empty dict
            media_embeds = {}

        # This is a workaround to make sure the dummy embeddings are consumed
        while media_embeds.get("dummy"):
            dummy_embed = media_embeds["dummy"].popleft()
            text_embeds += torch.sum(dummy_embed) * 0

        # Remove padding
        batch_size = labels.shape[0]
        text_embeds = [text_embeds[k][attention_mask[k]] for k in range(batch_size)]
        labels = [labels[k][attention_mask[k]] for k in range(batch_size)]

        # Build inverse mapping from token ID to media name
        media_tokens = {}
        for name, token_id in self.tokenizer.media_token_ids.items():
            media_tokens[token_id] = name

        # Fuse text and media embeddings
        inputs_m, labels_m = [], []
        for k in range(batch_size):
            inputs_mk, labels_mk = [], []
            pos = 0
            while pos < len(labels[k]):
                if input_ids[k][pos].item() in media_tokens:
                    end = pos + 1
                    name = media_tokens[input_ids[k][pos].item()]
                    input = media_embeds[name].popleft()
                    label = torch.full([input.shape[0]], IGNORE_INDEX, device=labels[k].device, dtype=labels[k].dtype)
                elif input_ids[k][pos].item() in self.pad_token_list:
                    # skip pad tokens
                    end = pos + 1
                    pos = end
                    continue
                else:
                    end = pos
                    while end < len(labels[k]) and input_ids[k][end].item() not in media_tokens:
                        end += 1
                    input = text_embeds[k][pos:end]
                    label = labels[k][pos:end]

                inputs_mk.append(input)
                labels_mk.append(label)
                pos = end
            inputs_m.append(torch.cat(inputs_mk, dim=0))
            labels_m.append(torch.cat(labels_mk, dim=0))
        inputs, labels = inputs_m, labels_m

        # Check if all media embeddings are consumed
        for name in media_embeds:
            if media_embeds[name]:
                raise ValueError(f"Not all {name} embeddings are consumed! Still {len(media_embeds[name])} left.")

        # Truncate sequences to `model_max_length` as media embeddings are inserted
        inputs, labels = self.__truncate_sequence(inputs, labels)

        # Pad sequences to the longest one in the batch
        return self.__batchify_sequence(inputs, labels)

    def __embed_media_tokens(
        self,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
    ) -> Dict[str, List[torch.Tensor]]:
        embeds = defaultdict(deque)
        for name in media:
            if self.training:
                # Gather metainfo of media objects from all ranks
                info = [{"shape": tensor.shape, "dtype": tensor.dtype} for tensor in media.get(name, [])]
                infos = list(chain(vila_all_gather(info)))

                # The entire batch does not contain any media objects of this type.
                if not infos:
                    continue

                # Create a dummy tensor to ensure the encoder is called, otherwise the training will hang.
                if media.get(name) is None or len(media[name]) == 0:
                    dummy = torch.zeros(infos[0]["shape"], dtype=infos[0]["dtype"], device=self.device)
                    embeds["dummy"].extend(self.encoders[name]([dummy], media_config[name]))
                    continue
            embeds[name] = deque(self.encoders[name](media[name], media_config[name]))
        return embeds

    def __truncate_sequence(
        self, inputs: List[torch.Tensor], labels: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.training and any(len(input) > self.tokenizer.model_max_length for input in inputs):
            warnings.warn(f"Truncating sequences to `model_max_length` ({self.tokenizer.model_max_length}).")
            inputs = [input[: self.tokenizer.model_max_length] for input in inputs]
            labels = [label[: self.tokenizer.model_max_length] for label in labels]
        return inputs, labels

    def __batchify_sequence(
        self, inputs: List[torch.Tensor], labels: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = len(inputs)
        device = inputs[0].device
        hidden_size = inputs[0].shape[1]
        max_length = max(inputs[k].shape[0] for k in range(batch_size))
        attention_mask = torch.ones((batch_size, max_length), dtype=torch.bool, device=device)

        inputs_p, labels_p = [], []
        for k in range(batch_size):
            size_pk = max_length - inputs[k].shape[0]
            inputs_pk = torch.zeros((size_pk, hidden_size), dtype=inputs[k].dtype, device=device)
            labels_pk = torch.full((size_pk,), IGNORE_INDEX, dtype=labels[k].dtype, device=device)
            if self.tokenizer.padding_side == "right":
                attention_mask[k, inputs[k].shape[0] :] = False
                inputs_pk = torch.cat([inputs[k], inputs_pk], dim=0)
                labels_pk = torch.cat([labels[k], labels_pk], dim=0)
            else:
                attention_mask[k, : -inputs[k].shape[0]] = False
                inputs_pk = torch.cat([inputs_pk, inputs[k]], dim=0)
                labels_pk = torch.cat([labels_pk, labels[k]], dim=0)
            inputs_p.append(inputs_pk)
            labels_p.append(labels_pk)

        inputs = torch.stack(inputs_p, dim=0)
        labels = torch.stack(labels_p, dim=0)
        return inputs, labels, attention_mask

    def repack_multimodal_data(self, inputs_embeds, attention_mask, position_ids, labels):
        # Handle sequence parallelism
        PROCESS_GROUP_MANAGER = get_pg_manager()

        # We do re-sharding instead of packing here to ensure the sequence length is the same across all ranks.
        if PROCESS_GROUP_MANAGER is not None:
            sp_degree = PROCESS_GROUP_MANAGER.sp_degree
            sp_rank = PROCESS_GROUP_MANAGER.sp_rank
            sp_group = PROCESS_GROUP_MANAGER.sp_pg
            ring_degree = PROCESS_GROUP_MANAGER.ring_degree
            ring_rank = PROCESS_GROUP_MANAGER.ring_rank
            ring_type = PROCESS_GROUP_MANAGER.ring_type
            ulysses_degree = PROCESS_GROUP_MANAGER.ulysses_degree
            ulysses_rank = PROCESS_GROUP_MANAGER.ulysses_rank

            bs, shard_seqlen = position_ids.shape
            sp_seq_len = [torch.zeros(1, dtype=torch.int64, device=position_ids.device) for _ in range(sp_degree)]
            dist.all_gather(sp_seq_len, torch.tensor(shard_seqlen, device=position_ids.device), group=sp_group)
            sp_seq_len_cat = torch.cat(sp_seq_len, dim=0)

            if sp_rank == 0:
                original_start_id = 0
            else:
                original_start_id = torch.sum(sp_seq_len_cat[:sp_rank]).item()
            original_end_id = torch.sum(sp_seq_len_cat[: sp_rank + 1]).item()

            # Gather attention_mask, position_ids, labels and input_embeds
            all_inputs_embeds = torch.zeros(
                bs,
                torch.sum(sp_seq_len_cat),
                inputs_embeds.shape[-1],
                dtype=inputs_embeds.dtype,
                device=inputs_embeds.device,
            ).contiguous()
            all_inputs_embeds[:, original_start_id:original_end_id, :] += inputs_embeds
            dist.barrier(group=sp_group)
            dist.all_reduce(all_inputs_embeds, group=sp_group)
            dist.barrier(group=sp_group)

            attention_mask_list = [
                torch.zeros((bs, sp_seq_len[i]), dtype=attention_mask.dtype, device=attention_mask.device)
                for i in range(sp_degree)
            ]
            position_ids_list = [
                torch.zeros((bs, sp_seq_len[i]), dtype=position_ids.dtype, device=position_ids.device)
                for i in range(sp_degree)
            ]
            labels_list = [
                torch.zeros((bs, sp_seq_len[i]), dtype=labels.dtype, device=labels.device) for i in range(sp_degree)
            ]

            dist.all_gather(attention_mask_list, attention_mask, group=sp_group)
            dist.all_gather(position_ids_list, position_ids, group=sp_group)
            dist.all_gather(labels_list, labels, group=sp_group)

            effective_seqlen_list = [attention_mask_list[i].sum(dim=-1) for i in range(sp_degree)]
            effective_seqlen = torch.stack(effective_seqlen_list, dim=-1)
            effective_seqlen_batch_list = torch.unbind(effective_seqlen, dim=0)

            global_attention_mask_list = []
            global_position_ids_list = []
            global_labels_list = []
            global_inputs_embeds_list = []
            for i in range(bs):
                global_attention_mask_batch_list = []
                global_position_ids_batch_list = []
                global_labels_batch_list = []
                global_inputs_embeds_batch_list = []
                for j in range(sp_degree):
                    eff_len = effective_seqlen_batch_list[i][j]
                    prev_len = torch.sum(sp_seq_len_cat[:j]).item() if j > 0 else 0

                    global_attention_mask_batch_list.append(attention_mask_list[j][i, :eff_len])
                    global_position_ids_batch_list.append(position_ids_list[j][i, :eff_len])
                    global_labels_batch_list.append(labels_list[j][i, :eff_len])
                    global_inputs_embeds_batch_list.append(all_inputs_embeds[i, prev_len : prev_len + eff_len, :])
                global_attention_mask_list.append(torch.cat(global_attention_mask_batch_list, dim=0))
                global_position_ids_list.append(torch.cat(global_position_ids_batch_list, dim=0))
                global_labels_list.append(torch.cat(global_labels_batch_list, dim=0))
                global_inputs_embeds_list.append(torch.cat(global_inputs_embeds_batch_list, dim=0))

                global_attention_mask = torch.nn.utils.rnn.pad_sequence(
                    global_attention_mask_list, batch_first=True, padding_value=False
                )
                global_position_ids = torch.nn.utils.rnn.pad_sequence(
                    global_position_ids_list, batch_first=True, padding_value=-1
                )
                global_labels = torch.nn.utils.rnn.pad_sequence(
                    global_labels_list, batch_first=True, padding_value=IGNORE_INDEX
                )
                global_inputs_embeds = torch.nn.utils.rnn.pad_sequence(
                    global_inputs_embeds_list, batch_first=True, padding_value=0
                )

            # Re-shard the inputs
            if ring_degree > 1:
                total_effective_seqlen = torch.sum(effective_seqlen, dim=1)
                new_seqlen_per_rank = total_effective_seqlen // sp_degree
                assert torch.all(
                    total_effective_seqlen % sp_degree == 0
                ), "total_effective_seqlen must be divisible by sp_degree"

                max_new_seqlen = torch.max(new_seqlen_per_rank).item()

                new_attention_mask = torch.zeros(
                    (bs, max_new_seqlen), dtype=global_attention_mask.dtype, device=global_attention_mask.device
                )
                new_position_ids = torch.zeros(
                    (bs, max_new_seqlen), dtype=global_position_ids.dtype, device=global_position_ids.device
                )
                new_labels = torch.full(
                    (bs, max_new_seqlen), IGNORE_INDEX, dtype=global_labels.dtype, device=global_labels.device
                )
                new_inputs_embeds = torch.zeros(
                    (bs, max_new_seqlen, global_inputs_embeds.shape[-1]),
                    dtype=global_inputs_embeds.dtype,
                    device=global_inputs_embeds.device,
                )

                if ring_type == "ring_varlen":
                    for i in range(bs):
                        start_idx = new_seqlen_per_rank[i] * sp_rank
                        end_idx = start_idx + new_seqlen_per_rank[i]
                        new_attention_mask[i, : new_seqlen_per_rank[i]] = global_attention_mask[i, start_idx:end_idx]
                        new_position_ids[i, : new_seqlen_per_rank[i]] = global_position_ids[i, start_idx:end_idx]
                        new_labels[i, : new_seqlen_per_rank[i]] = global_labels[i, start_idx:end_idx]
                        new_inputs_embeds[i, : new_seqlen_per_rank[i], :] = global_inputs_embeds[
                            i, start_idx:end_idx, :
                        ]
                elif ring_type == "zigzag_ring_varlen":
                    chunk_size = total_effective_seqlen // (2 * sp_degree)
                    for i in range(bs):
                        # Zigzag pattern indices
                        if sp_degree == ring_degree:
                            forward_rank_idx = sp_rank
                            backward_rank_idx = 2 * sp_degree - sp_rank - 1
                        else:
                            ulysses_offset = ulysses_rank * ring_degree * 2
                            forward_rank_idx = ring_rank + ulysses_offset
                            backward_rank_idx = sp_degree - ring_rank - 1 + ulysses_offset

                        # Calculate start and end indices for the forward and backward zigzag
                        start_idx_fwd = forward_rank_idx * chunk_size[i]
                        end_idx_fwd = start_idx_fwd + chunk_size[i]

                        start_idx_bwd = backward_rank_idx * chunk_size[i]
                        end_idx_bwd = start_idx_bwd + chunk_size[i]

                        # Fill new tensors with zigzag data
                        new_attention_mask[i, : chunk_size[i]] = global_attention_mask[i, start_idx_fwd:end_idx_fwd]
                        new_attention_mask[i, chunk_size[i] : 2 * chunk_size[i]] = global_attention_mask[
                            i, start_idx_bwd:end_idx_bwd
                        ]

                        new_position_ids[i, : chunk_size[i]] = global_position_ids[i, start_idx_fwd:end_idx_fwd]
                        new_position_ids[i, chunk_size[i] : 2 * chunk_size[i]] = global_position_ids[
                            i, start_idx_bwd:end_idx_bwd
                        ]

                        new_labels[i, : chunk_size[i]] = global_labels[i, start_idx_fwd:end_idx_fwd]
                        new_labels[i, chunk_size[i] : 2 * chunk_size[i]] = global_labels[i, start_idx_bwd:end_idx_bwd]

                        new_inputs_embeds[i, : chunk_size[i], :] = global_inputs_embeds[i, start_idx_fwd:end_idx_fwd, :]
                        new_inputs_embeds[i, chunk_size[i] : 2 * chunk_size[i], :] = global_inputs_embeds[
                            i, start_idx_bwd:end_idx_bwd, :
                        ]
                else:
                    raise ValueError(f"Invalid ring_type: {ring_type}")
            else:
                global_seq_len = global_attention_mask.shape[-1]
                seq_len_sharded = global_seq_len // sp_degree
                start_idx_reshard = seq_len_sharded * sp_rank
                end_idx_reshard = start_idx_reshard + seq_len_sharded if sp_rank < sp_degree - 1 else global_seq_len

                new_attention_mask = torch.narrow(
                    global_attention_mask, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard
                )
                new_position_ids = torch.narrow(
                    global_position_ids, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard
                )
                new_labels = torch.narrow(global_labels, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard)
                new_inputs_embeds = torch.narrow(
                    global_inputs_embeds, 1, start_idx_reshard, end_idx_reshard - start_idx_reshard
                )

            return new_inputs_embeds, new_attention_mask, new_position_ids, new_labels

        device = inputs_embeds.device
        batch_size = inputs_embeds.shape[0]
        seqlens = [attention_mask[k].sum().item() for k in range(batch_size)]

        # Pack all sequences together
        inputs_embeds_p = [inputs_embeds[k][attention_mask[k]] for k in range(batch_size)]
        attention_mask_p = [torch.ones(seqlens[k], dtype=torch.int, device=device) for k in range(batch_size)]
        position_ids_p = [torch.arange(seqlens[k], dtype=torch.int, device=device) for k in range(batch_size)]
        labels_p = [labels[k][attention_mask[k]] for k in range(batch_size)]

        # Add one dummy token at the end of the packed sequence to ensure that `_get_unpacked_data` will be called
        inputs_embeds_p.append(torch.zeros(1, inputs_embeds.shape[-1], dtype=inputs_embeds.dtype, device=device))
        attention_mask_p.append(torch.tensor([0], dtype=torch.int, device=device))
        position_ids_p.append(torch.tensor([0], dtype=torch.int, device=device))
        labels_p.append(torch.tensor([IGNORE_INDEX], dtype=torch.int, device=device))

        # Mask the first token of each sequence to avoid contamination
        for label in labels_p:
            label[0] = IGNORE_INDEX

        # Batch the data
        inputs_embeds_p = torch.cat(inputs_embeds_p, dim=0).unsqueeze(0)
        attention_mask_p = torch.cat(attention_mask_p, dim=0).unsqueeze(0)
        position_ids_p = torch.cat(position_ids_p, dim=0).unsqueeze(0)
        labels_p = torch.cat(labels_p, dim=0).unsqueeze(0)

        if hasattr(
            self, "pad_to_multiple_of"
        ):  # related to quantization, please refer to ModelArguments for more information.
            assert len(labels_p.shape) == 2
            batch_size, max_length, cur_length = labels_p.shape[0], labels_p.shape[1], labels_p.shape[1]
            hidden_size = inputs_embeds_p.shape[-1]

            if max_length % self.pad_to_multiple_of != 0:
                max_length = ((max_length // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of
                difference = max_length - cur_length

                inputs_embeds_p = torch.cat(
                    (
                        inputs_embeds_p,
                        torch.full((batch_size, difference, hidden_size), self.llm.pad_token_id).to(inputs_embeds_p),
                    ),
                    dim=1,
                )
                labels_p = torch.cat((labels_p, torch.full((batch_size, difference), IGNORE_INDEX).to(labels_p)), dim=1)
                attention_mask_p = torch.cat(
                    (
                        attention_mask_p,
                        torch.zeros((batch_size, difference), dtype=torch.bool).to(attention_mask_p),
                    ),
                    dim=1,
                )
                position_ids_p = torch.cat(
                    (position_ids_p, torch.full((batch_size, difference), -1).to(position_ids_p)), dim=1
                )

        return inputs_embeds_p, attention_mask_p, position_ids_p, labels_p

    def get_xgr_logits_processor(self, response_format) -> List[LogitsProcessor]:
        raise NotImplementedError("This method is not implemented for VILA model.")
        # Convert response format to logits processor
        import xgrammar as xgr

        logging.info("[XGrammar] Compiling grammar for contrained output")

        if self.grammar_compiler is None:
            # logging.info(f"[XGrammar] {self.tokenizer}, {self.tokenizer.vocab_size}, {self.vocab_size}")
            self.grammar_compiler = xgr.GrammarCompiler(
                xgr.TokenizerInfo.from_huggingface(self.tokenizer, vocab_size=self.vocab_size)
            )

        if response_format.type == "json_schema":
            compiled_grammar = self.grammar_compiler.compile_json_schema(
                response_format.json_schema.schema_,
                indent=2,
            )
        else:
            compiled_grammar = self.grammar_compiler.compile_builtin_json_grammar()

        return [xgr.contrib.hf.LogitsProcessor(compiled_grammar)]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        media: Optional[Dict[str, List[torch.Tensor]]] = None,
        images: Optional[torch.FloatTensor] = None,
        media_config: Optional[List] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        packing: bool = True,
        force_packing: bool = False,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        dpo_forward: bool = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self.freezed_module_patch()

        if images is not None:
            if media is not None:
                raise ValueError("Both 'media' and 'images' are provided. Please provide only one.")
            print("The 'images' argument is deprecated. Please use 'media' instead.")
            media = {"image": images}

        if media_config is None:
            media_config = defaultdict(dict)

        if inputs_embeds is None:
            inputs_embeds, labels, attention_mask = self._embed(input_ids, media, media_config, labels, attention_mask)

        if force_packing or (packing and self.training and not dpo_forward):
            if seqlens_in_batch is None:
                seqlens_in_batch = torch.sum(attention_mask, dim=1)
            set_seqlens_in_batch(seqlens_in_batch)

            (inputs_embeds, attention_mask, position_ids, labels) = self.repack_multimodal_data(
                inputs_embeds, attention_mask, position_ids, labels
            )

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels,
            **kwargs,
        )

        if self.training and getattr(self.config, "time_token_ids", []):
            outputs.loss = soft_cross_entropy(
                outputs.logits,
                labels,
                soft_tokens=self.config.time_token_ids,
                std=self.config.soft_ce_std,
            )

        if dpo_forward:
            return outputs.logits, labels

        return outputs

    # TODO(ligeng): check how qwen implements this function
    # @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        media: Optional[Dict[str, List[torch.Tensor]]] = None,
        media_config: Dict[str, Dict[str, Any]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        return_output_ids_only: bool = False,
        **generation_kwargs,
    ) -> torch.LongTensor:
        """
        input_tokens: <image> describe the image
        media:        [Tensor(1, 3, 384, 384), ]
        ----------->
        input_tokens:      36000      001 002 003 004
        input_emds:     <media emd>   001 002 003 004
        """
        # NOTE: hard code to move to GPU
        # input_ids = input_ids.cuda()
        # media = {k: [v.cuda() if v is not None for v in media[k]] for k in media}
        # if attention_mask is not None:
        #     attention_mask = attention_mask.cuda()
        inputs_embeds, _, attention_mask = self._embed(input_ids, media, media_config, None, attention_mask)
        output_ids = self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generation_kwargs)

        if return_output_ids_only:
            return_value = output_ids
        else:
            # by default, return the input_ids and output_ids concatenated to keep consistency with the community VLMs like qwen
            generation_config = generation_kwargs.get("generation_config", None)
            if generation_config is not None:
                num_generations = generation_config.num_return_sequences
                repeat_input_ids = input_ids.repeat_interleave(num_generations, dim=0)
                return_value = torch.cat([repeat_input_ids, output_ids], dim=-1)
            else:
                return_value = torch.cat([input_ids, output_ids], dim=-1)

        return return_value

    @torch.inference_mode()
    def generate_content(
        self,
        prompt: Union[str, List],
        generation_config: Optional[GenerationConfig] = None,
        response_format=None,
    ) -> str:
        # TODO(zhijianl): Support directly taking conversation as input
        conversation = [{"from": "human", "value": prompt}]

        # Convert response format to logits processor
        xgr_logits_processor = None

        # Extract media from the conversation

        # TODO (extract and preprocess should be done together, as the preprocess of image and video can be different, i.e. when dynamic res is used)
        media = extract_media(conversation, self.config)

        # Process media
        media_config = defaultdict(dict)
        for name in media:
            if name == "image":
                if len(media["image"]) == 1 and self.config.image_aspect_ratio in ["dynamic", "dynamic_s2"]:
                    self.config.image_processor = self.vision_tower.image_processor
                    if self.config.image_aspect_ratio == "dynamic":
                        images = process_image(media["image"][0], self.config, None, enable_dynamic_res=True).half()
                        conversation[0]["value"] = conversation[0]["value"].replace(
                            DEFAULT_IMAGE_TOKEN, f"{DEFAULT_IMAGE_TOKEN}\n" * images.shape[0]
                        )
                    else:
                        if type(self.config.s2_scales) is str:
                            self.config.s2_scales = list(map(int, self.config.s2_scales.split(",")))
                        images, block_sizes = process_image(
                            media["image"][0], self.config, None, enable_dynamic_s2=True
                        )
                        images = images.half()
                        media_config[name]["block_sizes"] = [block_sizes]
                else:
                    images = process_images(media["image"], self.vision_tower.image_processor, self.config).half()
                media[name] = [image for image in images]
            elif name == "video":
                if self.config.image_aspect_ratio == "dynamic" and self.config.video_max_tiles > 1:
                    media[name] = [
                        process_images(
                            images,
                            self.vision_tower.image_processor,
                            self.config,
                            enable_dynamic_res=True,
                            max_tiles=self.config.video_max_tiles,
                        ).half()
                        for images in media[name]
                    ]
                elif self.config.image_aspect_ratio == "dynamic_s2" and self.config.video_max_tiles > 1:
                    self.config.image_processor = self.vision_tower.image_processor
                    if type(self.config.s2_scales) is str:
                        self.config.s2_scales = list(map(int, self.config.s2_scales.split(",")))
                    media[name] = [
                        torch.cat(
                            [
                                process_image(
                                    image,
                                    self.config,
                                    None,
                                    enable_dynamic_s2=True,
                                    max_tiles=self.config.video_max_tiles,
                                )[0].half()
                                for image in images
                            ]
                        )
                        for images in media[name]
                    ]
                else:
                    media[name] = [
                        process_images(images, self.vision_tower.image_processor, self.config).half()
                        for images in media[name]
                    ]
            else:
                raise ValueError(f"Unsupported media type: {name}")

        # Tokenize the conversation
        input_ids = tokenize_conversation(conversation, self.tokenizer, add_generation_prompt=True).unsqueeze(0).cuda()

        # Set up the generation config
        generation_config = generation_config or self.default_generation_config

        # print("input_ids", input_ids.shape)
        # print(input_ids)
        # print(self.tokenizer.batch_decode(input_ids))
        # print("media", {k: len(v) for k, v in media.items()})
        # print("media_config", media_config)
        # print("generation_config", generation_config)
        # input("wait for debug")
        # Generate the response
        try:
            output_ids = self.generate(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                generation_config=generation_config,
                logits_processor=xgr_logits_processor,  # structured generation
            )
        except ValueError:
            if not generation_config.do_sample:
                raise
            # FIXME(zhijianl): This is a temporary workaround for the sampling issue
            logging.warning("Generation failed with sampling, retrying with greedy decoding.")
            generation_config.do_sample = False
            output_ids = self.generate(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                generation_config=generation_config,
                logits_processor=xgr_logits_processor,
            )

        # Decode the response
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        return response

    @property
    def default_generation_config(self) -> GenerationConfig:
        generation_config = copy.deepcopy(self.generation_config or GenerationConfig())
        if self.tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer must have an EOS token")
        if generation_config.max_length == GenerationConfig().max_length:
            generation_config.max_length = self.tokenizer.model_max_length
        if generation_config.pad_token_id is None:
            generation_config.pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if generation_config.bos_token_id is None:
            generation_config.bos_token_id = self.tokenizer.bos_token_id or self.tokenizer.eos_token_id
        if generation_config.eos_token_id is None:
            generation_config.eos_token_id = self.tokenizer.eos_token_id
        return generation_config
