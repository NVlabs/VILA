#    Copyright 2023 Haotian Liu
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
import json
import logging
import os
import os.path as osp
import warnings
from abc import ABC
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from itertools import chain
from math import ceil, floor
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from hydra.utils import instantiate
from transformers import AutoConfig, GenerationConfig, LogitsProcessor, PreTrainedModel
from transformers.modeling_utils import ContextManagers, no_init_weights
from llava.constants import MEDIA_TOKENS

from llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, MEDIA_TOKENS, NUM_EXTRA_TOKENS
from llava.conversation import Conversation
from llava.mm_utils import process_image, process_images
from llava.model.configuration_llava import LlavaConfig, ResponseFormat
from llava.model.language_model.builder import build_llm_and_tokenizer
from llava.model.multimodal_encoder.builder import build_vision_tower
from llava.model.multimodal_projector.builder import build_mm_projector
from llava.model.utils import get_model_config
from llava.train.sequence_parallel import get_pg_manager
from llava.utils import distributed
from llava.utils.media import extract_media
from llava.utils.tokenizer import tokenize_conversation


class LlavaMetaModel(ABC):
    def _init_llm(self, llm_cfg, config, *args, **kwargs):
        llm, tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        return llm, tokenizer

    def init_vlm(self, config: PreTrainedModel = None, *args, **kwargs):
        # TODO(ligeng): figure out how from_config and from_pretrained works in HF implementation.
        if hasattr(self, "llm") or hasattr(self, "vision_tower") or hasattr(self, "mm_projector"):
            # already initialized, skipped
            return

        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype

        cfgs = get_model_config(config)
        if len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
        else:
            raise ValueError("`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config.")

        self.llm, self.tokenizer = self._init_llm(llm_cfg, config, *args, **kwargs)
        self.vision_tower = build_vision_tower(vision_tower_cfg, config)
        self.mm_projector = build_mm_projector(mm_projector_cfg, config)
        # NOTE(ligeng): for xgrammer init, <image> <vila/video> and <vila/sentinel>
        self.vocab_size = self.llm.vocab_size + NUM_EXTRA_TOKENS

        # XGrammar tokenizer and grammar compiler
        # lazy init only when specified json output during inference
        self.grammar_compiler = None

        self.encoders = {}
        for name in ["image", "video"]:
            config = getattr(self.config, f"{name}_encoder")
            if isinstance(config, str):
                config = json.loads(config)
            self.encoders[name] = instantiate(config, parent=self)

        self.post_config()
        self.is_loaded = True

        assert (
            self.llm is not None or self.vision_tower is not None or self.mm_projector is not None
        ), "At least one of the components must be instantiated."

    @classmethod
    def load_from_config(cls, model_path_or_config, *args, **kwargs):
        pass

    ## FIXME we will use this function to load model in the future
    @classmethod
    def load_pretrained(cls, model_path_or_config, *args, **kwargs):
        kwargs.pop("config", None)

        if isinstance(model_path_or_config, str):
            config = AutoConfig.from_pretrained(model_path_or_config)
        elif isinstance(model_path_or_config, LlavaConfig):
            config = model_path_or_config
        else:
            raise NotImplementedError(
                f"wrong type, {type(model_path_or_config)} \
                                      {isinstance(model_path_or_config, LlavaConfig)}"
            )

        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype

        cfgs = get_model_config(config)
        if len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
        else:
            raise ValueError("`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config.")

        # print(llm_cfg, vision_tower_cfg, mm_projector_cfg); input("DEBUG load_pretrained")
        init_context = [
            no_init_weights(_enable=True),
        ]
        # print("Before Init Context")
        # if hasattr(config, "deepspeed") and "mics" in config.deepspeed:
        #     print("Using MiCS_Init")
        #     import deepspeed
        #     init_context.append(deepspeed.zero.MiCS_Init(config_dict_or_path=config.deepspeed))
        with ContextManagers(init_context):
            vlm = cls(config, *args, **kwargs)
        # print(llm_cfg, vision_tower_cfg, mm_projector_cfg); input("DEBUG load_pretrained finish")

        if hasattr(vlm, "llm") or hasattr(vlm, "vision_tower") or hasattr(vlm, "mm_projector"):
            if vlm.is_loaded:
                return vlm

        vlm.llm, vlm.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        vlm.vision_tower = build_vision_tower(vision_tower_cfg, config)
        vlm.mm_projector = build_mm_projector(mm_projector_cfg, config)

        self.post_config()
        self.is_loaded = True

        # FIXME(ligeng, yunhao): llm should never be none here.
        assert (
            vlm.llm is not None or vlm.vision_tower is not None or vlm.mm_projector is not None
        ), "At least one of the components must be instantiated."
        return vlm

    ## FIXME we will use this function to save the model in the future
    def save_pretrained(self, output_dir, state_dict=None):
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

    def get_top_down_prompt_head(self):
        top_down_prompt_head = getattr(self.get_mm_projector(), "top_down_prompt_head", None)
        return top_down_prompt_head

    def post_config(self):
        self.training = self.get_llm().training
        ## configuration
        if getattr(self.config, "llm_cfg", None) is None:
            self.config.llm_cfg = self.llm.config
        if getattr(self.config, "vision_tower_cfg", None) is None:
            self.config.vision_tower_cfg = self.vision_tower.config
        if getattr(self.config, "mm_projector_cfg", None) is None:
            self.config.mm_projector_cfg = self.mm_projector.config

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

    @staticmethod
    def merge_chessboard(x, num_split_h, num_split_w):
        """
        x: b * n * c or b * h * w * c
        out: b * c * h * w
        Assuming x contains num_split**2 sub-squares concatenated along batch dimension, merge the sub-squares back to the original whole square.
        """
        B = x.shape[0]
        if x.dim() == 3:
            N = x.shape[1]
            x = rearrange(x, "b (h w) c -> b c h w", h=int(N**0.5), w=int(N**0.5))

        assert B % (num_split_h * num_split_w) == 0
        b = B // (num_split_h * num_split_w)

        x_merge = torch.cat(
            [
                torch.cat(
                    [x[(i * num_split_w + j) * b : (i * num_split_w + j + 1) * b] for j in range(num_split_w)], dim=-1
                )
                for i in range(num_split_h)
            ],
            dim=-2,
        )

        return x_merge

    @staticmethod
    def split_chessboard(x, num_split_h, num_split_w):
        """
        x: b * c * h * w
        out: b * c * h * w
        Deividing x into num_split**2 sub-squares, and concatenate all the sub-squares on the batch dimension
        """
        B, C, H, W = x.shape
        assert H % num_split_h == 0 and W % num_split_w == 0
        h, w = H // num_split_h, W // num_split_w
        x_split = torch.cat(
            [x[:, :, i * h : (i + 1) * h, j * w : (j + 1) * w] for i in range(num_split_h) for j in range(num_split_w)],
            dim=0,
        )
        return x_split

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

        assert block_cnt == len(
            image_features
        ), f"The number of blocks ({block_cnt}) does not match length of image_features ({len(image_features)})!"

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

    ## @yunhao: is there a better way to handle function call and attributes for llm?
    ## support beam search
    def _temporary_reorder_cache(self, past_key_values, sorted_idx):
        return self.get_llm()._temporary_reorder_cache(past_key_values, sorted_idx)

    def get_input_embeddings(self):
        return self.get_llm().get_input_embeddings()

    def get_output_embeddings(self):
        return self.get_llm().get_output_embeddings()

    def resize_token_embeddings(self, embed_size):
        self.get_llm().resize_token_embeddings(embed_size)


class LlavaMetaForCausalLM(ABC):
    def _embed(
        self,
        input_ids: torch.Tensor,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = labels if labels is not None else torch.full_like(input_ids, IGNORE_INDEX)
        attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)

        PROCESS_GROUP_MANAGER = get_pg_manager()
        if PROCESS_GROUP_MANAGER is not None:
            for name in media:
                self.encoders[name].end_tokens = None

        # Extract text and media embeddings
        text_embeds = self.llm.model.embed_tokens(input_ids)
        media_embeds = self.__embed_media_tokens(media, media_config)

        if PROCESS_GROUP_MANAGER is not None:
            media_embeds_video = []
            for i, images in enumerate(media_embeds["video"]):
                num_video_frame = media["video"][i].shape[0]
                if False:  # self.encoders["video"].pool_sizes:
                    pool_size = self.encoders["video"].pool_sizes[0][0]
                    num_video_frame = num_video_frame // pool_size * pool_size
                media_embeds_video += torch.unbind(images.reshape(num_video_frame, -1, images.shape[-1]))
            media_embeds["video"] = deque(media_embeds_video)

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
                    name = media_tokens[input_ids[k][pos].item()] if PROCESS_GROUP_MANAGER is None else "video"
                    input = media_embeds[name].popleft()
                    label = torch.full([input.shape[0]], IGNORE_INDEX, device=labels[k].device, dtype=labels[k].dtype)
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
                raise ValueError(f"Not all {name} embeddings are consumed!")

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
                infos = list(chain(*distributed.all_gather(info)))

                # The entire batch does not contain any media objects of this type.
                if not infos:
                    continue

                # Create a dummy tensor to ensure the encoder is called, otherwise the training will hang.
                if media.get(name) is None or len(media[name]) == 0:
                    dummy = torch.zeros(infos[0]["shape"], dtype=infos[0]["dtype"], device=self.device)
                    # Also add a dummy block_sizes
                    if name == "image" and "block_sizes" in media_config[name]:
                        media_config[name]["block_sizes"].append(None)
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

    def get_xgr_logits_processor(self, response_format: ResponseFormat) -> List[LogitsProcessor]:
        # Convert response format to logits processor
        import xgrammar as xgr

        logging.info("[XGrammar] Compiling grammar for contrained output")

        if self.grammar_compiler is None:
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

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        media: Optional[Dict[str, List[torch.Tensor]]] = None,
        media_config: Dict[str, Dict[str, Any]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generation_kwargs,
    ):
        inputs_embeds, _, attention_mask = self._embed(input_ids, media, media_config, None, attention_mask)
        return self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generation_kwargs)

    @torch.inference_mode()
    def generate_content(
        self,
        prompt: Union[str, List],
        generation_config: Optional[GenerationConfig] = None,
        response_format: Optional[ResponseFormat] = None,
    ) -> str:
        # TODO(zhijianl): Support directly taking conversation as input
        conversation = [{"from": "human", "value": prompt}]

        # Convert response format to logits processor
        if response_format:
            xgr_logits_processor = self.get_xgr_logits_processor(response_format)
        else:
            xgr_logits_processor = None

        # Extract media from the conversation


        # Process media
        media_config = defaultdict(dict)

        # TODO (extract and preprocess should be done together, as the preprocess of image and video can be different, i.e. when dynamic res is used)
        media = extract_media(conversation, config=self.config)
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
        input_ids = tokenize_conversation(conversation, self.tokenizer, add_generation_prompt=True).cuda().unsqueeze(0)

        # Set up the generation config
        generation_config = generation_config or self.default_generation_config

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
            generation_config.eos_token_id = self.tokenizer.stop_token_ids
        return generation_config


class LlavaTopDownMetaForCausalLM(LlavaMetaForCausalLM):
    def encode_images_ps3(
        self,
        images,
        image_num_each_sample=None,
        top_down_prompts=None,
        concat_low_high_res_features=False,
        smooth_selection_prob=False,
        num_look_close=None,
        gt_selection_maps=None,
        original_image_sizes=None,
    ):

        if original_image_sizes is not None and getattr(self.config, "ps3_dynamic_aspect_ratio", False):
            assert len(images) == len(original_image_sizes)
            thresholds = [
                (
                    self.get_vision_tower().vision_tower.vision_model.s3_scales[i]
                    + self.get_vision_tower().vision_tower.vision_model.s3_scales[i + 1]
                )
                / 2
                for i in range(1, len(self.get_vision_tower().vision_tower.vision_model.s3_scales) - 1)
            ]
            only_select_first_n_scale = []
            for i, (img, original_size) in enumerate(zip(images, original_image_sizes)):
                if (
                    max(original_size) > thresholds[-1]
                    or self.get_vision_tower().vision_tower.vision_model.s3_scales[-1] <= 2000
                ):
                    only_select_first_n_scale.append(len(thresholds) + 1)
                else:
                    only_select_first_n_scale.append(len(thresholds))
        else:
            only_select_first_n_scale = None

        if only_select_first_n_scale is not None and num_look_close != 0:
            # if only_select_first_n_scale is not None, then we split the input into different groups, where each group has the same only_select_first_n_scale
            # Currently we only expect 1 or 2 groups, corresponding only_select_first_n_scale = 2 (756 + 1512) or 3 (756 + 1512 + 3780).
            # If there is only 1 group, then we add a dummy input as the second group.
            # The purpose is to make sure every rank processes 2 groups, so that the number of times forward() is called is the same across different ranks. Otherwise zero3 will hang.
            assert all([x == 2 or x == 3 for x in only_select_first_n_scale])
            x_masks_each_group = [
                [x == 2 for x in only_select_first_n_scale],
                [x == 3 for x in only_select_first_n_scale],
            ]
            image_features_each_group = []
            selection_maps_each_group = []
            select_probs_each_group = []
            for x_mask in x_masks_each_group:
                if sum(x_mask) == 0:
                    x_mask = deepcopy(x_mask)
                    x_mask[0] = True
                if num_look_close is not None:
                    image_features, selection_maps, select_probs = self.get_vision_tower()(
                        images[x_mask],
                        top_down_prompts[x_mask] if top_down_prompts is not None else None,
                        num_look_close,
                        None,
                        smooth_selection_prob,
                        gt_selection_maps[x_mask] if gt_selection_maps is not None else None,
                        [only_select_first_n_scale[i] for i, m in enumerate(x_mask) if m],
                    )
                elif self.num_token_look_close is not None:
                    image_features, selection_maps, select_probs = self.get_vision_tower()(
                        images[x_mask],
                        top_down_prompts[x_mask] if top_down_prompts is not None else None,
                        None,
                        self.num_token_look_close,
                        smooth_selection_prob,
                        gt_selection_maps[x_mask] if gt_selection_maps is not None else None,
                        [only_select_first_n_scale[i] for i, m in enumerate(x_mask) if m],
                    )
                else:
                    image_features, selection_maps, select_probs = self.get_vision_tower()(
                        images[x_mask],
                        top_down_prompts[x_mask] if top_down_prompts is not None else None,
                        self.num_look_close,
                        None,
                        smooth_selection_prob,
                        gt_selection_maps[x_mask] if gt_selection_maps is not None else None,
                        [only_select_first_n_scale[i] for i, m in enumerate(x_mask) if m],
                    )
                image_features_each_group.append(image_features)
                selection_maps_each_group.append(selection_maps)
                select_probs_each_group.append(select_probs)

            image_features = torch.zeros(
                images.shape[0],
                *image_features_each_group[0].shape[1:],
                device=image_features_each_group[0].device,
                dtype=image_features_each_group[0].dtype,
            )
            selection_maps = [
                torch.zeros(
                    images.shape[0],
                    *selection_maps_each_group[0][i].shape[1:],
                    device=selection_maps_each_group[0][i].device,
                    dtype=selection_maps_each_group[0][i].dtype,
                )
                for i in range(len(selection_maps_each_group[0]))
            ]
            select_probs = [
                torch.zeros(
                    images.shape[0],
                    *select_probs_each_group[0][i].shape[1:],
                    device=select_probs_each_group[0][i].device,
                    dtype=select_probs_each_group[0][i].dtype,
                )
                for i in range(len(select_probs_each_group[0]))
            ]

            for i, x_mask in enumerate(x_masks_each_group):
                if sum(x_mask) == 0:
                    image_features += image_features_each_group[i].sum() * 0
                    for j in range(len(selection_maps)):
                        select_probs[j] += select_probs_each_group[i][j].sum() * 0
                else:
                    image_features[x_mask] = image_features_each_group[i]
                    for j in range(len(selection_maps)):
                        selection_maps[j][x_mask] = selection_maps_each_group[i][j]
                        select_probs[j][x_mask] = select_probs_each_group[i][j]
        else:
            if num_look_close is not None:
                image_features, selection_maps, select_probs = self.get_vision_tower()(
                    images,
                    top_down_prompts,
                    num_look_close,
                    None,
                    smooth_selection_prob,
                    gt_selection_maps,
                    only_select_first_n_scale,
                )
            elif self.num_token_look_close is not None:
                image_features, selection_maps, select_probs = self.get_vision_tower()(
                    images,
                    top_down_prompts,
                    None,
                    self.num_token_look_close,
                    smooth_selection_prob,
                    gt_selection_maps,
                    only_select_first_n_scale,
                )
            else:
                image_features, selection_maps, select_probs = self.get_vision_tower()(
                    images,
                    top_down_prompts,
                    self.num_look_close,
                    None,
                    smooth_selection_prob,
                    gt_selection_maps,
                    only_select_first_n_scale,
                )

        # Each image has different size when we choose to process on the original aspect ratio. Complete each selection map to be square
        # selection_maps: list of selection maps, each element in the list is a list of maps at different scales. Each map is 1 * H * W
        if type(image_features) is list:
            image_features = [x.squeeze(0) for x in image_features]
            padded_selection_maps = []
            for smaps in selection_maps:
                size_each_scale = [smap.shape[-2:] for smap in smaps]
                new_size_each_scale = [(max(size[0], size[1]), max(size[0], size[1])) for size in size_each_scale]
                smaps = [
                    F.pad(smap, (0, new_size[1] - size[1], 0, new_size[0] - size[0]))
                    for smap, new_size, size in zip(smaps, new_size_each_scale, size_each_scale)
                ]
                padded_selection_maps.append(smaps)
            selection_maps = [
                torch.cat([smaps[i] for smaps in padded_selection_maps], dim=0)
                for i in range(len(padded_selection_maps[0]))
            ]

        if "mlp_downsample" in self.mm_projector.config.mm_projector_type:
            if (
                self.mm_projector.config.mm_projector_type == "mlp_downsample"
                or self.mm_projector.config.mm_projector_type == "mlp_downsample_2x2_fix"
            ):
                downsample_ratio = 2
            elif (
                self.mm_projector.config.mm_projector_type == "mlp_downsample_3x3"
                or self.mm_projector.config.mm_projector_type == "mlp_downsample_3x3_fix"
            ):
                downsample_ratio = 3

            # pass low-res tokens to mm_projector
            low_res_token_num = self.get_vision_tower().vision_tower.vision_model.low_res_token_num
            low_res_image_features = torch.stack([x[:low_res_token_num] for x in image_features], dim=0)
            low_res_image_features = self.get_mm_projector()(low_res_image_features)

            # add pos embed and scale embed to low-res tokens
            if self.use_high_res_pos_embed:
                pe = rearrange(
                    self.get_mm_projector().high_res_pos_embed,
                    "b (h w) c -> b c h w",
                    h=int(self.get_mm_projector().high_res_pos_embed.shape[1] ** 0.5),
                )
                pe = F.interpolate(pe, int(low_res_image_features.shape[1] ** 0.5), mode="area")
                pe = rearrange(pe, "b c h w -> b (h w) c")
                low_res_image_features = low_res_image_features + pe + self.get_mm_projector().high_res_scale_embed[0]

            if num_look_close == 0:
                return low_res_image_features, selection_maps, select_probs

            # For high-res tokens, since they are not the full image, if the mm_project is mlp_downsample then it's probalematic to reshape and downsample it.
            B, C = len(image_features), image_features[0].shape[-1]
            selection_maps_concat = torch.cat([m.flatten(1, 2) for m in selection_maps], dim=1)
            high_res_feature = torch.zeros(
                B, selection_maps_concat.shape[1], C, device=image_features[0].device, dtype=image_features[0].dtype
            )
            for id in range(B):
                high_res_feature[id][selection_maps_concat[id].bool()] = image_features[id][low_res_token_num:]

            prev_num_token = 0
            processed_high_res_features = []
            for scale_id in range(len(selection_maps)):
                cur_feature = high_res_feature[
                    :,
                    prev_num_token : prev_num_token
                    + selection_maps[scale_id].shape[1] * selection_maps[scale_id].shape[2],
                ]
                cur_feature = self.get_mm_projector()(cur_feature, feature_size=selection_maps[scale_id].shape[-2:])
                processed_high_res_features.append(cur_feature)
                prev_num_token += selection_maps[scale_id].shape[1] * selection_maps[scale_id].shape[2]
            processed_high_res_features = torch.cat(processed_high_res_features, dim=1)

            # add pos embed and scale embed to high-res tokens
            if self.use_high_res_pos_embed:
                pe = rearrange(
                    self.get_mm_projector().high_res_pos_embed,
                    "b (h w) c -> b c h w",
                    h=int(self.get_mm_projector().high_res_pos_embed.shape[1] ** 0.5),
                )
                pe_each_scale = [
                    F.interpolate(
                        pe,
                        (ceil(smap.shape[-2] / downsample_ratio), ceil(smap.shape[-1] / downsample_ratio)),
                        mode="area",
                    )
                    for smap in selection_maps
                ]
                pe_each_scale = [rearrange(pe, "b c h w -> b (h w) c") for pe in pe_each_scale]
                pe_each_scale = [
                    pe + self.get_mm_projector().high_res_scale_embed[i + 1] for i, pe in enumerate(pe_each_scale)
                ]
                high_res_pos_emb_flatten = torch.cat(pe_each_scale, dim=1)
                processed_high_res_features = processed_high_res_features + high_res_pos_emb_flatten

            downsampled_selection_maps = [
                F.interpolate(
                    m.unsqueeze(1),
                    (ceil(m.shape[-2] / downsample_ratio), ceil(m.shape[-1] / downsample_ratio)),
                    mode="area",
                )
                .squeeze(1)
                .flatten(1, 2)
                for m in selection_maps
            ]
            downsampled_selection_maps_concat = torch.cat(downsampled_selection_maps, dim=1)
            processed_high_res_features = [
                x[m == 1] for x, m in zip(processed_high_res_features, downsampled_selection_maps_concat)
            ]  # only keep the downsampled high-res token where all 2x2 original high-res tokens are selected

            processed_high_res_features = [
                x if x.shape[0] != 0 else low_res_image_features[i, :1]
                for i, x in enumerate(processed_high_res_features)
            ]  # in case no 2x2 original high-res tokens are selected

            # If look_close = 0, then throw away all high-res features
            if num_look_close == 0 or (num_look_close is None and self.num_look_close == 0):
                processed_high_res_features = [
                    low_res_image_features[i, :1] for i in range(len(processed_high_res_features))
                ]

            # aggregate low-res and high-res features
            if concat_low_high_res_features:
                image_features = [
                    torch.cat([low_res_x, high_res_x], dim=0)
                    for low_res_x, high_res_x in zip(low_res_image_features, processed_high_res_features)
                ]
            else:
                if sum(image_num_each_sample) == 0:
                    image_features = [low_res_image_features[0][:1] + processed_high_res_features[0][:1]]
                    return image_features, selection_maps, select_probs
                assert len(low_res_image_features) == sum(image_num_each_sample) and len(
                    processed_high_res_features
                ) == sum(image_num_each_sample)
                image_features = []
                cnt = 0
                for num in image_num_each_sample:
                    image_features.extend([low_res_image_features[i] for i in range(cnt, cnt + num)])
                    image_features.extend([processed_high_res_features[i] for i in range(cnt, cnt + num)])
                    cnt += num

        else:
            assert type(image_features) is not list
            B = image_features.shape[0]
            low_res_token_num = self.get_vision_tower().vision_tower.vision_model.low_res_token_num
            low_res_image_features = image_features[:, :low_res_token_num]
            high_res_image_features = image_features[:, low_res_token_num:]
            low_res_image_features = self.get_mm_projector()(low_res_image_features)
            high_res_image_features = self.get_mm_projector()(high_res_image_features)

            # add pos embed and scale embed to low-res tokens
            if self.use_high_res_pos_embed:
                low_res_image_features = (
                    low_res_image_features
                    + self.get_mm_projector().high_res_pos_embed
                    + self.get_mm_projector().high_res_scale_embed[0]
                )

            if num_look_close == 0:
                return low_res_image_features, selection_maps, select_probs

            # add pos embed and scale embed to high-res tokens
            if self.use_high_res_pos_embed:
                pe = rearrange(
                    self.get_mm_projector().high_res_pos_embed,
                    "b (h w) c -> b c h w",
                    h=int(self.get_mm_projector().high_res_pos_embed.shape[1] ** 0.5),
                )
                pe_each_scale = [F.interpolate(pe, smap.shape[-2:], mode="area") for smap in selection_maps]
                pe_each_scale = [rearrange(pe, "b c h w -> b (h w) c") for pe in pe_each_scale]
                pe_each_scale = [
                    pe + self.get_mm_projector().high_res_scale_embed[i + 1] for i, pe in enumerate(pe_each_scale)
                ]
                high_res_pos_emb_flatten = torch.cat(pe_each_scale, dim=1)
                selection_maps_flatten = torch.cat([m.flatten(1, 2) for m in selection_maps], dim=1)
                high_res_pos_emb_selected = high_res_pos_emb_flatten[selection_maps_flatten == 1].reshape(
                    B, -1, high_res_pos_emb_flatten.shape[-1]
                )
                high_res_image_features = [
                    x + high_res_pos_emb_selected[i] for i, x in enumerate(high_res_image_features)
                ]

            # If look_close = 0, then throw away all high-res features
            if num_look_close == 0 or (num_look_close is None and self.num_look_close == 0):
                high_res_image_features = [low_res_image_features[i, :1] for i in range(len(high_res_image_features))]

            # aggregate low-res and high-res features
            if concat_low_high_res_features:
                image_features = [
                    torch.cat([low_res_x, high_res_x], dim=0)
                    for low_res_x, high_res_x in zip(low_res_image_features, processed_high_res_features)
                ]
            else:
                if sum(image_num_each_sample) == 0:
                    image_features = [low_res_image_features[0][:1] + high_res_image_features[0][:1]]
                    return image_features, selection_maps, select_probs
                assert len(low_res_image_features) == sum(image_num_each_sample) and len(
                    high_res_image_features
                ) == sum(image_num_each_sample)
                image_features = []
                cnt = 0
                for num in image_num_each_sample:
                    image_features.extend([low_res_image_features[i] for i in range(cnt, cnt + num)])
                    image_features.extend([high_res_image_features[i] for i in range(cnt, cnt + num)])
                    cnt += num

        return image_features, selection_maps, select_probs

    def _embed(
        self,
        input_ids: torch.Tensor,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
        labels: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        top_down_prompts=None,
        instance_id_with_image=None,
        smooth_selection_prob=False,
        num_look_close=None,
        gt_selection_maps=None,
        original_image_sizes=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        labels = labels if labels is not None else torch.full_like(input_ids, IGNORE_INDEX)
        attention_mask = attention_mask if attention_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)

        IMAGE_TOKEN_INDEX = self.tokenizer.media_token_ids["image"]
        num_images_each_instance = (input_ids == IMAGE_TOKEN_INDEX).sum(dim=-1)[instance_id_with_image]

        # Extract text embeddings
        text_embeds = self.llm.model.embed_tokens(input_ids)

        # Extract media embeddings
        if top_down_prompts is not None:
            if num_images_each_instance.sum() > 0:
                top_down_prompts = top_down_prompts.repeat_interleave(
                    num_images_each_instance, dim=0
                )  # For sample with multiple images, copy the prompt so that each image has a copy of the prompt
                assert (
                    len(media["image"]) == top_down_prompts.shape[0]
                ), f"{len(media['image'])} != {top_down_prompts.shape[0]}, num_images_each_instance: {num_images_each_instance}, instance_id_with_image: {instance_id_with_image}"
        media_embeds, top_down_selection_maps, top_down_selection_probs = self.__embed_media_tokens(
            media,
            media_config,
            num_images_each_instance,
            top_down_prompts,
            True,
            smooth_selection_prob,
            num_look_close,
            gt_selection_maps,
            original_image_sizes,
        )

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
                raise ValueError(f"Not all {name} embeddings are consumed!")

        # Truncate sequences to `model_max_length` as media embeddings are inserted
        inputs, labels = self._LlavaMetaForCausalLM__truncate_sequence(inputs, labels)

        # Pad sequences to the longest one in the batch
        return (
            *self._LlavaMetaForCausalLM__batchify_sequence(inputs, labels),
            top_down_selection_maps,
            top_down_selection_probs,
        )

    def __embed_media_tokens(
        self,
        media: Dict[str, List[torch.Tensor]],
        media_config: Dict[str, Dict[str, Any]],
        image_num_each_sample=None,
        top_down_prompts=None,
        concat_low_high_res_features=False,
        smooth_selection_prob=False,
        num_look_close=None,
        gt_selection_maps=None,
        original_image_sizes=None,
    ) -> Dict[str, List[torch.Tensor]]:
        embeds = defaultdict(deque)
        for name in media:
            if self.training:
                # Gather metainfo of media objects from all ranks
                info = [{"shape": tensor.shape, "dtype": tensor.dtype} for tensor in media.get(name, [])]
                infos = list(chain(*distributed.all_gather(info)))

                # The entire batch does not contain any media objects of this type.
                if not infos:
                    if name == "image":
                        top_down_selection_maps = torch.zeros(10, 10, 10, device=self.device)
                        top_down_selection_probs = torch.zeros(10, 10, 10, device=self.device)
                    continue

                # Create a dummy tensor to ensure the encoder is called, otherwise the training will hang.
                if not media.get(name):
                    dummy = torch.zeros(infos[0]["shape"], dtype=infos[0]["dtype"], device=self.device)
                    dummy_embeds, top_down_selection_maps, top_down_selection_probs = self.encoders[name](
                        [dummy],
                        media_config[name],
                        ps3=True,
                        image_num_each_sample=image_num_each_sample,
                        top_down_prompts=top_down_prompts[:1] if top_down_prompts is not None else None,
                        concat_low_high_res_features=concat_low_high_res_features,
                        smooth_selection_prob=smooth_selection_prob,
                        num_look_close=num_look_close,
                        gt_selection_maps=gt_selection_maps,
                        original_image_sizes=[(dummy.shape[-2], dummy.shape[-1])],
                    )
                    embeds["dummy"].extend(dummy_embeds)
                    continue
            if name == "image":
                image_embeds, top_down_selection_maps, top_down_selection_probs = self.encoders[name](
                    media[name],
                    media_config[name],
                    ps3=True,
                    image_num_each_sample=image_num_each_sample,
                    top_down_prompts=top_down_prompts,
                    concat_low_high_res_features=concat_low_high_res_features,
                    smooth_selection_prob=smooth_selection_prob,
                    num_look_close=num_look_close,
                    gt_selection_maps=gt_selection_maps,
                    original_image_sizes=original_image_sizes,
                )
                embeds[name] = deque(image_embeds)
            else:
                embeds[name] = deque(self.encoders[name](media[name], media_config[name]))
        return embeds, top_down_selection_maps, top_down_selection_probs

    def get_user_prompt_end_text(self):
        assert self.tokenizer.chat_template is not None, "chat_template must be provided in the tokenizer."

        dummy_prompt = "Here is the prompt."
        dummy_conv = [{"role": "user", "content": dummy_prompt}]
        text = self.tokenizer.apply_chat_template(dummy_conv, add_generation_prompt=True, tokenize=False)
        user_prompt_end_text = text.split(dummy_prompt)[-1].strip()
        return user_prompt_end_text

    def get_user_prompt_end_id_pos(self, input_ids):
        user_prompt_ending_text = self.get_user_prompt_end_text()

        ending_ids = self.tokenizer(user_prompt_ending_text, return_tensors="pt").input_ids.to(input_ids.device)
        ending_ids_len = ending_ids.shape[1]
        end_pos = torch.ones(input_ids.shape[0], device=input_ids.device, dtype=torch.long) * input_ids.shape[1] - 1
        for i in range(input_ids.shape[1] - ending_ids_len + 1):
            mask = torch.all(input_ids[:, i : i + ending_ids_len] == ending_ids, dim=1)
            if mask.sum() > 0:
                end_pos[mask] = torch.minimum(end_pos[mask], torch.ones_like(end_pos[mask]) * i - 1)
        return end_pos

    @torch.inference_mode()
    def generate_content(
        self,
        prompt: Union[str, List],
        generation_config: Optional[GenerationConfig] = None,
        response_format: Optional[ResponseFormat] = None,
        # PS3 configs
        return_selection_probs: bool = False,
        smooth_selection_prob=False,
        gt_selection_map=None,
        original_image_sizes=None,
    ) -> str:
        # TODO(zhijianl): Support directly taking conversation as input
        conversation = [{"from": "human", "value": prompt}]

        # Convert response format to logits processor
        if response_format:
            xgr_logits_processor = self.get_xgr_logits_processor(response_format)
        else:
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
                media[name] = [
                    process_images(images, self.vision_tower.image_processor, self.config).half()
                    for images in media[name]
                ]
            else:
                raise ValueError(f"Unsupported media type: {name}")

        # Tokenize the conversation
        input_ids = tokenize_conversation(conversation, self.tokenizer, add_generation_prompt=True).cuda().unsqueeze(0)

        # Set up the generation config
        generation_config = generation_config or self.default_generation_config

        # Generate the response
        try:
            outputs = self.generate(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                return_selection_probs=return_selection_probs,
                smooth_selection_prob=smooth_selection_prob,
                gt_selection_map=gt_selection_map,
                original_image_sizes=original_image_sizes,
                generation_config=generation_config,
                logits_processor=xgr_logits_processor,  # structured generation
            )
        except ValueError:
            if not generation_config.do_sample:
                raise
            # FIXME(zhijianl): This is a temporary workaround for the sampling issue
            logging.warning("Generation failed with sampling, retrying with greedy decoding.")
            generation_config.do_sample = False
            outputs = self.generate(
                input_ids=input_ids,
                media=media,
                media_config=media_config,
                return_selection_probs=return_selection_probs,
                smooth_selection_prob=smooth_selection_prob,
                gt_selection_map=gt_selection_map,
                original_image_sizes=original_image_sizes,
                generation_config=generation_config,
                logits_processor=xgr_logits_processor,
            )

        if return_selection_probs:
            output_ids, top_down_selection_maps, top_down_selection_probs = outputs
        else:
            output_ids = outputs

        # Decode the response
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        if return_selection_probs:
            return response, top_down_selection_maps, top_down_selection_probs
        else:
            return response

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Optional[torch.FloatTensor] = None,
        media: Optional[Dict[str, List[torch.Tensor]]] = None,
        media_config: Dict[str, Dict[str, Any]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        # PS3 configs
        return_selection_probs: bool = False,
        smooth_selection_prob=False,
        gt_selection_map=None,
        original_image_sizes=None,
        **generation_kwargs,
    ):
        if media is not None:
            # Get look close positions
            if self.look_close_mode == "after_prompt":
                look_close_positions = self.get_user_prompt_end_id_pos(input_ids)
            elif self.look_close_mode == "after_image":
                # look_close_positions = [(ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0][-1] for ids in input_ids]
                # look_close_positions = torch.LongTensor(look_close_positions).to(input_ids.device)
                look_close_positions = None
            else:
                raise NotImplementedError

            # Get top down prompts
            if self.look_close_mode == "after_image":
                top_down_prompts = None
            elif self.look_close_mode == "after_prompt":
                top_down_prompts = self.forward(
                    input_ids,
                    media,
                    None,
                    media_config,
                    attention_mask,
                    look_close_positions=look_close_positions,
                    get_top_down_prompts_only=True,
                    original_image_sizes=original_image_sizes,
                )
            else:
                raise NotImplementedError

            instance_id_with_image = torch.LongTensor([0]).to(input_ids.device)

            # For now we still look close after image even when look_close_mode is after_prompt
            if self.look_close_mode == "after_prompt":
                # look_close_positions = [(ids == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0][-1] for ids in input_ids]
                # look_close_positions = torch.LongTensor(look_close_positions).to(input_ids.device)
                look_close_positions = None

            if getattr(self, "smooth_selection_prob", None) is not None:
                smooth_selection_prob = self.smooth_selection_prob

            inputs_embeds, _, attention_mask, top_down_selection_maps, top_down_selection_probs = self._embed(
                input_ids,
                media,
                media_config,
                None,
                attention_mask,
                top_down_prompts,
                instance_id_with_image,
                smooth_selection_prob=smooth_selection_prob,
                gt_selection_maps=gt_selection_map,
                original_image_sizes=original_image_sizes,
            )
        else:
            inputs_embeds, _, attention_mask = self._embed(input_ids, media, media_config, None, attention_mask)

        outputs = self.llm.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **generation_kwargs)

        if return_selection_probs:
            return outputs, top_down_selection_maps, top_down_selection_probs
        else:
            return outputs
