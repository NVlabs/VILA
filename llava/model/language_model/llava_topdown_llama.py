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

# This file is modified from https://github.com/haotian-liu/LLaVA/


import os
from collections import defaultdict
from dataclasses import dataclass
from math import ceil
from typing import Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.constants import MEDIA_TOKENS
from llava.model.loss import soft_cross_entropy
from llava.model.utils.packing import set_seqlens_in_batch
from llava.train.sequence_parallel.globals import get_pg_manager
from llava.utils.logging import logger

from ...train.utils import calculate_loss_weight
from ..configuration_llava import LlavaConfig
from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel, LlavaTopDownMetaForCausalLM


@dataclass
class CausalLMTopDownOutputWithPast(CausalLMOutputWithPast):
    top_down_selection_probs: Optional[torch.FloatTensor] = None


class LlavaTopDownLlamaConfig(LlavaConfig):
    model_type = "llava_topdown_llama"


# FIXME we will follow the convention to add a new class for CausalLM in the future
class LlavaTopDownLlamaModel(LlavaMetaModel, LlavaTopDownMetaForCausalLM, PreTrainedModel):
    config_class = LlavaTopDownLlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True
    _supports_flash_attn_2 = True

    def __init__(self, config: LlavaTopDownLlamaConfig = None, *args, **kwargs) -> None:
        super().__init__(config)
        self.init_vlm(config=config, *args, **kwargs)
        self.look_close_mode = getattr(config, "look_close_mode", None)
        self.num_look_close = getattr(config, "num_look_close", 1)
        self.num_token_look_close = getattr(config, "num_token_look_close", None)
        self.max_num_look_close = self.num_look_close
        self.use_high_res_pos_embed = getattr(config, "high_res_pos_embed", False)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs,
    ):
        if hasattr(cls, "load_pretrained"):
            return cls.load_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs,
            )
        return super(LlavaTopDownLlamaModel).from_pretrained(
            pretrained_model_name_or_path,
            *model_args,
            config=config,
            cache_dir=cache_dir,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            **kwargs,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        media: Optional[Dict[str, List[torch.Tensor]]] = None,
        images: Optional[torch.FloatTensor] = None,
        media_config: Optional[List] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        packing: bool = True,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        dpo_forward: bool = False,
        look_close_positions: Optional[torch.LongTensor] = None,
        top_down_prompts: Optional[torch.FloatTensor] = None,
        get_top_down_prompts_only: bool = False,
        smooth_selection_prob: bool = False,
        gt_selection_maps=None,
        original_image_sizes=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        self.freezed_module_patch()

        # Top-down is only applied to samples with images
        IMAGE_TOKEN_INDEX = self.tokenizer.media_token_ids["image"]
        num_images_each_instance = (input_ids == IMAGE_TOKEN_INDEX).sum(dim=-1)
        instance_id_with_image = (num_images_each_instance > 0).nonzero(as_tuple=True)[0]

        # Note that number of images is probably different from number of instances in input_ids because some input doesn't have images and some contains multiple images
        assert look_close_positions is None or instance_id_with_image.shape[0] == look_close_positions.shape[0]
        # assert top_down_prompts is None or instance_id_with_image.shape[0] == top_down_prompts.shape[0]

        if images is not None:
            if media is not None:
                raise ValueError("Both 'media' and 'images' are provided. Please provide only one.")
            logger.warning("The 'images' argument is deprecated. Please use 'media' instead.")
            media = {"image": images}

        if media_config is None:
            media_config = defaultdict(dict)

        if get_top_down_prompts_only:
            if num_images_each_instance.sum() == 0:
                instance_id_with_image = torch.zeros(1).to(torch.long).to(input_ids.device)
                look_close_positions = torch.ones(1).to(torch.long).to(input_ids.device)

            num_images_before_lookclose_each_image = [
                (input_ids[id, : look_close_positions[i] + 1] == IMAGE_TOKEN_INDEX).sum()
                for i, id in enumerate(instance_id_with_image)
            ]
            num_images_before_lookclose_each_image = torch.stack(num_images_before_lookclose_each_image)

            with torch.no_grad():
                if inputs_embeds is None:
                    (
                        inputs_embeds,
                        labels,
                        attention_mask,
                        top_down_selection_maps,
                        top_down_selection_probs,
                    ) = self._embed(
                        input_ids,
                        media,
                        media_config,
                        labels,
                        attention_mask,
                        top_down_prompts,
                        instance_id_with_image,
                        num_look_close=0,
                    )

                if self.mm_projector.config.mm_projector_type in ["mlp_downsample", "mlp_downsample_2x2_fix"]:
                    low_res_num_feature = (
                        ceil(int(self.vision_tower.vision_tower.vision_model.low_res_token_num**0.5) / 2) ** 2
                    )
                elif self.mm_projector.config.mm_projector_type == "mlp_downsample_3x3_fix":
                    low_res_num_feature = (
                        ceil(int(self.vision_tower.vision_tower.vision_model.low_res_token_num**0.5) / 3) ** 2
                    )
                else:
                    low_res_num_feature = self.vision_tower.vision_tower.vision_model.low_res_token_num

                # look_close_positions = look_close_positions + (low_res_num_feature - 1) * num_images_before_lookclose_each_image
                look_close_positions = (
                    look_close_positions + (low_res_num_feature - 1 + 1) * num_images_before_lookclose_each_image
                )  # +1 is because we appended '/n' after each image in the model/encoders/image/basic.py
                max_length = look_close_positions.max() + 1
                prompt_input_embeds = inputs_embeds[:, :max_length]
                prompt_position_ids = position_ids[:, :max_length] if position_ids is not None else None
                prompt_labels = labels[:, :max_length] if labels is not None else None

                outputs = self.llm(
                    inputs_embeds=prompt_input_embeds,
                    position_ids=prompt_position_ids,
                    past_key_values=past_key_values,
                    labels=prompt_labels,
                    output_hidden_states=True,
                    **kwargs,
                )

            num_instances_with_image = instance_id_with_image.shape[0]
            last_hidden_states = outputs.hidden_states[-1]
            top_down_hidden_states = last_hidden_states[instance_id_with_image][
                torch.arange(num_instances_with_image), look_close_positions.clamp(0, last_hidden_states.shape[1] - 1)
            ]
            top_down_prompts = self.get_mm_projector()(
                top_down_hidden_states, forward_top_down_prompt_head=True
            )  # .to(device)

            return top_down_prompts
        else:
            if inputs_embeds is None:
                (
                    inputs_embeds,
                    labels,
                    attention_mask,
                    top_down_selection_maps,
                    top_down_selection_probs,
                ) = self._embed(
                    input_ids,
                    media,
                    media_config,
                    labels,
                    attention_mask,
                    top_down_prompts,
                    instance_id_with_image,
                    smooth_selection_prob=smooth_selection_prob,
                    gt_selection_maps=gt_selection_maps,
                    original_image_sizes=original_image_sizes,
                )

            if packing and self.training and not dpo_forward:
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

        # update outputs
        if isinstance(outputs, tuple):
            outputs = outputs + (
                None,
                top_down_selection_probs,
            )
        else:
            outputs = CausalLMTopDownOutputWithPast(**outputs, top_down_selection_probs=top_down_selection_probs)

        if self.training and getattr(self.config, "time_token_ids", []):
            outputs.loss = soft_cross_entropy(
                outputs.logits,
                labels,
                soft_tokens=self.config.time_token_ids,
                std=self.config.soft_ce_std,
            )

        # Loss rescale for SP
        if get_pg_manager() is not None:
            loss_weight = calculate_loss_weight(labels)
            outputs.loss = outputs.loss * loss_weight

        if dpo_forward:
            return outputs.logits, labels

        return outputs


AutoConfig.register("llava_topdown_llama", LlavaTopDownLlamaConfig)
AutoModel.register(LlavaTopDownLlamaConfig, LlavaTopDownLlamaModel)
