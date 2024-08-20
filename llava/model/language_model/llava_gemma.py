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


PAD_TOKEN_ID = 0

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gemma import GemmaConfig, GemmaForCausalLM, GemmaModel

from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel


class LlavaGemmaConfig(GemmaConfig):
    model_type = "llava_gemma"


class LlavaGemmaModel(GemmaModel, LlavaMetaModel):
    config_class = LlavaGemmaConfig

    def __init__(self, config: GemmaConfig):
        super().__init__(config)


class LlavaGemmaForCausalLM(GemmaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaGemmaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = LlavaGemmaModel(config)
        self.pretraining_tp = 1
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def get_lm_head(self):
        return self.lm_head

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        seqlens_in_batch: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids, position_ids, attention_mask, past_key_values, labels, images
            )
        # TODO (kentang-mit@): fuse this function into the previous one.
        # current design makes unit-test easier.
        if self.training:
            (
                _,
                new_position_ids,
                new_attention_mask,
                _,
                new_inputs_embeds,
                new_labels,
                sorted_seqlens_in_batch,
            ) = self.repack_multimodal_data(
                input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels
            )
            if sorted_seqlens_in_batch is None:
                sorted_seqlens_in_batch = seqlens_in_batch
            new_input_ids = None
            past_key_values = None
            new_cache_position = None
        else:
            new_attention_mask = attention_mask
            new_position_ids = position_ids
            new_inputs_embeds = inputs_embeds
            new_labels = labels
            if attention_mask is not None:
                sorted_seqlens_in_batch = attention_mask.sum(-1).int()
            else:
                sorted_seqlens_in_batch = None
            new_input_ids = input_ids
            # kentang-mit@: This only works for batch=1 currently
            # model.generate of gemma does not correctly handle decoding stage currently
            # need to manually adjust decoding stage input = 1 token
            if past_key_values is not None:
                if new_inputs_embeds is not None:
                    new_inputs_embeds = new_inputs_embeds[:, [-1]]
                # kentang-mit@: seems to be a problem unique to gemma
                if new_position_ids is not None:
                    new_position_ids = new_position_ids[:, [-1]]
            new_cache_position = new_position_ids[0]

        outputs = super().forward(
            input_ids=new_input_ids,
            attention_mask=new_attention_mask,
            position_ids=new_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=new_inputs_embeds,
            labels=new_labels,
            use_cache=use_cache,
            cache_position=new_cache_position,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            seqlens_in_batch=sorted_seqlens_in_batch,
        )
        return outputs

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs["images"] = images
        return _inputs


AutoConfig.register("llava_gemma", LlavaGemmaConfig)
AutoModelForCausalLM.register(LlavaGemmaConfig, LlavaGemmaForCausalLM)
