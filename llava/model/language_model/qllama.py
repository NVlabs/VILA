# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
""" PyTorch LLaMA model."""
import math
import os
import time
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.modeling_flash_attention_utils import _flash_attention_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaFlashAttention2,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaLinearScalingRotaryEmbedding,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaSdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from ..qlinear_te import QLinearTE

try:
    import transformer_engine.pytorch as te
except:
    pass
from ..qfunction import *

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "QLlamaConfig"


class QLlamaConfig(LlamaConfig):
    model_type = "qllama"


class QLlamaMLP(LlamaMLP):
    def __init__(self, config, layer_idx):
        super().__init__(config)
        self.layer_idx = layer_idx

        # self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.gate_proj = QLinearTE(
            self.hidden_size, self.intermediate_size, bias=False, args=config, layer_idx=layer_idx
        )
        self.up_proj = QLinearTE(self.hidden_size, self.intermediate_size, bias=False, args=config, layer_idx=layer_idx)
        self.down_proj = QLinearTE(
            self.intermediate_size, self.hidden_size, bias=False, args=config, layer_idx=layer_idx
        )


class QLlamaAttention(LlamaAttention):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: QLlamaConfig, layer_idx):
        super().__init__(config)
        self.layer_idx = layer_idx

        # self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        # self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        # self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

        self.q_proj = QLinearTE(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
            args=config,
            layer_idx=layer_idx,
        )
        self.k_proj = QLinearTE(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            args=config,
            layer_idx=layer_idx,
        )
        self.v_proj = QLinearTE(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
            args=config,
            layer_idx=layer_idx,
        )
        self.o_proj = QLinearTE(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            args=config,
            layer_idx=layer_idx,
        )


class QLlamaFlashAttention2(QLlamaAttention):
    """
    Llama flash attention module. This module inherits from `LlamaAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    forward = LlamaFlashAttention2.forward


class QLlamaSdpaAttention(QLlamaAttention):
    """
    Llama attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LlamaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    forward = LlamaSdpaAttention.forward


QLLAMA_ATTENTION_CLASSES = {
    "eager": QLlamaAttention,
    "flash_attention_2": QLlamaFlashAttention2,
    "sdpa": QLlamaSdpaAttention,
}


class QLlamaDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: QLlamaConfig, layer_idx):
        super().__init__(config, layer_idx=layer_idx)
        self.hidden_size = config.hidden_size

        self.self_attn = QLLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = QLlamaMLP(config, layer_idx)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_idx = layer_idx

    forward = LlamaDecoderLayer.forward


class QLlamaPreTrainedModel(LlamaPreTrainedModel):
    config_class = QLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["QLlamaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear) or isinstance(module, QLinearTE):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class QLlamaModel(QLlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: QLlamaConfig
    """

    def __init__(self, config: QLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [QLlamaDecoderLayer(config, layer_idx=layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    _update_causal_mask = LlamaModel._update_causal_mask
    forward = LlamaModel.forward


class QLlamaForCausalLM(QLlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = QLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.forward_step_id = 0

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    forward = LlamaForCausalLM.forward
    prepare_inputs_for_generation = LlamaForCausalLM.prepare_inputs_for_generation


class QLlamaForSequenceClassification(QLlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = QLlamaModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    forward = LlamaForSequenceClassification.forward


AutoConfig.register("qllama", QLlamaConfig)
AutoModel.register(QLlamaConfig, QLlamaModel)
AutoModelForCausalLM.register(QLlamaConfig, QLlamaForCausalLM)


