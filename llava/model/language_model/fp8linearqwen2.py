# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Qwen2 model."""

import math
from functools import lru_cache
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention,
    Qwen2DecoderLayer,
    Qwen2FlashAttention2,
    Qwen2ForCausalLM,
    Qwen2MLP,
    Qwen2Model,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    Qwen2SdpaAttention,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from ..liger.cross_entropy import LigerForCausalLMLoss
from ..qlinear_te import QLinearTE
from .configuration_quantize import QuantizationConfig

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)


class FP8LinearQwen2Config(Qwen2Config):
    model_type = "fp8linear_qwen2"

    def __init__(
        self,
        coat_fp8_args=None,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            use_sliding_window,
            sliding_window,
            max_window_layers,
            attention_dropout,
            **kwargs,
        )

        self.coat_fp8_args = coat_fp8_args


# Copied from transformers.models.mistral.modeling_mistral.MistralMLP with Mistral->Qwen2
class FP8LinearQwen2MLP(Qwen2MLP):
    def __init__(self, config, layer_idx):
        super().__init__(config)
        # self.gate_proj = te.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.up_proj = te.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.down_proj = te.Linear(self.intermediate_size, self.hidden_size, bias=False)

        self.gate_proj = QLinearTE(
            self.hidden_size, self.intermediate_size, bias=False, args=config.coat_fp8_args, layer_idx=layer_idx
        )
        self.up_proj = QLinearTE(
            self.hidden_size, self.intermediate_size, bias=False, args=config.coat_fp8_args, layer_idx=layer_idx
        )
        self.down_proj = QLinearTE(
            self.intermediate_size, self.hidden_size, bias=False, args=config.coat_fp8_args, layer_idx=layer_idx
        )

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class FP8LinearQwen2Attention(Qwen2Attention):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: FP8LinearQwen2Config, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)

        self.q_proj = QLinearTE(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=True,
            args=config.coat_fp8_args,
            layer_idx=layer_idx,
        )
        self.k_proj = QLinearTE(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
            args=config.coat_fp8_args,
            layer_idx=layer_idx,
        )
        self.v_proj = QLinearTE(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=True,
            args=config.coat_fp8_args,
            layer_idx=layer_idx,
        )
        self.o_proj = QLinearTE(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            args=config.coat_fp8_args,
            layer_idx=layer_idx,
        )

    forward = Qwen2Attention.forward


class FP8LinearQwen2FlashAttention2(FP8LinearQwen2Attention):
    """
    Qwen2 flash attention module, following Qwen2 attention module. This module inherits from `Qwen2Attention`
    as the weights of the module stays untouched. The only required change would be on the forward pass
    where it needs to correctly call the public API of flash attention and deal with padding tokens
    in case the input contains any of them. Additionally, for sliding window attention, we apply SWA only to the bottom
    config.max_window_layers layers.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    forward = Qwen2FlashAttention2.forward


# Copied from transformers.models.mixtral.modeling_mixtral.MixtralSdpaAttention with Mixtral->Qwen2
class FP8LinearQwen2SdpaAttention(FP8LinearQwen2Attention):
    """
    Qwen2 attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `Qwen2Attention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    forward = Qwen2SdpaAttention.forward


FP8LINEARQWEN2_ATTENTION_CLASSES = {
    "eager": FP8LinearQwen2Attention,
    "flash_attention_2": FP8LinearQwen2FlashAttention2,
    "sdpa": FP8LinearQwen2SdpaAttention,
}


class FP8LinearQwen2DecoderLayer(nn.Module):
    def __init__(self, config: FP8LinearQwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = FP8LINEARQWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.mlp = FP8LinearQwen2MLP(config, layer_idx)
        self.input_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    forward = Qwen2DecoderLayer.forward


class FP8LinearQwen2PreTrainedModel(Qwen2PreTrainedModel):
    config_class = FP8LinearQwen2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FP8LinearQwen2DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class FP8LinearQwen2Model(FP8LinearQwen2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: Qwen2Config
    """

    def __init__(self, config: FP8LinearQwen2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [FP8LinearQwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen2RotaryEmbedding(config=config)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    forward = Qwen2Model.forward

    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask
    _update_causal_mask = Qwen2Model._update_causal_mask


class FP8LinearQwen2ForCausalLM(FP8LinearQwen2PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = FP8LinearQwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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

    @property
    @lru_cache
    def loss_function(self):
        return LigerForCausalLMLoss

    forward = Qwen2ForCausalLM.forward

    # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
    prepare_inputs_for_generation = Qwen2ForCausalLM.prepare_inputs_for_generation


AutoConfig.register("fp8linear_qwen2", FP8LinearQwen2Config)
AutoModel.register(FP8LinearQwen2Config, FP8LinearQwen2Model)
AutoModelForCausalLM.register(FP8LinearQwen2Config, FP8LinearQwen2ForCausalLM)


