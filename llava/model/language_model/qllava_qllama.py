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


import inspect
import os
import os.path as osp
import warnings
from typing import List, Optional, Tuple, Union

import torch
from transformers import (
    AutoConfig,
    AutoModel,
    GenerationConfig,
    LlamaConfig,
    LlamaForCausalLM,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import ContextManagers, no_init_weights

from ..configuration_llava import LlavaConfig
from ..llava_arch import LlavaMetaForCausalLM, LlavaMetaModel
from ..multimodal_encoder.builder import build_vision_tower
from ..multimodal_projector.builder import build_mm_projector
from ..utils import get_model_config, get_model_config_fp8
from .builder import build_llm_and_tokenizer
from .llava_llama import LlavaLlamaConfig, LlavaLlamaModel

quantize_args_to_model_class = {
    "fp8Linear_llama": "QLlamaForCausalLM",
    "fp8LinearAndActivation_llama": "QMemLlamaForCausalLM",
    "fp8Linear_qwen2": "FP8LinearQwen2ForCausalLM",
    "fp8Activation_qwen2": "FP8ActivationQwen2ForCausalLM",
    "fp8ActivationResidual_qwen2": "FP8ActivationResidualQwen2ForCausalLM",
}


class QLlavaLlamaConfig(LlavaLlamaConfig):
    model_type = "qllava_qllama"


## FIXME we will follow the convention to add a new class for CausalLM in the future
class QLlavaLlamaModel(LlavaLlamaModel):
    config_class = QLlavaLlamaConfig
    main_input_name = "input_embeds"
    supports_gradient_checkpointing = True

    def __init__(self, config: QLlavaLlamaConfig = None, model_args=None, *args, **kwargs) -> None:
        PreTrainedModel.__init__(self, config)
        return self.init_vlm(config=config, model_args=model_args, *args, **kwargs)

    # rewrite to support QLlama
    def init_vlm(self, config: PreTrainedModel = None, model_args=None, *args, **kwargs):
        # TODO(ligeng): figure out how from_config and from_pretrained works in HF implementation.
        if hasattr(self, "llm") or hasattr(self, "vision_tower") or hasattr(self, "mm_projector"):
            # already initialized, skipped
            return

        model_dtype = getattr(config, "model_dtype", "torch.float16")
        if not hasattr(config, "model_dtype"):
            warnings.warn("model_dtype not found in config, defaulting to torch.float16.")
            config.model_dtype = model_dtype

        if model_args.quantize_model in ["fp8Activation_qwen2", "fp8ActivationResidual_qwen2"]:
            cfgs = get_model_config_fp8(config)  # The first cfg is fp8
        else:
            cfgs = get_model_config(config)
        if len(cfgs) == 3:
            llm_cfg, vision_tower_cfg, mm_projector_cfg = cfgs
        elif len(cfgs) == 4:
            llm_cfg, vision_tower_cfg, mm_projector_cfg, fp8_llm_cfg = cfgs
            kwargs.update({"fp8_llm_cfg": fp8_llm_cfg})
        else:
            raise ValueError("`llm_cfg` `mm_projector_cfg` `vision_tower_cfg` not found in the config.")

        kwargs.update(
            {
                "quantize_model_class": quantize_args_to_model_class[model_args.quantize_model],
                "model_args": model_args,
            }
        )
        self.llm, self.tokenizer = build_llm_and_tokenizer(llm_cfg, config, *args, **kwargs)
        self.vision_tower = build_vision_tower(vision_tower_cfg, config)
        self.mm_projector = build_mm_projector(mm_projector_cfg, config)

        for name, module in self.llm.named_modules():
            module.layer_name = name

        self.pad_to_multiple_of = model_args.pad_to_multiple_of

        self.post_config()
        self.is_loaded = True

        assert (
            self.llm is not None or self.vision_tower is not None or self.mm_projector is not None
        ), "At least one of the components must be instantiated."


AutoConfig.register("qllava_qllama", QLlavaLlamaConfig)
AutoModel.register(QLlavaLlamaConfig, QLlavaLlamaModel)


