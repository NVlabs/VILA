from .language_model.llava_llama import LlavaLlamaConfig, LlavaLlamaModel

# FP8 related comments, development in progress (PI: ligeng zhu, haochen xi)
# NOTE: VLM + LLM
# from .language_model.qllava_qllama import QLlavaLlamaConfig, QLlavaLlamaModel
# NOTE: Linear -> fp8, similar to transformer engine
# from .language_model.qllama import QLlamaConfig, QLlamaForCausalLM, QLlamaModel
# NOTE: Linear + Activation -> fp8, haochen's iclr version
# from .language_model.qmemllama import QMemLlamaConfig, QMemLlamaForCausalLM, QMemLlamaModel
"""
TODO:
    linear(weights):
        simulated fp8: done
        real fp8: in-progress (code already implmented)
    activation:
        simulated fp8: done
        real fp8: in-progress (still coding)
    optimizers:
        current VILA: bf16
        simulated fp8: done
        real fp8 + fsdp (single node): done
        real fp8 + fsdp (multiple node): in-progress
1. linear fp8
2. activation fp8
3. fp8 infernce example (load directly from a fp8 and fwd)
4. bind fp8 related configs to QLlamaConfig {"coat_fp8_args": {}}
"""
from .language_model.fp8linearqwen2 import FP8LinearQwen2Config, FP8LinearQwen2Model
from .language_model.qllava_qllama import QLlavaLlamaConfig, QLlavaLlamaModel


