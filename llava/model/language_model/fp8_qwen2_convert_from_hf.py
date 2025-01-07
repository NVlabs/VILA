import argparse
import os
from dataclasses import asdict, dataclass, field
from typing import Optional

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM

from ..coat.activation.models._fp8_quantization_config import QuantizationConfig
from .fp8activationqwen2 import FP8ActivationQwen2Config, make_state_dict_compatible


@dataclass
class ConvertArguments:
    model_name: str = field(metadata={"help": "The model name or path to download the LLaMA model"})
    save_path: str = field(metadata={"help": "The path where the converted model weights will be saved"})
    cache_dir: str = field(default=None, metadata={"help": "Directory to cache the model"})


def download_and_convert_qwen2(convert_args: ConvertArguments, quantization_args: QuantizationConfig):
    """
    Downloads a LLaMA model, converts its weights using `make_state_dict_compatible`,
    and saves the converted model.

    Args:
        model_name (str): The model name or path to download the LLaMA model.
        save_path (str): The path where the converted model weights will be saved.
        cache_dir (Optional[str]): Directory to cache the model. Defaults to None.

    Returns:
        None
    """
    model_name = convert_args.model_name
    save_path = convert_args.save_path
    cache_dir = convert_args.cache_dir

    # Step 1: Download the original LLaMA model
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

    # Step 2: Initialize the model configuration for FP8 or other custom config
    config = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)

    # Step 3: Apply make_state_dict_compatible to convert weights
    compatible_state_dict = make_state_dict_compatible(model.state_dict())

    # Step 4: Create a new model instance with compatible configuration
    fp8_config = FP8ActivationQwen2Config(**config.to_dict())
    fp8_config.coat_fp8_args = asdict(quantization_args)
    fp8_config._name_or_path = save_path

    converted_model = AutoModelForCausalLM.from_config(fp8_config, torch_dtype=torch.bfloat16)
    converted_model.load_state_dict(compatible_state_dict)

    # Step 5: Save the converted model and configuration using save_pretrained
    os.makedirs(save_path, exist_ok=True)
    converted_model.save_pretrained(save_path)
    print(f"Converted model saved at {save_path}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = transformers.HfArgumentParser((ConvertArguments, QuantizationConfig))  # NOTE: FP8
    convert_args, quantization_args = parser.parse_args_into_dataclasses()

    # Call the function with parsed arguments
    download_and_convert_qwen2(convert_args, quantization_args)


