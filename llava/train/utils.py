import os
import re
import torch
import pathlib
from dataclasses import dataclass
from transformers import PretrainedConfig, PreTrainedModel
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from accelerate.hooks import add_hook_to_module


def rprint(*args, **kwargs):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        return print(f"[dist-{rank}-of-{world_size}]", *args, **kwargs)
    else:
        return print(*args, **kwargs)


def mprint(*args, **kwargs):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        if rank == 0:
            return print(f"[dist-{rank}-of-{world_size}]", *args, **kwargs)
        else:
            return
    else:
        return print(*args, **kwargs)


def is_local(model_name_or_path: str) -> bool:
    return os.path.isdir(model_name_or_path)


def get_checkpoint_path(
    output_dir: str, checkpoint_prefix: str = "checkpoint"
) -> str | None:
    output_dir = os.path.abspath(output_dir)
    pathlib_dir = pathlib.Path(output_dir)

    if list(pathlib_dir.glob("config.json")):
        # training has been finished
        return output_dir, False
    else:
        try:
            ordering_and_checkpoint_path = []
            glob_checkpoints = [
                str(x)
                for x in pathlib.Path(output_dir).glob(f"{checkpoint_prefix}-*")
                if os.path.isdir(x)
            ]
            for path in glob_checkpoints:
                regex_match = re.match(f".*{checkpoint_prefix}-([0-9]+)", path)
                if regex_match is not None and regex_match.groups() is not None:
                    ordering_and_checkpoint_path.append(
                        (int(regex_match.groups()[0]), path)
                    )
            checkpoints_sorted = sorted(ordering_and_checkpoint_path)
            return checkpoints_sorted[-1][1], True
        except:
            return None, True


def prepare_config_for_training(
    config: PretrainedConfig, model_args: dataclass, training_args: dataclass, data_args: dataclass
) -> None:
    assert model_args.vision_tower is not None, "requires vision tower"
    ## set module configurations
    if getattr(config, "llm_cfg", None) is None:
        config.llm_cfg = model_args.model_name_or_path
    if getattr(config, "vision_tower_cfg", None) is None:
        config.vision_tower_cfg = model_args.vision_tower
    if getattr(config, "mm_projector_cfg", None) is None:
        config.mm_projector_cfg = model_args.mm_projector
    ## set default dtype
    config.model_dtype = torch.bfloat16 if training_args.bf16 else torch.float16
    config.model_dtype = config.model_dtype.__str__()
    ## set tuning modules
    config.tune_language_model = training_args.tune_language_model
    config.tune_vision_tower = training_args.tune_vision_tower
    config.tune_mm_projector = training_args.tune_mm_projector
    ## set data args
    config.image_aspect_ratio = data_args.image_aspect_ratio
    ## extra vision tower configuration
    if getattr(config, "vision_tower_cfg", None) is not None:
        config.mm_vision_select_layer = model_args.mm_vision_select_layer
        config.mm_vision_select_feature = model_args.mm_vision_select_feature
        ## vision tower configurations
        config.vision_resolution = model_args.vision_resolution
        config.interpolate_mode = model_args.interpolate_mode
        config.drop_path_rate = model_args.drop_path_rate
        config.s2 = model_args.s2
        config.s2_scales = model_args.s2_scales
        config.s2_max_split_size = model_args.s2_max_split_size


def vision_resolution_elevation(model: PreTrainedModel, config: PretrainedConfig):
    vision_tower = model.get_vision_tower()
    if (
        vision_tower is not None
        and "radio" not in vision_tower.__class__.__name__.lower()
    ):
        vision_tower._maybe_resize_pos_embeds(
            model=vision_tower.vision_tower,
            image_processor=vision_tower.image_processor,
            resolution=getattr(config, "vision_resolution", -1),
            interpolate_mode=getattr(config, "interpolate_mode", "linear"),
        )


def unit_test_rope_scaling(
    model: PreTrainedModel, config: PretrainedConfig, training_args: dataclass
):
    return False
