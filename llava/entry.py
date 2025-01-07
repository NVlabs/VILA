import os
import typing
from typing import List, Optional

if typing.TYPE_CHECKING:
    from transformers import PreTrainedModel
else:
    PreTrainedModel = None

__all__ = ["load"]


def load(
    model_path: str,
    model_base: Optional[str] = None,
    devices: Optional[List[int]] = None,
    **kwargs,
) -> PreTrainedModel:
    import torch

    from llava.conversation import auto_set_conversation_mode
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    auto_set_conversation_mode(model_path)

    model_name = get_model_name_from_path(model_path)
    model_path = os.path.expanduser(model_path)
    if os.path.exists(os.path.join(model_path, "model")):
        model_path = os.path.join(model_path, "model")

    # Set `max_memory` to constrain which GPUs to use
    if devices is not None:
        assert "max_memory" not in kwargs, "`max_memory` should not be set when `devices` is set"
        kwargs.update(max_memory={device: torch.cuda.get_device_properties(device).total_memory for device in devices})

    model = load_pretrained_model(model_path, model_name, model_base, **kwargs)[1]
    return model


