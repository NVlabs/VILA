import os
from typing import Optional

from transformers import PreTrainedModel

from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model

__all__ = ["load"]


def load(model_path: str, model_base: Optional[str] = None) -> PreTrainedModel:
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    if os.path.exists(os.path.join(model_path, "model")):
        model_path = os.path.join(model_path, "model")
    _, model, _, _ = load_pretrained_model(model_path, model_name, model_base)
    return model
