from importlib import import_module
from typing import Tuple

import torch
import transformers
from torch import nn
from torch.nn import functional as F

__all__ = ["patch"]


def _get_unpad_data(attention_mask: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if hasattr(_get_unpad_data, "seqlens_in_batch"):
        seqlens_in_batch = _get_unpad_data.seqlens_in_batch
    else:
        seqlens_in_batch = torch.sum(attention_mask, dim=1)

    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def set_seqlens_in_batch(seqlens_in_batch: torch.Tensor) -> None:
    _get_unpad_data.seqlens_in_batch = seqlens_in_batch


def patch(model: nn.Module) -> None:
    if transformers.__version__ < "4.43.0":
        m = import_module(model.__module__)
        if not hasattr(m, "_get_unpad_data"):
            raise ValueError(f"Module {m} does not have function '_get_unpad_data' for packing")
        m._get_unpad_data = _get_unpad_data
    else:
        transformers.modeling_flash_attention_utils._get_unpad_data = _get_unpad_data


