# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

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
