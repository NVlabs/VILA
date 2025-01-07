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

import re

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels))

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


class DownSampleBlock(nn.Module):
    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.flat_square(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds

    def flat_square(self, x):
        n, w, h, c = x.size()
        if w % 2 == 1:
            x = torch.concat([x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
            n, w, h, c = x.size()
        if h % 2 == 1:
            x = torch.concat([x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
            n, w, h, c = x.size()
        x = x.contiguous()
        x = x.view(n, w, int(h / 2), int(c * 2))
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
        x = x.permute(0, 2, 1, 3).contiguous()
        return x


class DownSample2x2BlockFix(nn.Module):
    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = flat_square_2x2(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds


def flat_square_2x2(x):
    n, w, h, c = x.size()
    if w % 2 == 1:
        x = torch.concat([x, torch.zeros((n, 1, h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
        n, w, h, c = x.size()
    x = x.contiguous()
    if h % 2 == 1:
        x = torch.concat([x, torch.zeros((n, w, 1, c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
        n, w, h, c = x.size()
    x = x.view(n, w, int(h / 2), int(c * 2))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h / 2), int(w / 2), int(c * 4))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


class DownSample3x3BlockFix(nn.Module):
    def forward(self, x):
        vit_embeds = x
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = flat_square_3x3(vit_embeds)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds


def flat_square_3x3(x):
    n, w, h, c = x.size()
    if w % 3 != 0:
        x = torch.concat([x, torch.zeros((n, 3 - (w % 3), h, c), dtype=x.dtype).to(x.device)], dim=1).contiguous()
        n, w, h, c = x.size()
    x = x.contiguous()
    if h % 3 != 0:
        x = torch.concat([x, torch.zeros((n, w, 3 - (h % 3), c), dtype=x.dtype).to(x.device)], dim=2).contiguous()
        n, w, h, c = x.size()
    x = x.view(n, w, int(h / 3), int(c * 3))
    x = x.permute(0, 2, 1, 3).contiguous()
    x = x.view(n, int(h / 3), int(w / 3), int(c * 9))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x


class MultimodalProjectorConfig(PretrainedConfig):
    model_type = "v2l_projector"

    def __init__(self, mm_projector_type: str = None, **kwargs):
        super().__init__()
        self.mm_projector_type = mm_projector_type


class MultimodalProjector(PreTrainedModel):
    config_class = MultimodalProjectorConfig

    def __init__(self, mm_projector_cfg: MultimodalProjectorConfig, config: PretrainedConfig):
        super().__init__(mm_projector_cfg)
        mm_projector_type = mm_projector_cfg.mm_projector_type
        self.downsample_rate = 1
        if mm_projector_type == "identity":
            self.layers = IdentityMap()
        elif mm_projector_type == "linear":
            self.layers = nn.Linear(config.mm_hidden_size, config.hidden_size)
        elif mm_projector_type == "mlp_downsample":
            self.layers = nn.Sequential(
                DownSampleBlock(),
                nn.LayerNorm(config.mm_hidden_size * 4),
                nn.Linear(config.mm_hidden_size * 4, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            self.downsample_rate = 2
        elif mm_projector_type == "mlp_downsample_2x2_fix":
            self.layers = nn.Sequential(
                DownSample2x2BlockFix(),
                nn.LayerNorm(config.mm_hidden_size * 4),
                nn.Linear(config.mm_hidden_size * 4, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            self.downsample_rate = 2
        elif mm_projector_type == "mlp_downsample_3x3_fix":
            self.layers = nn.Sequential(
                DownSample3x3BlockFix(),
                nn.LayerNorm(config.mm_hidden_size * 9),
                nn.Linear(config.mm_hidden_size * 9, config.mm_hidden_size * 3),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size * 3),
                nn.Linear(config.mm_hidden_size * 3, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            self.downsample_rate = 3
        elif mm_projector_type == "mlp_downsample_3x3_s2":
            self.layers = nn.Sequential(
                DownSample3x3BlockFix(),
                nn.LayerNorm(config.mm_hidden_size * 9),
                nn.Linear(config.mm_hidden_size * 9, config.mm_hidden_size * 3),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size * 3),
                nn.Linear(config.mm_hidden_size * 3, config.mm_hidden_size),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size),
                nn.Linear(config.mm_hidden_size, config.mm_hidden_size // 3),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size // 3),
                nn.Linear(config.mm_hidden_size // 3, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        elif mm_projector_type == "mlp_downsample_3x3_s2_new":
            self.layers = nn.Sequential(
                DownSample3x3BlockFix(),
                nn.LayerNorm(config.mm_hidden_size * 9),
                nn.Linear(config.mm_hidden_size * 9, config.mm_hidden_size * 4),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size * 4),
                nn.Linear(config.mm_hidden_size * 4, config.mm_hidden_size * 2),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size * 2),
                nn.Linear(config.mm_hidden_size * 2, config.mm_hidden_size),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size),
                nn.Linear(config.mm_hidden_size, config.mm_hidden_size // 3),
                nn.GELU(),
                nn.LayerNorm(config.mm_hidden_size // 3),
                nn.Linear(config.mm_hidden_size // 3, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        else:
            mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", mm_projector_type)
            if mlp_gelu_match:
                mlp_depth = int(mlp_gelu_match.group(1))
                modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
                for _ in range(1, mlp_depth):
                    modules.append(nn.GELU())
                    modules.append(nn.Linear(config.hidden_size, config.hidden_size))
                self.layers = nn.Sequential(*modules)
            else:
                raise ValueError(f"Unknown projector type: {mm_projector_type}")

    def forward(self, x, *args, **kwargs):
        return self.layers(x)


AutoConfig.register("v2l_projector", MultimodalProjectorConfig)
AutoModel.register(MultimodalProjectorConfig, MultimodalProjector)


