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
import warnings
from argparse import Namespace
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, CLIPVisionConfig

from llava.model.multimodal_encoder.vision_encoder import VisionTower
from llava.train.utils import mprint, rprint

from .image_processor import ImageProcessor
from .visualize_features import get_pca_map


def get_prefix_state_dict(state_dict: Dict[str, Any], prefix: str):
    mod_state_dict = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    return mod_state_dict


def is_rank0():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


class RADIOVisionTower(VisionTower):
    """
    Vision Tower for the RADIO model.

    Args:
        vision_tower (str): Vision tower name. This is passed on
            the command line with the `--vision_tower` argument.
            The string is expected in the pattern of:
            `radio:<image_size>:<checkpoint>:<extra_config>`.
            Where <extra_config> is a comma-separated list of key=value pairs.
            <image_size> can also be a comma-separated list of resolutions in
            the case of multi-res inference. Limitations apply, e.g. only two
            resolutions are supported and the second resolution must be a divisor
            of the first one.
        args (Namespace): Arguments.
        delay_load (bool): Delay loading the model.
    """

    def __init__(self, vision_tower, args, delay_load=False):
        """Initialization Routine."""

        super().__init__(vision_tower, args, delay_load)

        mprint(f"RADIOVisionTower: {vision_tower}. Args: {args} Delay load: {delay_load}")

        assert not delay_load

        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        extra_config = {}

        # Check if vision_tower is a valid path.
        if os.path.exists(vision_tower):
            self.vision_tower_name = self.vision_tower_checkpoint = vision_tower
            vision_cfg = getattr(args, "vision_tower_cfg")
            self.image_size = vision_cfg["image_size"]
        else:
            self.vision_tower_name = vision_tower[len("radio:") :]
            config_items = self.vision_tower_name.split(":")
            self.image_size = int(config_items[0])

            self.vision_tower_checkpoint = config_items[1]

            if len(config_items) > 2:
                # Parse extra config items. These are provided as a comma-separated list
                # of key=value pairs.
                extra_config_items = config_items[2].split(",")

                for item in extra_config_items:
                    key, value = item.split("=")
                    extra_config[key] = value

        self.image_aspect_ratio = args.image_aspect_ratio
        self.skip_layer_norm = eval(extra_config.get("skip_layer_norm", "False"))

        if not delay_load:
            self.load_model()
        else:
            raise ValueError("Delay load not supported for RADIOVisionTower.")

        self.sample_count = 0
        self.debug = True

    def get_hidden_size(self):
        if self.select_feature == "cls":
            hidden_size = 5120
        elif self.select_feature == "dense":
            hidden_size = 4 * 1280
        else:
            hidden_size = 1280

        return hidden_size

    def load_model(self):
        if self.image_aspect_ratio == "resize":
            self.image_processor = ImageProcessor(
                size={"width": self.image_size, "height": self.image_size},
                do_pad=False,
                do_normalize=True,
                do_convert_rgb=True,
            )
        else:
            self.image_processor = ImageProcessor(
                size={"longest_edge": self.image_size},
                do_pad=True,
                pad_multiple=16,
                do_normalize=True,
                do_convert_rgb=True,
                pad_value=0.456,
            )
        # For compatibility with CLIP Image Processor: the data loader uses width/height to
        # create dummy blank images for samples that don't have an image.
        self.image_processor.crop_size = {"width": self.image_size, "height": self.image_size}

        mprint(self.image_processor)

        config = AutoConfig.from_pretrained(self.vision_tower_checkpoint, trust_remote_code=True)
        mprint("RADIO config", config)
        self.vision_tower = AutoModel.from_pretrained(self.vision_tower_checkpoint, trust_remote_code=True)
        self.vision_tower.radio_model.make_preprocessor_external()

        #        # NOTE: do a lazy import of Timm to avoid issues with
        #        # DeepSpeed's ZeRO-3.
        from timm.models.vision_transformer import VisionTransformer

        #
        if isinstance(self.vision_tower.model, VisionTransformer):
            hidden_size = self.vision_tower.model.embed_dim
        else:
            raise ValueError(f"Unknown model type: {self.vision_tower}")

        # Override hidden size for OpenAI CLIP.
        hidden_size = self.get_hidden_size()

        if hasattr(self.vision_tower.model, "patch_generator"):
            patch_gen = self.vision_tower.model.patch_generator
            # Cropped Positional Embedding (CPE) case.
            patch_size = patch_gen.patch_size
        else:
            # Standard ViT case.
            patch_size = self.vision_tower.model.patch_embed.patch_size[0]

        self.vision_tower.config.image_size = self.image_size
        self.vision_tower.config.hidden_size = hidden_size
        self.vision_tower.config.patch_size = patch_size

        self.vision_tower.cuda().eval()
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True
        self._to_dtype = None

        if self.skip_layer_norm:
            mprint(f"Removing layer norm from the model: {self.vision_tower.model.norm}")
            self.vision_tower.model.norm = torch.nn.Identity()

    def to(self, *args, **kwargs):
        # Prevent casting the RADIO model's weights
        kwargs = dict(kwargs)
        # self._to_dtype = kwargs.get('dtype', None)
        self._to_dtype = kwargs.pop("dtype", None)
        mprint(f"RADIO: bypass cast to dtype={self._to_dtype}")
        super().to(*args, **kwargs)
        pass

    def train(self, mode=True):
        """Intercept call."""
        # Drop a warning if mode is True.
        if mode:
            warnings.warn("RADIOEncoder is always in eval mode.")
        pass

    def _get_summary_and_patch_from_tokens(self, tokens):
        model = self.vision_tower.model
        patch_gen = getattr(model, "patch_generator", None)
        if patch_gen is not None:
            all_summary = tokens[:, : patch_gen.num_cls_tokens]
            if self.vision_tower.radio_model.summary_idxs is not None:
                summary = all_summary[:, self.vision_tower.radio_model.summary_idxs]
            else:
                summary = all_summary
            all_feat = tokens[:, patch_gen.num_skip :]
        elif model.global_pool == "avg":
            all_summary = tokens[:, model.num_prefix_tokens :].mean(dim=1)
            summary = all_summary
            all_feat = tokens
        else:
            all_summary = tokens[:, 0]
            summary = all_summary
            all_feat = tokens[:, 1:]
        return summary, all_feat

    @torch.no_grad()
    def get_features(self, x: torch.Tensor):
        x_dtype = x.dtype
        x = x.float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.select_feature == "dense":

                # Layers to return activations of in case of "return_multilayer=True".
                num_layers = len(self.vision_tower.model.blocks)
                multilayers = [
                    num_layers // 4 - 1,
                    num_layers // 2 - 1,
                    num_layers // 4 * 3 - 1,
                ]

                features = []
                intermediate_features = []

                x = self.vision_tower.input_conditioner(x)
                x = self.vision_tower.model.patch_generator(x)

                for i, blk in enumerate(self.vision_tower.model.blocks):
                    x = blk(x)
                    _, blk_features = self._get_summary_and_patch_from_tokens(x)
                    intermediate_features.append(blk_features)
                    if i in multilayers:
                        intermediate_features = torch.stack(intermediate_features, dim=0)
                        intermediate_features = torch.sum(intermediate_features, dim=0) / intermediate_features.shape[0]
                        features.append(intermediate_features)
                        intermediate_features = []
                x = self.vision_tower.model.norm(x)
                last_summary, last_features = self._get_summary_and_patch_from_tokens(x)
                features.append(last_features)
                features = torch.cat(features, dim=-1)
                summary = last_summary
            else:
                summary, features = self.vision_tower(x)

        return summary, features.to(dtype=x_dtype)

    @torch.no_grad()
    def forward(self, images: torch.Tensor):
        """Main forward pass."""
        input_shape = images.shape

        x = images
        # Add a batch dimension if necessary.
        if len(input_shape) == 3:
            x = x.unsqueeze(0)

        # Convert the input to the model's dtype (we assume
        # that the model only has one dtype for all parameters).
        param0 = next(self.vision_tower.parameters())

        rprint(
            f"input shape={input_shape}->{x.shape} device={x.device} mean={x.mean().item()} std={x.std().item()} dtype={x.dtype} param0.device={param0.device} param0.dtype={param0.dtype}"
        )

        summary, features = self.get_features(x)  # B, T, C

        if len(summary.shape) == 2:
            if self.select_feature == "cls4":
                # Add a token dimension if necessary.
                B, C = summary.shape
                summary = summary.reshape(B, 4, C // 4)
            else:
                # Add a token dimension if necessary.
                summary = summary.unsqueeze(1)

        B, _, H, W = x.shape
        _, _, C = features.shape
        patch_size = self.vision_tower.config.patch_size
        spatial_features = features.reshape(B, H // patch_size, W // patch_size, C)
        spatial_features = spatial_features.permute(0, 3, 1, 2)  # B, C, H/patch_size, W/patch_size

        if self.debug and is_rank0() and self.sample_count % 1000 == 0:
            spatial_features_hwc = spatial_features.permute(0, 2, 3, 1)
            # create the debug directory
            os.makedirs("radio-debug", exist_ok=True)
            torch.save(x, f"radio-debug/sample_{self.sample_count}_input.pt")
            torch.save(features, f"radio-debug/sample_{self.sample_count}_features.pt")
            torch.save(spatial_features_hwc, f"radio-debug/sample_{self.sample_count}_features_reshaped.pt")
            for i in range(B):
                image = x[i].permute(1, 2, 0).float() * 255
                image = Image.fromarray(image.cpu().numpy().astype(np.uint8))
                image.save(os.path.join("radio-debug/", f"sample_{self.sample_count}_preprocessed_{i}.png"))
                pca_map = get_pca_map(spatial_features_hwc[i : i + 1], x.shape[-2:])
                torch.save(pca_map, f"radio-debug/sample_{self.sample_count}_pca_map_{i}.pt")
                image = pca_map * 255
                image = Image.fromarray(image.astype(np.uint8))
                image.save(os.path.join("radio-debug/", f"sample_{self.sample_count}_pca_map_{i}.png"))
                pass

        if self.select_feature in ["patch", "cls_patch", "dense"]:
            # Ignore cls-patch for now.
            pass
        # elif self.select_feature == "cls_patch":
        #    features = torch.cat([summary, features], dim=1)
        elif self.select_feature in ["cls", "cls4"]:
            features = summary
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")

        # Remove the batch dimension if we added it.
        if len(input_shape) == 3:
            features = features.squeeze(0)

        # Cast back to the input's dtype.
        features = features.to(images.dtype)

        rprint(
            f"features shape={features.shape} mean={features.mean().item()} std={features.std().item()} dtype={features.dtype}"
        )

        if features.shape[-1] != self.get_hidden_size():
            raise ValueError(f"Unexpected hidden size: {features.shape[-1]} != {self.get_hidden_size()}")

        self.sample_count += 1

        return features


