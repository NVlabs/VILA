from functools import partial
from typing import Any, Dict, List, Optional

import torch

from llava.model.encoders.base import BaseEncoder

__all__ = ["BasicImageEncoder"]


class BasicImageEncoder(BaseEncoder):
    def __init__(
        self,
        parent: torch.nn.Module,
        start_tokens: Optional[str] = None,
        end_tokens: Optional[str] = "\n",
    ) -> None:
        super().__init__(parent)
        self.start_tokens = start_tokens
        self.end_tokens = end_tokens

    def embed_tokens(self, tokens: Optional[str]) -> Optional[torch.Tensor]:
        if tokens is None:
            return None
        token_ids = self.parent.tokenizer(tokens).input_ids
        token_ids = torch.tensor(token_ids, device=self.parent.device)
        return self.parent.llm.model.embed_tokens(token_ids)

    def _process_features(
        self,
        features: torch.Tensor,
        start_token_embeds: Optional[torch.Tensor],
        end_token_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if start_token_embeds is not None:
            features = torch.cat([start_token_embeds, features], dim=0)
        if end_token_embeds is not None:
            features = torch.cat([features, end_token_embeds], dim=0)
        return features

    def forward(
        self,
        images: List[torch.Tensor],
        config: Dict[str, Any],
        ps3=False,
        image_num_each_sample=None,
        top_down_prompts=None,
        concat_low_high_res_features=False,
        smooth_selection_prob=False,
        num_look_close=None,
        gt_selection_maps=None,
        original_image_sizes=None,
    ) -> List[torch.Tensor]:
        images = torch.stack(images, dim=0)

        if ps3:
            features, top_down_selection_maps, top_down_selection_probs = self.parent.encode_images_ps3(
                images,
                image_num_each_sample,
                top_down_prompts,
                concat_low_high_res_features,
                smooth_selection_prob,
                num_look_close,
                gt_selection_maps,
                original_image_sizes,
            )
        else:
            features = self.parent.encode_images(images, block_sizes=config.get("block_sizes"))

        process_features = partial(
            self._process_features,
            start_token_embeds=self.embed_tokens(self.start_tokens),
            end_token_embeds=self.embed_tokens(self.end_tokens),
        )

        if ps3:
            return [process_features(f) for f in features], top_down_selection_maps, top_down_selection_probs
        else:
            return [process_features(f) for f in features]
