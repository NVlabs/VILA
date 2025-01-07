from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import torch

from .basic import BasicVideoEncoder

__all__ = ["TSPVideoEncoder"]


def pool(x: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    return x.view(x.shape[:dim] + (-1, size) + x.shape[dim + 1 :]).mean(dim + 1)


class TSPVideoEncoder(BasicVideoEncoder):
    def __init__(
        self,
        parent: torch.nn.Module,
        pool_sizes: List[Tuple[int, int, int]],
        start_tokens: Optional[str] = None,
        end_tokens: Optional[str] = "\n",
        sep_tokens: Optional[str] = None,
    ) -> None:
        super().__init__(parent, start_tokens=start_tokens, end_tokens=end_tokens)
        self.pool_sizes = pool_sizes
        self.sep_tokens = sep_tokens

    def _process_features(
        self,
        inputs: torch.Tensor,
        start_token_embeds: Optional[torch.Tensor],
        end_token_embeds: Optional[torch.Tensor],
        sep_token_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        nt, ns = inputs.shape[:2]
        nl = int(ns**0.5)
        outputs = []
        for pool_size in self.pool_sizes:
            features = inputs.view(nt, nl, nl, -1)
            for dim, p in enumerate(pool_size):
                features = pool(features, p, dim=dim)
            features = features.flatten(1, 2)
            features = super()._process_features(
                features,
                start_token_embeds=start_token_embeds,
                end_token_embeds=end_token_embeds,
            )
            if sep_token_embeds is not None:
                features = torch.cat([features, sep_token_embeds], dim=0)
            outputs.append(features)
        return torch.cat(outputs, dim=0)

    def forward(self, videos: List[torch.Tensor], config: Dict[str, Any]) -> List[torch.Tensor]:
        num_frames = [video.shape[0] for video in videos]
        images = torch.cat(videos, dim=0)
        features = self.parent.encode_images(images)
        features = torch.split(features, num_frames)
        process_features = partial(
            self._process_features,
            start_token_embeds=self.embed_tokens(self.start_tokens),
            end_token_embeds=self.embed_tokens(self.end_tokens),
            sep_token_embeds=self.embed_tokens(self.sep_tokens),
        )
        return [process_features(f) for f in features]


