from torch import nn

__all__ = ["BaseEncoder"]


class BaseEncoder(nn.Module):
    def __init__(self, parent: nn.Module) -> None:
        super().__init__()
        self._parent = [parent]

    @property
    def parent(self) -> nn.Module:
        return self._parent[0]


