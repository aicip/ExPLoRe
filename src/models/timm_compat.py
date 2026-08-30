"""Small PyTorch-only compatibility layer for core ExPLoRe model tests.

The research training environment still supports ``timm`` and should use it when
installed.  This module exists so the core model/routing test suite can execute
in minimal CPU environments where downloading third-party packages is not
possible.  It implements only the narrow layer surface used by the core models;
it is not a replacement for timm's data, optimizer, or model registry APIs.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

LayerNorm = nn.LayerNorm
trunc_normal_ = nn.init.trunc_normal_


def to_2tuple(value: int | Tuple[int, int]) -> Tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("expected a 2-tuple")
        return int(value[0]), int(value[1])
    return int(value), int(value)


def use_fused_attn() -> bool:
    """Return False to select the explicit attention path in the fallback."""

    return False


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Per-sample stochastic depth, matching timm's DropPath contract."""

    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - float(drop_prob)
    if keep_prob <= 0.0:
        return torch.zeros_like(x)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    return x * random_tensor / keep_prob


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    """Minimal timm.layers.Mlp-compatible feed-forward network."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[[], nn.Module] = nn.GELU,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        bias: bool | tuple[bool, bool] = True,
        drop: float | tuple[float, float] = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        hidden = int(hidden_features or in_features)
        out = int(out_features or in_features)
        if isinstance(bias, tuple):
            bias1, bias2 = bool(bias[0]), bool(bias[1])
        else:
            bias1 = bias2 = bool(bias)
        if isinstance(drop, tuple):
            drop1, drop2 = float(drop[0]), float(drop[1])
        else:
            drop1 = drop2 = float(drop)
        self.fc1 = nn.Linear(in_features, hidden, bias=bias1)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop1)
        self.norm = norm_layer(hidden) if norm_layer is not None else nn.Identity()
        self.fc2 = nn.Linear(hidden, out, bias=bias2)
        self.drop2 = nn.Dropout(drop2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        return self.drop2(x)


class SwiGLU(nn.Module):
    """Minimal SwiGLU MLP matching the timm call shape used by ExPLoRe."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        bias: bool = True,
        drop: float = 0.0,
        **_: object,
    ) -> None:
        super().__init__()
        hidden = int(hidden_features or in_features)
        out = int(out_features or in_features)
        self.fc1 = nn.Linear(in_features, hidden * 2, bias=bias)
        self.norm = norm_layer(hidden) if norm_layer is not None else nn.Identity()
        self.drop = nn.Dropout(float(drop))
        self.fc2 = nn.Linear(hidden, out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, value = self.fc1(x).chunk(2, dim=-1)
        x = F.silu(gate) * value
        x = self.norm(x)
        x = self.drop(x)
        return self.fc2(x)


class PatchEmbed(nn.Module):
    """2-D convolutional image-to-patch embedding used by the core ViT."""

    def __init__(
        self,
        img_size: int | Tuple[int, int] = 224,
        patch_size: int | Tuple[int, int] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable[[int], nn.Module]] = None,
        **_: object,
    ) -> None:
        super().__init__()
        self.img_size = to_2tuple(img_size)
        self.patch_size = to_2tuple(patch_size)
        if self.img_size[0] % self.patch_size[0] or self.img_size[1] % self.patch_size[1]:
            raise ValueError("image dimensions must be divisible by patch dimensions")
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(
            int(in_chans),
            int(embed_dim),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.norm = norm_layer(embed_dim) if norm_layer is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"PatchEmbed expects [B,C,H,W], got {tuple(x.shape)}")
        if tuple(x.shape[-2:]) != self.img_size:
            raise ValueError(
                f"input image size {tuple(x.shape[-2:])} does not match configured {self.img_size}"
            )
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)
