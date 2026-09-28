"""Rotary position embedding helpers shared by adapters."""
from __future__ import annotations

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [T, heads, Dh]; cos/sin: [1, T, rot_dim] (rot_dim <= Dh for partial rotary). Returns [T, heads, Dh]."""
    cos, sin = cos[0][:, None, :], sin[0][:, None, :]
    rot = cos.shape[-1]
    xr, xp = x[..., :rot], x[..., rot:]
    xr = xr * cos + rotate_half(xr) * sin
    return torch.cat([xr, xp], dim=-1) if xp.shape[-1] else xr
