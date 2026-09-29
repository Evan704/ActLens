"""Rotary position embedding helpers shared by adapters."""
from __future__ import annotations

import torch

from .base import ActDef, Reader, TraceCtx, get

ROPE_ACTS = ("q_rope", "k_rope")


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


def rope_act(raw: Reader, heads: int, head_dim: int) -> ActDef:
    """`q_rope`/`k_rope`: `raw` reads the pre-RoPE projection (host [T, heads*head_dim]); needs `capture_rope`
    to have run in the adapter's `prepare`."""
    def read(b, c: TraceCtx):
        x = raw(b, c).reshape(c.n_tokens, heads, head_dim)
        return apply_rope(x, *c.aux["rope"]).reshape(c.n_tokens, -1)
    return ActDef(read, heads * head_dim, heads, head_dim)


def capture_rope(model, ctx: TraceCtx, act: str, rotary_path: str) -> None:
    """Stash the model's cos/sin tables in `ctx.aux` when `act` needs them; they are produced before block 0 runs."""
    if act in ROPE_ACTS:
        cos, sin = get(model, rotary_path).output
        ctx.aux["rope"] = (ctx.host(cos), ctx.host(sin))
