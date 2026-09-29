"""Rotary position embedding helpers shared by adapters."""
from __future__ import annotations

import inspect

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
    """`q_rope`/`k_rope`: `raw` reads the pre-RoPE projection (host [T, heads*head_dim]); needs `rope_tables` to have
    run in the adapter's `setup`."""
    def read(b, c: TraceCtx):
        x = raw(b, c).reshape(c.n_tokens, heads, head_dim)
        return apply_rope(x, *c.aux["rope"](c.layer)).reshape(c.n_tokens, -1)
    return ActDef(read, heads * head_dim, heads, head_dim)


def rope_tables(root: torch.nn.Module, ctx: TraceCtx, act: str, rotary_path: str) -> None:
    """Stash `ctx.aux["rope"]`, a function layer -> (cos, sin) on the host, when `act` needs it.

    The tables come from calling the model's own rotary module on the real model (before the trace), the way its
    forward pass does. Models whose rotary module takes a `layer_type` (Gemma-3: local and global RoPE) get one table
    per layer type, chosen by `config.layer_types`."""
    if act not in ROPE_ACTS:
        return
    rot = get(root, rotary_path)
    p = next(root.parameters())
    x = torch.zeros(1, ctx.n_tokens, 1, dtype=p.dtype, device=p.device)  # the tables only take dtype/device from x
    pos = torch.arange(ctx.n_tokens, device=p.device)[None]
    types = getattr(root.config, "layer_types", None) if "layer_type" in inspect.signature(rot.forward).parameters else None
    cache: dict = {}

    def tables(layer: int):
        kind = types[layer] if types else None
        if kind not in cache:
            with torch.no_grad():
                cos, sin = rot(x, pos) if kind is None else rot(x, pos, kind)
            cache[kind] = (ctx.host(cos), ctx.host(sin))
        return cache[kind]

    ctx.aux["rope"] = tables
