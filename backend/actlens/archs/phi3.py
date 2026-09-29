"""Phi-3 (and Phi-3.5/Phi-4-mini): the Llama layout with fused projections. `qkv_proj` outputs [q | k | v] and
`gate_up_proj` outputs [gate | up] along the channel axis; the gate activation module is `mlp.activation_fn`."""
from __future__ import annotations

from .base import Dims, PreNormBlockAdapter, Reader
from .llama import LlamaAdapter
from .registry import register


@register
class Phi3Adapter(LlamaAdapter):
    name = "phi3"
    model_types = ("phi3",)
    act_fn = "mlp.activation_fn"

    def probe_paths(self):
        model_paths, block_paths = PreNormBlockAdapter.probe_paths(self)
        return [*model_paths, "model.embed_tokens"], [*block_paths, "self_attn.qkv_proj", "mlp.gate_up_proj", self.act_fn]

    def proj_out(self, name: str, d: Dims) -> Reader:
        q_end, kv = d.n_heads * d.head_dim, d.n_kv_heads * d.head_dim
        lo, hi = {"q": (0, q_end), "k": (q_end, q_end + kv), "v": (q_end + kv, q_end + 2 * kv)}[name]
        return lambda b, c: c.tok(b.self_attn.qkv_proj.output)[:, lo:hi]

    def gate_up(self, d: Dims) -> tuple[Reader, Reader]:
        I = d.inter
        return (lambda b, c: c.tok(b.mlp.gate_up_proj.output)[:, :I],
                lambda b, c: c.tok(b.mlp.gate_up_proj.output)[:, I:])
