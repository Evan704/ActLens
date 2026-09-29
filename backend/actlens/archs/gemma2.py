"""Gemma-2 and Gemma-3 (text): the Llama layout with a *sandwich* of norms around each branch. Attention and the MLP
read `input_layernorm(x)` / `pre_feedforward_layernorm(x)`, and their outputs go through `post_attention_layernorm` /
`post_feedforward_layernorm` before the residual add, so `o_norm`/`down_norm` are what enters the stream.

Also: softmax scale `query_pre_attn_scalar**-0.5`, attention-logit soft-capping (Gemma-2), sliding-window layers, and
(Gemma-3) per-head QK-norm and a separate RoPE table for local and global layers (see `rope.rope_tables`)."""
from __future__ import annotations

from dataclasses import replace

from .base import ActDef, Dims, Reader, get
from .llama import LlamaAdapter
from .registry import register


@register
class Gemma2Adapter(LlamaAdapter):
    name = "gemma2"
    model_types = ("gemma2", "gemma3_text")
    norm2 = "pre_feedforward_layernorm"
    sandwich_norm = True
    post_attn_norm, post_mlp_norm = "post_attention_layernorm", "post_feedforward_layernorm"

    def probe_paths(self):
        model_paths, block_paths = super().probe_paths()
        return model_paths, [*block_paths, self.post_attn_norm, self.post_mlp_norm]

    def dims(self, root) -> Dims:
        cfg = root.config
        types = getattr(cfg, "layer_types", None) or ()
        window = getattr(cfg, "sliding_window", None)
        return replace(super().dims(root), attn_scale=cfg.query_pre_attn_scalar ** -0.5,
                       attn_softcap=getattr(cfg, "attn_logit_softcapping", None), branch_norm=True,
                       windows=tuple(window if t == "sliding_attention" else None for t in types) if window else ())

    def qk_norm_out(self, name: str) -> Reader:
        # Gemma-3 normalizes q/k after the head transpose: the module output is [1, heads, T, head_dim]
        return lambda b, c: c.host(get(b, f"self_attn.{name}_norm").output)[0].transpose(0, 1).reshape(c.n_tokens, -1)

    def shared_acts(self, d: Dims) -> dict[str, ActDef]:
        return {**super().shared_acts(d), **self.branch_norm_acts(d, self.post_attn_norm, self.post_mlp_norm)}
