"""Llama-style decoders: Llama, Qwen2/2.5/3, Mistral, Gemma, OLMo, SmolLM, ... (gated MLP, RoPE, optional QK-norm, GQA)."""
from __future__ import annotations

import torch

from .base import ActDef, Dims, PreNormBlockAdapter, TraceCtx, get, has
from .registry import register
from .rope import apply_rope

ROPE_ACTS = ("q_rope", "k_rope")


@register
class LlamaAdapter(PreNormBlockAdapter):
    name = "llama"
    model_types = ("llama", "qwen2", "qwen3", "mistral", "gemma", "olmo")
    layers_path = "model.layers"
    norm1, attn, o_proj = "input_layernorm", "self_attn", "self_attn.o_proj"
    norm2, down_proj = "post_attention_layernorm", "mlp.down_proj"

    def probe_paths(self):
        model_paths, block_paths = super().probe_paths()
        return ([*model_paths, "model.embed_tokens"],
                [*block_paths, "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.act_fn"])

    def dims(self, root) -> Dims:
        cfg = root.config
        n_heads = cfg.num_attention_heads
        return Dims(n_layers=len(get(root, self.layers_path)), hidden=cfg.hidden_size,
                    inter=cfg.intermediate_size, n_heads=n_heads,
                    n_kv_heads=getattr(cfg, "num_key_value_heads", None) or n_heads,
                    head_dim=getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads)

    def acts(self, root, d: Dims) -> dict[str, ActDef]:
        block0 = get(root, self.layers_path)[0]
        has_qk_norm = has(block0, "self_attn.q_norm") and has(block0, "self_attn.k_norm")
        has_rope = has(root, "model.rotary_emb")
        act_fn_is_module = isinstance(block0.mlp.act_fn, torch.nn.Module)
        nH, nKV, Dh, I = d.n_heads, d.n_kv_heads, d.head_dim, d.inter
        # OLMo clamps q/k/v in place after the projections. No adapter model has both this and QK-norm (OLMoE would
        # clamp after the norm, which `rope` below does not model).
        clip = getattr(root.config, "clip_qkv", None)
        out = self.shared_acts(d)

        def clamp(t):
            return t if clip is None else t.clamp(-clip, clip)

        def proj(name, heads):
            return ActDef(lambda b, c: clamp(c.tok(get(b, f"self_attn.{name}_proj").output)), heads * Dh, heads, Dh)

        def qk_norm(name, heads):
            return ActDef(lambda b, c: c.tok(get(b, f"self_attn.{name}_norm").output), heads * Dh, heads, Dh)

        def rope(name, heads):
            src = f"self_attn.{name}_norm" if has_qk_norm else f"self_attn.{name}_proj"

            def read(b, c: TraceCtx):
                x = c.host(get(b, src).output)
                x = (x if has_qk_norm else clamp(x)).reshape(c.n_tokens, heads, Dh)
                return apply_rope(x, *c.aux["rope"]).reshape(c.n_tokens, -1)
            return ActDef(read, heads * Dh, heads, Dh)

        out |= {"q": proj("q", nH), "k": proj("k", nKV), "v": proj("v", nKV)}
        if has_qk_norm:
            out |= {"q_norm": qk_norm("q", nH), "k_norm": qk_norm("k", nKV)}
        if has_rope:
            out |= {"q_rope": rope("q", nH), "k_rope": rope("k", nKV)}

        def silu(b, c):
            if act_fn_is_module:
                return c.tok(b.mlp.act_fn.output)
            return c.tok(torch.nn.functional.silu(b.mlp.gate_proj.output))

        cfg = root.config  # the ids stay silu/swiglu; only the names follow the model's real gate activation
        act = getattr(cfg, "hidden_activation", None) or getattr(cfg, "hidden_act", None) or "silu"
        named = {} if act == "silu" else {
            "silu": dict(label=f"silu — act_fn(gate) [{act}]", description=f"The {act} activation applied to gate."),
            "swiglu": dict(label=f"swiglu — {act}(gate)·up",
                           description=f"Product {act}(gate)*up, the input of down_proj (MLP neurons)."),
        }
        out |= {
            "gate": ActDef(lambda b, c: c.tok(b.mlp.gate_proj.output), I),
            "up": ActDef(lambda b, c: c.tok(b.mlp.up_proj.output), I),
            "silu": ActDef(silu, I, **named.get("silu", {})),
            "swiglu": ActDef(lambda b, c: c.tok(b.mlp.down_proj.input), I, **named.get("swiglu", {})),
        }
        return out

    def prepare(self, model, ctx: TraceCtx, act: str) -> None:
        if act in ROPE_ACTS:  # cos/sin are produced before the first block runs
            cos, sin = model.model.rotary_emb.output
            ctx.aux["rope"] = (ctx.host(cos), ctx.host(sin))
