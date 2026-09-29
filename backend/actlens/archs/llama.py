"""Llama-style decoders: Llama, Qwen2/2.5/3, Mistral, Gemma, OLMo, StableLM, Granite, SmolLM, ... (gated MLP, RoPE, optional QK-norm, GQA)."""
from __future__ import annotations

import torch

from .base import ActDef, Dims, PreNormBlockAdapter, Reader, TraceCtx, get, has
from .registry import register
from .rope import rope_act, rope_tables


@register
class LlamaAdapter(PreNormBlockAdapter):
    name = "llama"
    model_types = ("llama", "qwen2", "qwen3", "mistral", "gemma", "olmo", "stablelm", "granite")
    layers_path = "model.layers"
    norm1, attn, o_proj = "input_layernorm", "self_attn", "self_attn.o_proj"
    norm2, down_proj = "post_attention_layernorm", "mlp.down_proj"
    act_fn = "mlp.act_fn"  # the gate activation module, relative to a block
    rotary_path = "model.rotary_emb"  # module producing (cos, sin) before the first block

    def probe_paths(self):
        model_paths, block_paths = super().probe_paths()
        return ([*model_paths, "model.embed_tokens"],
                [*block_paths, "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "mlp.gate_proj", "mlp.up_proj", self.act_fn])

    # ----- how the raw projections are read; fused-projection variants (Phi-3) override these two -----
    def proj_out(self, name: str, d: Dims) -> Reader:
        """Reader for the raw q/k/v projection output ("q" | "k" | "v"): host [T, heads*head_dim]."""
        return lambda b, c: c.tok(get(b, f"self_attn.{name}_proj").output)

    def gate_up(self, d: Dims) -> tuple[Reader, Reader]:
        """Readers for the MLP gate and up projection outputs: host [T, inter]."""
        return (lambda b, c: c.tok(b.mlp.gate_proj.output)), (lambda b, c: c.tok(b.mlp.up_proj.output))

    def dims(self, root) -> Dims:
        cfg = root.config
        n_heads = cfg.num_attention_heads
        return Dims(n_layers=len(get(root, self.layers_path)), hidden=cfg.hidden_size,
                    inter=cfg.intermediate_size, n_heads=n_heads,
                    n_kv_heads=getattr(cfg, "num_key_value_heads", None) or n_heads,
                    head_dim=getattr(cfg, "head_dim", None) or cfg.hidden_size // n_heads,
                    attn_scale=getattr(cfg, "attention_multiplier", None),  # Granite
                    residual_scale=getattr(cfg, "residual_multiplier", 1.0))

    def acts(self, root, d: Dims) -> dict[str, ActDef]:
        block0 = get(root, self.layers_path)[0]
        has_qk_norm = has(block0, "self_attn.q_norm") and has(block0, "self_attn.k_norm")
        # StableLM's per-head qk LayerNorm (`q_layernorm`) sits between the projection and RoPE and is not modelled
        has_rope = has(root, self.rotary_path) and not has(block0, "self_attn.q_layernorm")
        act_fn_is_module = isinstance(get(block0, self.act_fn), torch.nn.Module)
        act_name = self.act_fn.rsplit(".", 1)[-1]
        nH, nKV, Dh, I = d.n_heads, d.n_kv_heads, d.head_dim, d.inter
        # OLMo clamps q/k/v in place after the projections. No adapter model has both this and QK-norm (OLMoE would
        # clamp after the norm, which `rope` below does not model).
        clip = getattr(root.config, "clip_qkv", None)
        out = self.shared_acts(d)

        def clamp(t):
            return t if clip is None else t.clamp(-clip, clip)

        def proj(name, heads):
            raw = self.proj_out(name, d)
            return ActDef(lambda b, c: clamp(raw(b, c)), heads * Dh, heads, Dh)

        def qk_norm(name, heads):
            return ActDef(lambda b, c: c.tok(get(b, f"self_attn.{name}_norm").output), heads * Dh, heads, Dh)

        def rope(name, heads):
            raw = (lambda b, c: c.tok(get(b, f"self_attn.{name}_norm").output)) if has_qk_norm \
                else self.proj_out(name, d)
            return rope_act(raw if has_qk_norm else (lambda b, c: clamp(raw(b, c))), heads, Dh)

        out |= {"q": proj("q", nH), "k": proj("k", nKV), "v": proj("v", nKV)}
        if has_qk_norm:
            out |= {"q_norm": qk_norm("q", nH), "k_norm": qk_norm("k", nKV)}
        if has_rope:
            out |= {"q_rope": rope("q", nH), "k_rope": rope("k", nKV)}

        gate, up = self.gate_up(d)

        def silu(b, c):
            if act_fn_is_module:
                return c.tok(get(b, self.act_fn).output)
            return torch.nn.functional.silu(gate(b, c))

        cfg = root.config  # the ids stay silu/swiglu; only the names follow the model's real gate activation
        act = getattr(cfg, "hidden_activation", None) or getattr(cfg, "hidden_act", None) or "silu"
        named = {} if act == "silu" and act_name == "act_fn" else {
            "silu": dict(label=f"silu — {act_name}(gate)" + ("" if act == "silu" else f" [{act}]"),
                         description=f"The {act} activation applied to gate."),
            "swiglu": dict(label=f"swiglu — {act}(gate)·up",
                           description=f"Product {act}(gate)*up, the input of down_proj (MLP neurons)."),
        }
        out |= {
            "gate": ActDef(gate, I),
            "up": ActDef(up, I),
            "silu": ActDef(silu, I, **named.get("silu", {})),
            "swiglu": ActDef(lambda b, c: c.tok(b.mlp.down_proj.input), I, **named.get("swiglu", {})),
        }
        return out

    def setup(self, root, ctx, act: str) -> None:
        rope_tables(root, ctx, act, self.rotary_path)
