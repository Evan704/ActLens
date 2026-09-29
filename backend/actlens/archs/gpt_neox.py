"""GPT-NeoX (Pythia, Dolly, RedPajama-INCITE, ...): LayerNorm blocks, fused QKV that is interleaved per head
(`[q_h | k_h | v_h]` for each head h), partial-rotary RoPE, non-gated MLP, and by default a *parallel* residual
(`x + attn(ln1(x)) + mlp(ln2(x))`), where no `resid_mid` tensor exists."""
from __future__ import annotations

from .base import ActDef, Dims, PreNormBlockAdapter, get, has
from .registry import register
from .rope import rope_act, rope_tables


@register
class GPTNeoXAdapter(PreNormBlockAdapter):
    name = "gpt_neox"
    model_types = ("gpt_neox",)
    layers_path = "gpt_neox.layers"
    rotary_path = "gpt_neox.rotary_emb"
    norm1, attn, o_proj = "input_layernorm", "attention", "attention.dense"
    norm2, down_proj = "post_attention_layernorm", "mlp.dense_4h_to_h"

    def probe_paths(self):
        model_paths, block_paths = super().probe_paths()
        return ([*model_paths, "gpt_neox.embed_in"],
                [*block_paths, "attention.query_key_value", "mlp.dense_h_to_4h", "mlp.act"])

    def dims(self, root) -> Dims:
        cfg = root.config
        return Dims(n_layers=len(get(root, self.layers_path)), hidden=cfg.hidden_size, inter=cfg.intermediate_size,
                    n_heads=cfg.num_attention_heads, n_kv_heads=cfg.num_attention_heads,
                    head_dim=cfg.hidden_size // cfg.num_attention_heads,
                    parallel_residual=bool(cfg.use_parallel_residual))

    def acts(self, root, d: Dims) -> dict[str, ActDef]:
        H, Dh, I = d.n_heads, d.head_dim, d.inter
        out = self.shared_acts(d)
        if d.parallel_residual:  # the MLP reads the block input, so there is no stream between the two branches
            del out["resid_mid"]

        def qkv(i):  # query_key_value's output is [T, H, 3, Dh]: q, k, v of one head sit next to each other
            return lambda b, c: c.tok(b.attention.query_key_value.output).reshape(c.n_tokens, H, 3, Dh)[:, :, i] \
                .reshape(c.n_tokens, -1)

        for i, name in enumerate("qkv"):
            out[name] = ActDef(qkv(i), H * Dh, H, Dh, label=f"{name} — query_key_value[{name}]")
        out |= {
            "up": ActDef(lambda b, c: c.tok(b.mlp.dense_h_to_4h.output), I, label="up — dense_h_to_4h"),
            "mlp_act": ActDef(lambda b, c: c.tok(b.mlp.dense_4h_to_h.input), I),
        }
        if has(root, self.rotary_path):
            out |= {"q_rope": rope_act(qkv(0), H, Dh), "k_rope": rope_act(qkv(1), H, Dh)}
        return out

    def setup(self, root, ctx, act: str) -> None:
        rope_tables(root, ctx, act, self.rotary_path)
