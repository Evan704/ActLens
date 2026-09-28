"""GPT-2 (and DistilGPT-2): LayerNorm pre-norm blocks, fused QKV, learned absolute positions (no RoPE),
non-gated MLP. A compact example of adding an architecture: a layout, the fused-QKV split and the MLP internals."""
from __future__ import annotations

from .base import ActDef, Dims, PreNormBlockAdapter, get
from .registry import register


@register
class GPT2Adapter(PreNormBlockAdapter):
    name = "gpt2"
    model_types = ("gpt2",)
    layers_path = "transformer.h"
    norm1, attn, o_proj = "ln_1", "attn", "attn.c_proj"
    norm2, down_proj = "ln_2", "mlp.c_proj"

    def probe_paths(self):
        model_paths, block_paths = super().probe_paths()
        return ([*model_paths, "transformer.wte", "transformer.wpe"],
                [*block_paths, "attn.c_attn", "mlp.c_fc", "mlp.c_proj"])

    def dims(self, root) -> Dims:
        cfg = root.config  # GPT2Config maps hidden_size / num_attention_heads onto n_embd / n_head
        return Dims(n_layers=len(get(root, self.layers_path)), hidden=cfg.hidden_size,
                    inter=getattr(cfg, "n_inner", None) or 4 * cfg.hidden_size, n_heads=cfg.num_attention_heads,
                    n_kv_heads=cfg.num_attention_heads, head_dim=cfg.hidden_size // cfg.num_attention_heads)

    def acts(self, root, d: Dims) -> dict[str, ActDef]:
        D, H, Dh, I = d.hidden, d.n_heads, d.head_dim, d.inter
        out = self.shared_acts(d)

        def qkv(i, name):  # c_attn's output is [q | k | v] along the channel axis
            return ActDef(lambda b, c: c.tok(b.attn.c_attn.output)[:, i * D:(i + 1) * D], D, H, Dh,
                          label=f"{name} — c_attn[{name}]")

        out |= {
            "q": qkv(0, "q"), "k": qkv(1, "k"), "v": qkv(2, "v"),
            "up": ActDef(lambda b, c: c.tok(b.mlp.c_fc.output), I, label="up — c_fc"),
            # c_proj's input is exactly act(c_fc(x)); reading it avoids depending on how `act` is implemented.
            "mlp_act": ActDef(lambda b, c: c.tok(b.mlp.c_proj.input), I),
        }
        return out
