"""OLMo-2: the *post-norm* layout. Attention and the MLP read the raw residual stream; a norm is applied
to each branch's output before it is added (`x + post_attention_layernorm(attn(x))`), and q/k are RMS-normalized
over the whole projection before RoPE. There is no input norm, so `attn_norm`/`mlp_norm` do not exist and
`o_norm`/`down_norm` are what actually enter the stream."""
from __future__ import annotations

from dataclasses import replace

from .base import ActDef, Dims, get, has
from .llama import LlamaAdapter
from .registry import register


@register
class Olmo2Adapter(LlamaAdapter):
    name = "olmo2"
    model_types = ("olmo2",)
    post_attn_norm, post_mlp_norm = "post_attention_layernorm", "post_feedforward_layernorm"

    def probe_paths(self):
        return (["model.embed_tokens"],
                [self.post_attn_norm, self.post_mlp_norm, "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                 "self_attn.o_proj", "self_attn.q_norm", "self_attn.k_norm", "mlp.gate_proj", "mlp.up_proj",
                 "mlp.down_proj", self.act_fn])

    def probe(self, root) -> list[str]:
        missing = super().probe(root)
        if missing:
            return missing
        if has(get(root, self.layers_path)[0], "input_layernorm"):
            return ["(no input_layernorm: this adapter is for post-norm blocks)"]  # e.g. Gemma3 has all the above
        if len(set(getattr(root.config, "layer_types", None) or ())) > 1:
            # OLMo-3: sliding-window layers use their own RoPE tables, which this adapter does not select per layer
            return ["(mixed layer_types: per-layer-type RoPE and sliding windows are not supported)"]
        return missing

    def acts(self, root, d: Dims) -> dict[str, ActDef]:
        out = super().acts(root, d)
        whole = dict(  # unlike Qwen3's per-head norm, OLMo-2 normalizes the whole projection across heads
            description="Queries/keys after the RMSNorm over the entire projection (all heads together), before RoPE.")
        for name, proj in (("q_norm", "q"), ("k_norm", "k")):
            out[name] = replace(out[name], label=f"{name} — {proj} after RMSNorm over all heads", **whole)
        return out

    def dims(self, root) -> Dims:
        return Dims(**{**vars(super().dims(root)), "pre_norm": False, "branch_norm": True})

    def shared_acts(self, d: Dims) -> dict[str, ActDef]:
        D = d.hidden
        pa, pm = self.post_attn_norm, self.post_mlp_norm
        return {
            **self.core_acts(d),
            # attention reads the block input directly (q_proj is its first module) and the MLP reads the mid stream
            "resid_pre": ActDef(lambda b, c: c.tok(b.self_attn.q_proj.input), D,
                                label="resid_pre — attention input", description="Block input (the residual stream "
                                "before attention); layer 0 is the embedding."),
            "resid_mid": ActDef(lambda b, c: c.tok(b.mlp.gate_proj.input), D, label="resid_mid — MLP input"),
            "o_norm": ActDef(lambda b, c: c.tok(get(b, pa).output), D, label=f"o_norm — {pa}"),
            "down_norm": ActDef(lambda b, c: c.tok(get(b, pm).output), D, label=f"down_norm — {pm}"),
        }
