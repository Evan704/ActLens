"""The dataflow of one decoder block: which activation is computed from which.

`FLOW` describes the *largest* block (every id in `capture.REGISTRY`). A model exposes a subset, and `resolve_flow`
prunes the description to it: a node the model does not expose is skipped and replaced by what it is computed from
(no `attn_norm` -> `q` reads `resid_pre`; no `o_norm` -> the residual add reads `o`). So an adapter never describes
edges: exposing the right activations is enough, and UIs draw the graph without knowing any architecture.

Each entry is a tuple of input *slots*. A slot is a preference list: the first id the model exposes is the input. Only
real alternatives need more than one id, e.g. the MLP output comes from `swiglu` (gated) or `mlp_act` (plain), and the
MLP reads `resid_mid` when the block has one (sequential) and the block input otherwise (parallel residual).
"""
from __future__ import annotations

Slot = tuple[str, ...]

# The residual-stream nodes, in order: the spine of the block. Everything else is computed inside a branch.
STREAM = ("resid_pre", "resid_mid", "resid_post")

FLOW: dict[str, tuple[Slot, ...]] = {
    "resid_pre": (),
    "attn_norm": (("resid_pre",),),
    "q": (("attn_norm",),), "k": (("attn_norm",),), "v": (("attn_norm",),),
    "q_norm": (("q",),), "k_norm": (("k",),),
    "q_rope": (("q_norm",),), "k_rope": (("k_norm",),),
    "attn_pattern": (("q_rope",), ("k_rope",)),
    "attn_ctx": (("attn_pattern",), ("v",)),
    "o": (("attn_ctx",),),
    "o_norm": (("o",),),
    "resid_mid": (("resid_pre",), ("o_norm",)),
    "mlp_norm": (("resid_mid", "resid_pre"),),
    "gate": (("mlp_norm",),), "up": (("mlp_norm",),),
    "silu": (("gate",),),
    "swiglu": (("silu",), ("up",)),
    "mlp_act": (("up",),),
    "down": (("swiglu", "mlp_act"),),
    "down_norm": (("down",),),
    "resid_post": (("resid_mid",), ("down_norm",)),
}


def _slot(slot: Slot, exposed: set[str]) -> list[str]:
    for act in slot:
        if act in exposed:
            return [act]
    if len(slot) > 1:
        return []  # a real alternative with no side exposed: guessing which one the model has would invent edges
    return [i for s in FLOW[slot[0]] for i in _slot(s, exposed)]  # skip the node, use what it is computed from


def resolve_flow(exposed: set[str]) -> dict[str, tuple[str, ...]]:
    """`{activation id: ids it is computed from}` for the exposed activations (all inputs are exposed ids). An
    activation left with no input (its whole ancestry is unexposed) hangs off the block input, so the graph stays
    connected."""
    flow = {act: tuple(dict.fromkeys(i for s in FLOW[act] for i in _slot(s, exposed))) for act in exposed}
    root = "resid_pre"
    return {act: ins or ((root,) if act != root and root in exposed else ()) for act, ins in flow.items()}
