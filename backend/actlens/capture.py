"""Activation registry (what a model can expose) and the containers for captured data."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .flow import STREAM

GROUPS = ("Residual", "Attention", "MLP")

# id -> (group, label, description). Order = display order. Which of these a model exposes, and under what
# label/description, is decided by its architecture adapter (see `archs.ArchAdapter.acts`); these are the defaults.
REGISTRY: dict[str, tuple[str, str, str]] = {
    "resid_pre": ("Residual", "resid_pre — attn_norm input",
                  "Block input (the residual stream before attention); layer 0 is the embedding."),
    "resid_mid": ("Residual", "resid_mid — mlp_norm input",
                  "Residual stream after the attention residual add."),
    "resid_post": ("Residual", "resid_post — block output",
                   "Block output; resid_post[l] == resid_pre[l+1]."),
    "attn_norm": ("Attention", "attn_norm — input_layernorm", "Output of the pre-attention normalization."),
    "q": ("Attention", "q — q_proj", "Query projection output (channel = head*head_dim + d)."),
    "k": ("Attention", "k — k_proj", "Key projection output (kv heads)."),
    "v": ("Attention", "v — v_proj", "Value projection output (kv heads)."),
    "q_norm": ("Attention", "q_norm — q after per-head RMSNorm", "Queries after the per-head q RMSNorm (QK-norm)."),
    "k_norm": ("Attention", "k_norm — k after per-head RMSNorm", "Keys after the per-head k RMSNorm (QK-norm)."),
    "q_rope": ("Attention", "q_rope — q after RoPE", "Queries after the rotary position embedding (what attention sees)."),
    "k_rope": ("Attention", "k_rope — k after RoPE", "Keys after the rotary position embedding (what attention sees)."),
    "attn_pattern": ("Attention", "attn_pattern — softmax probabilities",
                     "Attention probabilities [heads, query, key] (causal)."),
    "attn_ctx": ("Attention", "attn_ctx — o_proj input", "Per-head attention-weighted values, heads concatenated."),
    "o": ("Attention", "o — o_proj output", "Attention output before the residual add."),
    "o_norm": ("Attention", "o_norm — post-attention norm",
               "Attention output after the post-norm; this is what is added to the residual stream."),
    "mlp_norm": ("MLP", "mlp_norm — post_attention_layernorm", "Output of the pre-MLP normalization."),
    "gate": ("MLP", "gate — gate_proj", "gate_proj output."),
    "up": ("MLP", "up — up_proj", "up_proj output."),
    "silu": ("MLP", "silu — act_fn(gate)", "Activation function applied to gate."),
    "swiglu": ("MLP", "swiglu — silu(gate)·up", "Product silu(gate)*up, the input of down_proj (MLP neurons)."),
    "mlp_act": ("MLP", "mlp_act — act(up)",
                "Non-gated MLP: the activation function applied to up, the input of down."),
    "down": ("MLP", "down — down_proj output", "MLP output before the residual add."),
    "down_norm": ("MLP", "down_norm — post-feedforward norm",
                  "MLP output after the post-norm; this is what is added to the residual stream."),
}
ACT_IDS = tuple(REGISTRY)
ATTN_ACT = "attn_pattern"


@dataclass(frozen=True)
class ActivationSpec:
    id: str
    label: str
    group: str
    kind: str  # "token" | "attn"
    channels: int | None  # C for token kind, None for attn
    n_layers: int
    n_heads: int | None = None  # heads on this tensor when the channel axis is heads x head_dim
    head_dim: int | None = None
    description: str = ""
    inputs: tuple[str, ...] = ()  # exposed activations this one is computed from (see `flow.resolve_flow`)

    @property
    def stream(self) -> bool:
        """Whether this is a residual-stream node (the spine of the block) rather than something computed inside a branch."""
        return self.id in STREAM

    def info(self) -> dict:
        """The `ActivationInfo` object of the HTTP API."""
        return {
            "id": self.id, "label": self.label, "group": self.group, "kind": self.kind,
            "n_layers": self.n_layers, "dim": self.channels,
            "layer_labels": [str(i) for i in range(self.n_layers)],
            "n_heads": self.n_heads, "head_dim": self.head_dim, "description": self.description,
            "stream": self.stream, "inputs": list(self.inputs),
        }


def make_spec(act: str, n_layers: int, channels: int | None = None, n_heads: int | None = None,
              head_dim: int | None = None, label: str | None = None, description: str | None = None,
              inputs: tuple[str, ...] = ()) -> ActivationSpec:
    group, default_label, default_desc = REGISTRY[act]
    return ActivationSpec(act, label or default_label, group, "attn" if act == ATTN_ACT else "token", channels,
                          n_layers, n_heads, head_dim, description or default_desc, inputs)


@dataclass
class Run:
    """A tokenized prompt. Activations are captured lazily and live in the ActivationCache."""
    run_id: str
    model_id: str
    text: str
    token_ids: list[int]
    tokens: list[str]
    truncated: bool
    specs: dict[str, ActivationSpec]
    model: dict = field(default_factory=dict)
    elapsed_ms: float = 0.0

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    def spec_of_kind(self, kind: str) -> ActivationSpec | None:
        """The (first) activation of this run with the given kind, e.g. the attention pattern for "attn"."""
        return next((s for s in self.specs.values() if s.kind == kind), None)


@dataclass
class Capture:
    """One captured activation of one run: [L, T, C] float32 (token) or [L, H, T, T] float16 (attn_pattern)."""
    act: str
    arr: np.ndarray
    # Lazily computed derived data (orders, ranges, attention metrics); dropped together with the array.
    cache: dict = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return self.arr.shape[0]

    @property
    def n_heads(self) -> int:
        return self.arr.shape[1]

    @property
    def n_tokens(self) -> int:
        return self.arr.shape[-2]

    @property
    def is_attn(self) -> bool:
        return self.arr.ndim == 4

    def layer_labels(self) -> list[str]:
        return [str(i) for i in range(self.n_layers)]

    def nbytes(self) -> int:
        return self.arr.nbytes
