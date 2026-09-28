"""Activation registry (what a model can expose) and the containers for captured data."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GROUPS = ("Residual", "Attention", "MLP")

# id -> (group, label, description). Order = display order. Which of these a model exposes is
# decided by the provider (see `NNsightProvider.activations`).
REGISTRY: dict[str, tuple[str, str, str]] = {
    "resid_pre": ("Residual", "resid_pre — attn_norm input",
                  "Block input (the residual stream before attention); layer 0 is the embedding."),
    "resid_mid": ("Residual", "resid_mid — mlp_norm input",
                  "Residual stream after the attention residual add."),
    "resid_post": ("Residual", "resid_post — block output",
                   "Block output; resid_post[l] == resid_pre[l+1]."),
    "attn_norm": ("Attention", "attn_norm — input_layernorm", "Output of the pre-attention RMSNorm."),
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
    "mlp_norm": ("MLP", "mlp_norm — post_attention_layernorm", "Output of the pre-MLP RMSNorm."),
    "gate": ("MLP", "gate — gate_proj", "gate_proj output."),
    "up": ("MLP", "up — up_proj", "up_proj output."),
    "silu": ("MLP", "silu — act_fn(gate)", "Activation function applied to gate."),
    "swiglu": ("MLP", "swiglu — silu(gate)·up", "Product silu(gate)*up, the input of down_proj (MLP neurons)."),
    "down": ("MLP", "down — down_proj output", "MLP output before the residual add."),
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

    def info(self) -> dict:
        """The `ActivationInfo` object of the HTTP API."""
        return {
            "id": self.id, "label": self.label, "group": self.group, "kind": self.kind,
            "n_layers": self.n_layers, "dim": self.channels,
            "layer_labels": [str(i) for i in range(self.n_layers)],
            "n_heads": self.n_heads, "head_dim": self.head_dim, "description": self.description,
        }


def make_spec(act: str, n_layers: int, channels: int | None = None, n_heads: int | None = None,
              head_dim: int | None = None) -> ActivationSpec:
    group, label, desc = REGISTRY[act]
    return ActivationSpec(act, label, group, "attn" if act == ATTN_ACT else "token", channels, n_layers,
                          n_heads, head_dim, desc)


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
