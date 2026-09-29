"""The architecture-adapter contract.

An adapter answers two questions about one family of Hugging Face models: *which* activations does it expose, and
*how* is each one read from a block during a forward pass. Everything else (tokenizing, tracing, caching, the HTTP
API, the UI) is architecture-neutral and only sees the resulting `ActivationSpec`s and arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch


@dataclass(frozen=True)
class Dims:
    """Model dimensions the provider and the specs need. `inter` is None for models without an MLP width."""
    n_layers: int
    hidden: int
    inter: int | None
    n_heads: int
    n_kv_heads: int
    head_dim: int
    attn_scale: float | None = None  # softmax scale applied to q.k; None means 1/sqrt(head_dim)
    residual_scale: float = 1.0  # attention/MLP outputs are multiplied by this before the residual add (Granite)
    parallel_residual: bool = False  # x + attn(ln1 x) + mlp(ln2 x): the MLP reads the block input, no `resid_mid`


@dataclass
class TraceCtx:
    """Per-capture state handed to readers. `aux` holds values an adapter computes once before the blocks run
    (see `ArchAdapter.prepare`), e.g. the RoPE cos/sin tables."""
    n_tokens: int
    aux: dict[str, Any] = field(default_factory=dict)

    def host(self, t: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return t.detach().to(dtype).cpu()

    def tok(self, t: torch.Tensor) -> torch.Tensor:
        """[1, T, C] or [1, T, heads, Dh] -> host float32 [T, C]."""
        return self.host(t).reshape(self.n_tokens, -1)


# (block, ctx) -> host tensor: [T, C] float32 for token activations, [H, T, T] float16 for attention patterns.
# Runs inside an nnsight trace, so `block.<module>.input/output` are lazy proxies; readers must touch modules in
# forward-execution order and use `ctx` to move data to the host.
Reader = Callable[[Any, TraceCtx], torch.Tensor]


@dataclass(frozen=True)
class ActDef:
    """How one activation is exposed on one architecture. `label`/`description` override the registry defaults
    (use them to name the model's real modules); channels/n_heads/head_dim become the `ActivationSpec`."""
    read: Reader
    channels: int | None = None  # None only for attention patterns
    n_heads: int | None = None  # set when the channel axis is heads x head_dim (attention patterns: n_heads)
    head_dim: int | None = None
    label: str | None = None
    description: str | None = None


class ArchAdapter:
    """Base class. Subclass, fill in the class attributes, implement `probe`, `dims` and `acts`, and register with
    `@register` (see `archs/gpt2.py` for a small complete example)."""

    name: str = ""
    # `config.model_type` values that select this adapter directly. Adapters are also tried structurally (`probe`),
    # so a model with a different `model_type` but the same module layout works without being listed here.
    model_types: tuple[str, ...] = ()
    layers_path: str = ""  # dotted path from the model to the list of blocks, e.g. "model.layers"

    def probe(self, root: torch.nn.Module) -> list[str]:
        """Paths this adapter needs but `root` lacks. An empty list means the model is compatible."""
        raise NotImplementedError

    def dims(self, root: torch.nn.Module) -> Dims:
        raise NotImplementedError

    def acts(self, root: torch.nn.Module, dims: Dims) -> dict[str, ActDef]:
        """The activations this model exposes. Keys are registry ids (`capture.REGISTRY`); anything the model
        does not have (QK-norm, RoPE, a gate, ...) is simply left out. Display order is the registry's."""
        raise NotImplementedError

    def prepare(self, model, ctx: TraceCtx, act: str) -> None:
        """Runs inside the trace before any block executes; may stash values in `ctx.aux` for readers.
        Only do work `act` needs, since every hook adds cost."""


# ----- helpers shared by adapters -----
def get(obj, path: str):
    """Resolve a dotted path; all-digit parts index. Works on `nn.Module`s and nnsight envoys alike."""
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def has(obj, path: str) -> bool:
    """Whether `path` resolves to a module. An attribute set to None counts as absent (e.g. StableLM's
    `post_attention_layernorm` with parallel residuals)."""
    try:
        return get(obj, path) is not None
    except (AttributeError, IndexError, TypeError, KeyError):
        return False


def missing_paths(root, paths: list[str], block_paths: list[str], layers_path: str) -> list[str]:
    """Which of `paths` (relative to the model) and `block_paths` (relative to block 0) are absent from `root`."""
    missing = [p for p in [layers_path, *paths] if not has(root, p)]
    if missing:
        return missing
    return [f"{layers_path}.0.{p}" for p in block_paths if not has(get(root, layers_path)[0], p)]


def layer_output(block) -> torch.Tensor:
    out = block.output
    return out[0] if isinstance(out, tuple) else out


class PreNormBlockAdapter(ArchAdapter):
    """Sequential pre-norm blocks (`x + attn(norm1(x))`, then `x + mlp(norm2(x))`): Llama, Qwen, Mistral, GPT-2, ...

    Implements the activations that do not depend on how the projections are laid out (residual stream, norms,
    attention pattern/context/output, MLP output). Subclasses add q/k/v (+ RoPE, QK-norm) and the MLP internals.
    Module names are relative to a block.
    """

    norm1: str  # pre-attention norm
    attn: str
    o_proj: str  # attention output projection, relative to the block
    norm2: str  # pre-MLP norm
    down_proj: str  # MLP output projection, relative to the block

    def shared_acts(self, d: Dims) -> dict[str, ActDef]:
        n1, n2, o, dn = self.norm1, self.norm2, self.o_proj, self.down_proj
        D, H, Dh = d.hidden, d.n_heads, d.head_dim
        o_name, dn_name = o.rsplit(".", 1)[-1], dn.rsplit(".", 1)[-1]  # labels name the module, not its path

        return {
            "resid_pre": ActDef(lambda b, c: c.tok(get(b, n1).input), D),
            "resid_mid": ActDef(lambda b, c: c.tok(get(b, n2).input), D),
            "resid_post": ActDef(lambda b, c: c.tok(layer_output(b)), D),
            "attn_norm": ActDef(lambda b, c: c.tok(get(b, n1).output), D, label=f"attn_norm — {n1}"),
            "attn_pattern": ActDef(lambda b, c: c.host(get(b, self.attn).output[1], torch.float16)[0],
                                   None, H, None),
            "attn_ctx": ActDef(lambda b, c: c.tok(get(b, o).input), H * Dh, H, Dh, label=f"attn_ctx — {o_name} input"),
            "o": ActDef(lambda b, c: c.tok(get(b, o).output), D, label=f"o — {o_name} output"),
            "mlp_norm": ActDef(lambda b, c: c.tok(get(b, n2).output), D, label=f"mlp_norm — {n2}"),
            "down": ActDef(lambda b, c: c.tok(get(b, dn).output), D, label=f"down — {dn_name} output"),
        }

    def probe_paths(self) -> tuple[list[str], list[str]]:
        """(model-level paths, block-level paths) that must exist; subclasses extend."""
        return [], [self.norm1, self.norm2, self.attn, self.o_proj, self.down_proj]

    def probe(self, root) -> list[str]:
        model_paths, block_paths = self.probe_paths()
        return missing_paths(root, model_paths, block_paths, self.layers_path)
