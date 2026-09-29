"""Activation providers. The rest of the app only sees `ActivationProvider`."""
from __future__ import annotations

import gc
from typing import Protocol

import numpy as np
import torch

from .archs import TraceCtx, get, resolve_adapter
from .capture import ACT_IDS, ActivationSpec, make_spec
from .flow import resolve_flow


class ActivationProvider(Protocol):
    model_id: str
    info: dict

    def tokenize(self, text: str, max_tokens: int) -> tuple[list[int], list[str], bool]: ...
    def activations(self) -> list[ActivationSpec]: ...
    def capture(self, token_ids: list[int], act: str) -> np.ndarray: ...
    def close(self) -> None: ...


def pick_device(requested: str | None = None) -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class NNsightProvider:
    """Captures one activation at a time (for all layers) with nnsight on top of a plain HF model.

    Everything architecture-specific lives in the `ArchAdapter` resolved for the model (see `actlens.archs`).
    `model` may be an already-loaded HF model (it must use eager attention); mainly for tests and notebooks.
    """

    def __init__(self, model_id: str, device: str = "auto", dtype: str = "float32", *, model=None):
        from nnsight import LanguageModel

        self.model_id = model_id
        self.device = pick_device(device)
        if model is None:
            # eager attention is required to get attention probabilities back.
            # Load on CPU and move afterwards: device_map="mps" hangs on transformers 5.x.
            self.model = LanguageModel(model_id, dispatch=True, attn_implementation="eager", dtype=getattr(torch, dtype))
        else:
            self.model = LanguageModel(model, dispatch=True)
            dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
        self.model._model.to(self.device)
        self.model._model.eval()
        self.tokenizer = self.model.tokenizer

        root = self.model._model
        self.adapter = resolve_adapter(root, model_id)
        self.dims = self.adapter.dims(root)
        self.layers = get(self.model, self.adapter.layers_path)
        self._defs = self.adapter.acts(root, self.dims)

        d, cfg = self.dims, root.config
        self.max_positions = getattr(cfg, "max_position_embeddings", None) or 1 << 30
        self.info = {
            "model_id": model_id,
            "device": self.device,
            "dtype": dtype,
            "n_layers": d.n_layers,
            "hidden_size": d.hidden,
            "intermediate_size": d.inter,
            "n_heads": d.n_heads,
            "n_kv_heads": d.n_kv_heads,
            "head_dim": d.head_dim,
            "vocab_size": cfg.vocab_size,
            "params_m": round(sum(p.numel() for p in root.parameters()) / 1e6, 1),
            "arch": cfg.model_type,
            "adapter": self.adapter.name,
        }

    # ----- registry -----
    def activations(self) -> list[ActivationSpec]:
        L, flow = self.dims.n_layers, resolve_flow(set(self._defs))
        return [make_spec(a, L, self._defs[a].channels, self._defs[a].n_heads, self._defs[a].head_dim,
                          self._defs[a].label, self._defs[a].description, flow[a])
                for a in ACT_IDS if a in self._defs]

    # ----- tokenization -----
    def tokenize(self, text: str, max_tokens: int) -> tuple[list[int], list[str], bool]:
        tok = self.tokenizer
        full_ids = tok(text, add_special_tokens=True)["input_ids"]
        max_tokens = min(max_tokens, self.max_positions)  # learned position tables (GPT-2) end here
        truncated = len(full_ids) > max_tokens
        ids = full_ids[:max_tokens]
        if not ids:
            raise ValueError("Empty prompt")
        return ids, [tok.decode([i]) for i in ids], truncated

    # ----- capture -----
    def capture(self, token_ids: list[int], act: str) -> np.ndarray:
        import nnsight

        if act not in self._defs:
            raise ValueError(f"unknown activation {act!r}")
        read = self._defs[act].read
        model, layers = self.model, self.layers
        ctx = TraceCtx(n_tokens=len(token_ids))
        input_ids = torch.tensor([token_ids], device=self.device)
        self.adapter.setup(model._model, ctx, act)

        def read_layer(i):
            ctx.layer = i
            return read(layers[i], ctx)

        with model.trace({"input_ids": input_ids}):
            self.adapter.prepare(model, ctx, act)
            outs = [read_layer(i) for i in range(self.dims.n_layers)]
            saved = nnsight.save(outs)

        if saved[0] is None:
            raise RuntimeError("Model did not return attention weights (needs attn_implementation='eager')")
        return torch.stack(list(saved), dim=0).numpy()

    def close(self) -> None:
        del self.model
        gc.collect()
        if self.device == "mps":
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()
