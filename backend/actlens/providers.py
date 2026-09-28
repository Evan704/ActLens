"""Activation providers. The rest of the app only sees `ActivationProvider`."""
from __future__ import annotations

import gc
from typing import Protocol

import numpy as np
import torch

from .capture import ACT_IDS, ATTN_ACT, ActivationSpec, make_spec


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


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [T, heads, Dh]; cos/sin: [1, T, rot_dim] (rot_dim <= Dh for partial rotary). Returns [T, heads, Dh]."""
    cos, sin = cos[0][:, None, :], sin[0][:, None, :]
    rot = cos.shape[-1]
    xr, xp = x[..., :rot], x[..., rot:]
    xr = xr * cos + _rotate_half(xr) * sin
    return torch.cat([xr, xp], dim=-1) if xp.shape[-1] else xr


UNSUPPORTED = ("Unsupported architecture for {mid}: expected a Llama-style decoder (model.layers[i] with "
               "input_layernorm, self_attn.{{q,k,v,o}}_proj, post_attention_layernorm, mlp.{{gate,up,down}}_proj; "
               "e.g. Qwen2/2.5/3, Llama, Mistral, SmolLM). Missing: {missing}. GPT-2 style models are not supported.")


class NNsightProvider:
    """Captures one activation at a time (for all layers) with nnsight on top of a plain HF model."""

    def __init__(self, model_id: str, device: str = "auto", dtype: str = "float32"):
        from nnsight import LanguageModel

        self.model_id = model_id
        self.device = pick_device(device)
        torch_dtype = getattr(torch, dtype)
        # eager attention is required to get attention probabilities back.
        # Load on CPU and move afterwards: device_map="mps" hangs on transformers 5.x.
        self.model = LanguageModel(model_id, dispatch=True, attn_implementation="eager", dtype=torch_dtype)
        self.model._model.to(self.device)
        self.model._model.eval()
        self.tokenizer = self.model.tokenizer

        root = self.model._model
        self._check_arch(root)
        self.layers = self.model.model.layers
        layer0 = root.model.layers[0]
        cfg = self.model.config
        self.n_layers = len(root.model.layers)
        self.n_heads = cfg.num_attention_heads
        self.n_kv_heads = getattr(cfg, "num_key_value_heads", None) or self.n_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // self.n_heads
        self.hidden = cfg.hidden_size
        self.inter = cfg.intermediate_size
        self.has_qk_norm = hasattr(layer0.self_attn, "q_norm") and hasattr(layer0.self_attn, "k_norm")
        self.has_rope = hasattr(root.model, "rotary_emb")
        self.act_fn_is_module = isinstance(layer0.mlp.act_fn, torch.nn.Module)

        n_params = sum(p.numel() for p in root.parameters())
        self.info = {
            "model_id": model_id,
            "device": self.device,
            "dtype": dtype,
            "n_layers": self.n_layers,
            "hidden_size": self.hidden,
            "intermediate_size": self.inter,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": cfg.vocab_size,
            "params_m": round(n_params / 1e6, 1),
            "arch": cfg.model_type,
        }

    def _check_arch(self, root) -> None:
        paths = ["model.layers", "model.embed_tokens"]
        layer_paths = ["input_layernorm", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                       "post_attention_layernorm", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "mlp.act_fn"]
        missing = [p for p in paths if not _has(root, p)]
        if not missing:
            layer0 = root.model.layers[0]
            missing = [f"layers[0].{p}" for p in layer_paths if not _has(layer0, p)]
        if missing:
            raise ValueError(UNSUPPORTED.format(mid=self.model_id, missing=", ".join(missing)))

    # ----- registry -----
    def activations(self) -> list[ActivationSpec]:
        L, D, I = self.n_layers, self.hidden, self.inter
        nH, nKV, Dh = self.n_heads, self.n_kv_heads, self.head_dim
        chans = {
            "resid_pre": (D, None, None), "resid_mid": (D, None, None), "resid_post": (D, None, None),
            "attn_norm": (D, None, None),
            "q": (nH * Dh, nH, Dh), "k": (nKV * Dh, nKV, Dh), "v": (nKV * Dh, nKV, Dh),
            "q_norm": (nH * Dh, nH, Dh), "k_norm": (nKV * Dh, nKV, Dh),
            "q_rope": (nH * Dh, nH, Dh), "k_rope": (nKV * Dh, nKV, Dh),
            "attn_ctx": (nH * Dh, nH, Dh), "o": (D, None, None),
            "mlp_norm": (D, None, None),
            "gate": (I, None, None), "up": (I, None, None), "silu": (I, None, None), "swiglu": (I, None, None),
            "down": (D, None, None),
        }
        out = []
        for act in ACT_IDS:
            if act in ("q_norm", "k_norm") and not self.has_qk_norm:
                continue
            if act in ("q_rope", "k_rope") and not self.has_rope:
                continue
            if act == ATTN_ACT:
                out.append(make_spec(act, L, None, nH, None))
            else:
                c, h, d = chans[act]
                out.append(make_spec(act, L, c, h, d))
        return out

    # ----- tokenization -----
    def tokenize(self, text: str, max_tokens: int) -> tuple[list[int], list[str], bool]:
        tok = self.tokenizer
        full_ids = tok(text, add_special_tokens=True)["input_ids"]
        truncated = len(full_ids) > max_tokens
        ids = full_ids[:max_tokens]
        if not ids:
            raise ValueError("Empty prompt")
        return ids, [tok.decode([i]) for i in ids], truncated

    # ----- capture -----
    def capture(self, token_ids: list[int], act: str) -> np.ndarray:
        import nnsight

        if act not in {s.id for s in self.activations()}:
            raise ValueError(f"unknown activation {act!r}")
        model, layers = self.model, self.layers
        T = len(token_ids)
        input_ids = torch.tensor([token_ids], device=self.device)

        def host(t: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
            return t.detach().to(dtype).cpu()

        def tok(t: torch.Tensor) -> torch.Tensor:  # [1, T, C] or [1, T, heads, Dh] -> [T, C]
            return host(t).reshape(T, -1)

        # Sources that are read from the layer; each reader returns a tensor in forward-execution order.
        rope = act in ("q_rope", "k_rope")
        is_q = act.startswith("q")
        nh = self.n_heads if is_q else self.n_kv_heads

        def read(layer, cs):
            sa, mlp = layer.self_attn, layer.mlp
            if act == "resid_pre":
                return tok(layer.input_layernorm.input)
            if act == "attn_norm":
                return tok(layer.input_layernorm.output)
            if act in ("q", "k", "v"):
                return tok(getattr(sa, f"{act}_proj").output)
            if act in ("q_norm", "k_norm"):
                return tok(getattr(sa, act).output)
            if rope:
                base = act[0]
                if self.has_qk_norm:
                    x = getattr(sa, f"{base}_norm").output
                else:
                    x = getattr(sa, f"{base}_proj").output
                x = host(x).reshape(T, nh, self.head_dim)
                return apply_rope(x, *cs).reshape(T, -1)
            if act == ATTN_ACT:
                return host(sa.output[1], torch.float16)[0]  # [H, T, T]
            if act == "attn_ctx":
                return tok(sa.o_proj.input)
            if act == "o":
                return tok(sa.o_proj.output)
            if act == "resid_mid":
                return tok(layer.post_attention_layernorm.input)
            if act == "mlp_norm":
                return tok(layer.post_attention_layernorm.output)
            if act in ("gate", "up"):
                return tok(getattr(mlp, f"{act}_proj").output)
            if act == "silu":
                if self.act_fn_is_module:
                    return tok(mlp.act_fn.output)
                return tok(torch.nn.functional.silu(mlp.gate_proj.output))
            if act == "swiglu":
                return tok(mlp.down_proj.input)
            if act == "down":
                return tok(mlp.down_proj.output)
            if act == "resid_post":
                return tok(layer.output[0] if isinstance(layer.output, tuple) else layer.output)
            raise ValueError(f"unknown activation {act!r}")

        with model.trace({"input_ids": input_ids}):
            cs = None
            if rope:  # cos/sin are produced before the first layer runs
                cs_raw = model.model.rotary_emb.output
                cs = (host(cs_raw[0]), host(cs_raw[1]))
            outs = [read(layers[i], cs) for i in range(self.n_layers)]
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


def _has(root, path: str) -> bool:
    obj = root
    for part in path.split("."):
        if part.isdigit():
            try:
                obj = obj[int(part)]
            except (IndexError, TypeError):
                return False
        elif hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return False
    return True
