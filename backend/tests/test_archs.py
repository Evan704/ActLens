"""Contract tests: every adapter with a tiny model in `tiny_models.TINY` must satisfy the same identities.

Adding an architecture = an adapter + one line in TINY; these tests then check it without any download.
"""
import math
import textwrap

import numpy as np
import pytest
import torch

from actlens import archs
from actlens.archs import registry
from actlens.providers import NNsightProvider

from . import tiny_models as tiny

IDS = [5, 17, 3, 42, 8, 99 % tiny.VOCAB, 1, 23, 64, 11, 2, 30]


@pytest.fixture(scope="module", params=sorted(tiny.TINY))
def prov(request):
    return NNsightProvider(f"tiny/{request.param}", device="cpu", model=tiny.TINY[request.param]())


@pytest.fixture(scope="module")
def acts(prov):
    return {s.id: prov.capture(IDS, s.id) for s in prov.activations()}


def close(a, b, rtol=1e-4, atol=1e-5):
    np.testing.assert_allclose(np.asarray(a, np.float64), np.asarray(b, np.float64), rtol=rtol, atol=atol)


def per_head(a, heads, dh):  # [L, T, heads*dh] -> [L, heads, T, dh]
    L, T, _ = a.shape
    return torch.from_numpy(a).reshape(L, T, heads, dh).permute(0, 2, 1, 3).double()


def test_registry_ids_and_shapes(prov, acts):
    T, L = len(IDS), prov.dims.n_layers
    specs = prov.activations()
    assert {"resid_pre", "resid_mid", "resid_post", "attn_norm", "q", "k", "v", "attn_pattern", "attn_ctx", "o",
            "mlp_norm", "down"} <= {s.id for s in specs}
    for s in specs:
        a = acts[s.id]
        if s.kind == "attn":
            assert a.shape == (L, s.n_heads, T, T) and a.dtype == np.float16
        else:
            assert a.shape == (L, T, s.channels) and a.dtype == np.float32, s.id
            if s.n_heads:
                assert s.channels == s.n_heads * s.head_dim, s.id
        assert np.isfinite(a.astype(np.float32)).all(), s.id
        assert s.label and s.description and s.group


def test_unknown_activation_rejected(prov):
    with pytest.raises(ValueError):
        prov.capture(IDS, "nope")


def test_capture_is_deterministic_and_independent_of_other_acts(prov, acts):
    for act in ("q", "attn_pattern", "down"):
        np.testing.assert_array_equal(prov.capture(IDS, act), acts[act])


def test_residual_stream_identities(acts):
    close(acts["resid_mid"], acts["resid_pre"] + acts["o"])
    close(acts["resid_post"], acts["resid_mid"] + acts["down"])
    close(acts["resid_post"][:-1], acts["resid_pre"][1:], rtol=1e-5, atol=1e-6)


def test_attention_pattern_is_causal_and_normalized(acts):
    p = acts["attn_pattern"].astype(np.float64)
    close(p.sum(-1), 1.0, rtol=0, atol=2e-3)
    T = p.shape[-1]
    assert (p[..., np.triu_indices(T, 1)[0], np.triu_indices(T, 1)[1]] == 0).all()


def test_attention_pattern_is_softmax_of_the_captured_q_and_k(prov, acts):
    """q/k (post-RoPE where the model has RoPE) reproduce attn_pattern. This also checks any fused-QKV split."""
    d = prov.dims
    q, k = acts.get("q_rope", acts["q"]), acts.get("k_rope", acts["k"])
    Q, K = per_head(q, d.n_heads, d.head_dim), per_head(k, d.n_kv_heads, d.head_dim)
    K = K.repeat_interleave(d.n_heads // d.n_kv_heads, dim=1)
    T = Q.shape[2]
    scores = (Q @ K.transpose(-1, -2) / math.sqrt(d.head_dim)).masked_fill(
        ~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf"))
    np.testing.assert_allclose(acts["attn_pattern"].astype(np.float64), scores.softmax(-1).numpy(), atol=2e-3)


def test_attn_ctx_is_pattern_times_v(prov, acts):
    d = prov.dims
    P = torch.from_numpy(acts["attn_pattern"].astype(np.float32)).double()
    V = per_head(acts["v"], d.n_kv_heads, d.head_dim).repeat_interleave(d.n_heads // d.n_kv_heads, dim=1)
    ctx = (P @ V).permute(0, 2, 1, 3).reshape(P.shape[0], P.shape[2], -1).numpy()
    close(acts["attn_ctx"], ctx, rtol=5e-3, atol=5e-3)


# ----- registry / plugins -----
def test_builtin_adapters_registered():
    names = {a.name for a in archs.ADAPTERS}
    assert {"llama", "gpt2"} <= names


def test_plugin_module_can_register_an_adapter(tmp_path, monkeypatch):
    src = tmp_path / "my_arch.py"
    src.write_text(textwrap.dedent('''
        from actlens.archs import ArchAdapter, register

        @register
        class Mine(ArchAdapter):
            name = "mine"
            model_types = ("mine-lm",)
            layers_path = "blocks"
            def probe(self, root):
                return [] if hasattr(root, "blocks") else ["blocks"]
    '''))
    monkeypatch.setenv("ACTLENS_ARCH_MODULES", str(src))
    monkeypatch.setattr(registry, "_plugins_loaded", False)
    before = list(registry.ADAPTERS)
    try:
        registry.load_plugins()
        assert "mine" in {a.name for a in registry.ADAPTERS} and "mine-lm" in registry.supported_model_types()
        root = type("M", (), {"blocks": [], "config": type("C", (), {"model_type": "mine-lm"})()})()
        assert registry.resolve_adapter(root).name == "mine"
    finally:
        registry.ADAPTERS[:] = before
