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
    d, ids = prov.dims, {s.id for s in specs}
    required = {"resid_pre", "resid_post", "q", "k", "v", "attn_pattern", "attn_ctx", "o", "down"}
    required |= set() if d.parallel_residual else {"resid_mid"}
    required |= {"attn_norm", "mlp_norm"} if d.pre_norm else set()
    required |= {"o_norm", "down_norm"} if d.branch_norm else set()
    assert required <= ids
    assert d.pre_norm == ("attn_norm" in ids) and d.parallel_residual == ("resid_mid" not in ids)
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


def test_residual_stream_identities(prov, acts):
    r = prov.dims.residual_scale
    o, down = (acts["o_norm"], acts["down_norm"]) if prov.dims.branch_norm else (acts["o"], acts["down"])
    if prov.dims.parallel_residual:  # both branches read the block input; there is no resid_mid
        assert "resid_mid" not in acts
        close(acts["resid_post"], acts["resid_pre"] + r * (o + down))
    else:
        close(acts["resid_mid"], acts["resid_pre"] + r * o)
        close(acts["resid_post"], acts["resid_mid"] + r * down)
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
    scale = 1 / math.sqrt(d.head_dim) if d.attn_scale is None else d.attn_scale
    scores = Q @ K.transpose(-1, -2) * scale
    if d.attn_softcap:
        scores = torch.tanh(scores / d.attn_softcap) * d.attn_softcap
    i, j = torch.arange(T)[:, None], torch.arange(T)[None]
    allowed = torch.stack([(j <= i) & ((i - j) < w if w else True) for w in d.windows or [None] * Q.shape[0]])
    scores = scores.masked_fill(~allowed[:, None], float("-inf"))
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


def test_gated_mlp_labels_follow_the_models_activation():
    def label(name):
        p = NNsightProvider(f"tiny/{name}", device="cpu", model=tiny.TINY[name]())
        return {s.id: s.label for s in p.activations()}["silu"]
    assert label("llama") == "silu — act_fn(gate)"
    assert "gelu" in label("gemma")
    assert label("phi3") == "silu — activation_fn(gate)"  # names the model's real module


def test_stablelm_variants_the_llama_adapter_cannot_model():
    with pytest.raises(ValueError, match="post_attention_layernorm"):  # parallel residual: no second norm
        NNsightProvider("tiny/x", device="cpu", model=tiny.auto("stablelm", use_parallel_residual=True)())
    p = NNsightProvider("tiny/x", device="cpu", model=tiny.auto("stablelm", qk_layernorm=True)())
    ids = {s.id for s in p.activations()}
    assert {"q", "k"} <= ids and not ({"q_rope", "k_rope"} & ids)  # per-head qk norm sits before RoPE


def test_post_norm_outputs_are_the_models_norm_of_the_branch_output(prov, acts):
    """`o_norm`/`down_norm` are what enters the residual stream: the block's norm applied to `o`/`down`."""
    if not prov.dims.branch_norm:
        pytest.skip("pre-norm architecture")
    layers = prov.model._model.model.layers
    for norm, raw, out in (("post_attention_layernorm", "o", "o_norm"), ("post_feedforward_layernorm", "down", "down_norm")):
        expect = np.stack([getattr(layers[i], norm)(torch.from_numpy(acts[raw][i])).detach().numpy()
                           for i in range(prov.dims.n_layers)])
        close(acts[out], expect, rtol=1e-4, atol=1e-5)


def test_olmo3_style_mixed_layer_types_are_rejected_not_silently_misread():
    """Sliding-window layers with their own RoPE tables would get the full-attention cos/sin: refuse, don't guess."""
    model = tiny.auto("olmo3", layer_types=["sliding_attention"] * (tiny.LAYERS - 1) + ["full_attention"])()
    with pytest.raises(ValueError, match="layer_types"):
        NNsightProvider("tiny/olmo3", device="cpu", model=model)


def test_olmo2_norm_labels_say_the_norm_spans_all_heads():
    p = NNsightProvider("tiny/olmo2", device="cpu", model=tiny.TINY["olmo2"]())
    labels = {s.id: s.label for s in p.activations()}
    assert "all heads" in labels["q_norm"] and "all heads" in labels["k_norm"]


def test_qk_norm_output_is_laid_out_heads_major_within_a_token(prov, acts):
    """`q_norm`/`k_norm` equal the model's own norm applied to `q`/`k`: this catches a head/token axis mix-up,
    which no shape check can (Gemma-3 normalizes after the head transpose, Qwen3 before)."""
    if "q_norm" not in acts:
        pytest.skip("no QK-norm")
    d, layers = prov.dims, prov.model._model.model.layers
    for name, heads in (("q", d.n_heads), ("k", d.n_kv_heads)):
        for i in range(d.n_layers):
            norm = getattr(layers[i].self_attn, f"{name}_norm")
            x = torch.from_numpy(acts[name][i])  # [T, heads*Dh]
            per_head = norm.weight.numel() == d.head_dim  # else the norm spans the whole projection (OLMo-2)
            expect = norm(x.reshape(len(IDS), heads, d.head_dim) if per_head else x).detach().reshape(len(IDS), -1)
            close(acts[f"{name}_norm"][i], expect.numpy(), rtol=1e-4, atol=1e-5)


def test_multimodal_gemma3_is_rejected_with_a_pointer_to_the_text_checkpoint():
    root = type("M", (), {"config": type("C", (), {"model_type": "gemma3"})()})()
    with pytest.raises(ValueError, match=r"(?i)unsupported architecture.*gemma3_text"):
        registry.resolve_adapter(root, "google/gemma-3-4b-it")
