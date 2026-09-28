"""Slow tests on a real model (Qwen/Qwen3-0.6B): the captured activations must satisfy the architecture's identities.

Skipped when ACTLENS_SKIP_REAL=1 or the model is not in the local Hugging Face cache (no downloads are attempted).
"""
import math
import os
import time

import numpy as np
import pytest

MODEL_ID = "Qwen/Qwen3-0.6B"
PROMPT = "The capital of France is Paris, and the capital of Italy is Rome. Attention is all you need."


def _cached(model_id: str) -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(model_id, local_files_only=True)
        return True
    except Exception:
        return False


if os.environ.get("ACTLENS_SKIP_REAL") == "1":
    pytest.skip("ACTLENS_SKIP_REAL=1", allow_module_level=True)
if not _cached(MODEL_ID):
    pytest.skip(f"{MODEL_ID} is not in the local HF cache", allow_module_level=True)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from actlens.providers import NNsightProvider  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def prov():
    p = NNsightProvider(MODEL_ID)
    yield p
    p.close()


@pytest.fixture(scope="module")
def ids(prov):
    token_ids, _, _ = prov.tokenize(PROMPT, 64)
    assert 12 <= len(token_ids) <= 64
    return token_ids


@pytest.fixture(scope="module")
def acts(prov, ids):
    """Every activation the model exposes, captured once."""
    return {s.id: prov.capture(ids, s.id) for s in prov.activations()}


def close(a, b, rtol=2e-3, atol_frac=2e-3):
    """allclose with an absolute tolerance relative to the largest magnitude (Qwen has massive activations)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    scale = max(np.abs(b).max(), 1e-6)
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol_frac * scale)


def test_registry_lists_what_qwen3_has(prov):
    specs = {s.id: s for s in prov.activations()}
    assert list(specs) == ["resid_pre", "resid_mid", "resid_post", "attn_norm", "q", "k", "v", "q_norm", "k_norm",
                           "q_rope", "k_rope", "attn_pattern", "attn_ctx", "o", "mlp_norm", "gate", "up", "silu",
                           "swiglu", "down"]
    L, D, I, nH, nKV, Dh = 28, 1024, 3072, 16, 8, 128
    assert prov.info["n_layers"] == L and prov.info["head_dim"] == Dh
    for a in ("resid_pre", "resid_mid", "resid_post", "attn_norm", "o", "mlp_norm", "down"):
        assert (specs[a].channels, specs[a].n_heads, specs[a].head_dim) == (D, None, None)
    for a in ("gate", "up", "silu", "swiglu"):
        assert specs[a].channels == I and specs[a].n_heads is None
    for a in ("q", "q_norm", "q_rope", "attn_ctx"):
        assert (specs[a].channels, specs[a].n_heads, specs[a].head_dim) == (nH * Dh, nH, Dh)
    for a in ("k", "v", "k_norm", "k_rope"):
        assert (specs[a].channels, specs[a].n_heads, specs[a].head_dim) == (nKV * Dh, nKV, Dh)
    ap = specs["attn_pattern"]
    assert (ap.kind, ap.channels, ap.n_heads, ap.head_dim) == ("attn", None, nH, None)
    assert all(s.n_layers == L for s in specs.values())


def test_capture_shapes_and_dtypes(prov, ids, acts):
    T = len(ids)
    for s in prov.activations():
        a = acts[s.id]
        if s.kind == "attn":
            assert a.shape == (s.n_layers, s.n_heads, T, T) and a.dtype == np.float16
        else:
            assert a.shape == (s.n_layers, T, s.channels) and a.dtype == np.float32, s.id
        assert np.isfinite(a.astype(np.float32)).all(), s.id


def test_unknown_activation_rejected(prov, ids):
    with pytest.raises(ValueError):
        prov.capture(ids, "nope")


def test_capture_is_deterministic_and_independent_of_other_acts(prov, ids, acts):
    np.testing.assert_array_equal(prov.capture(ids, "q_rope"), acts["q_rope"])


def test_residual_stream_identities(acts):
    close(acts["resid_mid"], acts["resid_pre"] + acts["o"])
    close(acts["resid_post"], acts["resid_mid"] + acts["down"])
    close(acts["resid_post"][:-1], acts["resid_pre"][1:], rtol=1e-4, atol_frac=1e-4)


def test_mlp_identities(acts):
    silu = F.silu(torch.from_numpy(acts["gate"])).numpy()
    close(acts["silu"], silu)
    close(acts["swiglu"], acts["silu"] * acts["up"])


def rms_norm(x, weight, eps):
    x = torch.from_numpy(x).double()
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight.double()).numpy()


def test_norm_outputs_are_rmsnorms_of_the_residual(prov, acts):
    root = prov.model._model.model
    eps = root.config.rms_norm_eps
    for l in (0, 5, 27):
        layer = root.layers[l]
        w_in = layer.input_layernorm.weight.detach().float().cpu()
        w_post = layer.post_attention_layernorm.weight.detach().float().cpu()
        close(acts["attn_norm"][l], rms_norm(acts["resid_pre"][l], w_in, eps), rtol=2e-3, atol_frac=2e-3)
        close(acts["mlp_norm"][l], rms_norm(acts["resid_mid"][l], w_post, eps), rtol=2e-3, atol_frac=2e-3)


def test_layer0_resid_pre_is_the_embedding(prov, ids, acts):
    emb = prov.model._model.model.embed_tokens.weight.detach().float().cpu().numpy()
    np.testing.assert_allclose(acts["resid_pre"][0], emb[ids], rtol=1e-6, atol=1e-6)


def test_projection_identities(prov, acts):
    """o = attn_ctx @ W_o^T and q/k/v = attn_norm @ W^T, checked on a few layers."""
    root = prov.model._model.model
    for l in (0, 13, 27):
        sa = root.layers[l].self_attn
        x = torch.from_numpy(acts["attn_norm"][l]).double()
        for name in ("q", "k", "v"):
            w = getattr(sa, f"{name}_proj").weight.detach().cpu().double()
            close(acts[name][l], (x @ w.T).numpy())
        wo = sa.o_proj.weight.detach().cpu().double()
        close(acts["o"][l], (torch.from_numpy(acts["attn_ctx"][l]).double() @ wo.T).numpy())


def per_head(a, heads, dh):  # [L, T, heads*dh] -> [L, heads, T, dh]
    L, T, _ = a.shape
    return torch.from_numpy(a).reshape(L, T, heads, dh).permute(0, 2, 1, 3).double()


def attention_probs(q, k, nH, nKV, Dh):
    """Recompute softmax(q k^T / sqrt(Dh)) with GQA repeat and a causal mask. q: [L,T,nH*Dh], k: [L,T,nKV*Dh]."""
    Q, K = per_head(q, nH, Dh), per_head(k, nKV, Dh)
    K = K.repeat_interleave(nH // nKV, dim=1)  # query head h uses kv head h // (nH // nKV)
    T = Q.shape[2]
    scores = Q @ K.transpose(-1, -2) / math.sqrt(Dh)
    scores = scores.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf"))
    return scores.softmax(-1).numpy()


def check_rope_identity(prov, acts, base_q, base_k, min_gap=0.05):
    nH, nKV, Dh = prov.dims.n_heads, prov.dims.n_kv_heads, prov.dims.head_dim
    assert nH != nKV  # exercises the GQA repeat
    probs = attention_probs(acts["q_rope"], acts["k_rope"], nH, nKV, Dh)
    got = acts["attn_pattern"].astype(np.float64)
    assert got.shape == probs.shape
    np.testing.assert_allclose(got, probs, atol=2e-3)
    assert np.abs(got - probs).mean() < 2e-5
    # each row sums to 1 and the future is exactly masked
    np.testing.assert_allclose(got.sum(-1), 1.0, atol=2e-3)
    T = got.shape[-1]
    assert (got[..., np.triu_indices(T, 1)[0], np.triu_indices(T, 1)[1]] == 0).all()
    # negative control: without RoPE the recomputation is clearly different, so the check above is discriminating
    no_rope = attention_probs(acts[base_q], acts[base_k], nH, nKV, Dh)
    assert np.abs(got - no_rope).max() > min_gap  # a random tiny model has near-uniform attention: small gap


def test_rope_identity_attention_probs_from_q_rope_k_rope(prov, acts):
    """The most important check: attn_pattern == softmax(q_rope k_rope^T / sqrt(Dh)) with GQA and a causal mask."""
    check_rope_identity(prov, acts, "q_norm", "k_norm")


def test_llama_style_model_without_qk_norm(tmp_path, ids):
    """A tiny random Llama (no q_norm/k_norm) saved locally: the q/k -> RoPE path must satisfy the same identity."""
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    tok = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256)
    LlamaForCausalLM(cfg).save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    p = NNsightProvider(str(tmp_path), device="cpu")
    try:
        got = [s.id for s in p.activations()]
        assert "q_norm" not in got and "k_norm" not in got and "q_rope" in got and "k_rope" in got
        acts = {a: p.capture(ids, a) for a in got}
        check_rope_identity(p, acts, "q", "k", min_gap=0.004)
        close(acts["resid_mid"], acts["resid_pre"] + acts["o"])
        close(acts["resid_post"], acts["resid_mid"] + acts["down"])
        close(acts["swiglu"], acts["silu"] * acts["up"])
    finally:
        p.close()


def test_rope_is_identity_at_position_zero_and_preserves_norms(prov, acts):
    Dh = prov.dims.head_dim
    for name, base, heads in (("q_rope", "q_norm", prov.dims.n_heads), ("k_rope", "k_norm", prov.dims.n_kv_heads)):
        np.testing.assert_allclose(acts[name][:, 0], acts[base][:, 0], rtol=1e-5, atol=1e-5)  # angle 0
        a = acts[name].reshape(*acts[name].shape[:2], heads, Dh)
        b = acts[base].reshape(*acts[base].shape[:2], heads, Dh)
        np.testing.assert_allclose(np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1), rtol=1e-4, atol=1e-4)


def test_attn_ctx_is_pattern_times_v(prov, acts):
    nH, nKV, Dh = prov.dims.n_heads, prov.dims.n_kv_heads, prov.dims.head_dim
    P = torch.from_numpy(acts["attn_pattern"].astype(np.float32)).double()  # [L, nH, T, T]
    V = per_head(acts["v"], nKV, Dh).repeat_interleave(nH // nKV, dim=1)  # [L, nH, T, Dh]
    ctx = (P @ V).permute(0, 2, 1, 3).reshape(P.shape[0], P.shape[2], nH * Dh).numpy()
    close(acts["attn_ctx"], ctx, rtol=5e-3, atol_frac=5e-3)


def test_through_the_http_api(prov, ids):
    """End to end: /api/run is lazy, then /slice, /axis_stats and /attn capture on demand with the real provider."""
    from fastapi.testclient import TestClient

    from actlens.app import ModelManager, create_app

    mgr = ModelManager(factory=lambda *a: prov, cache_bytes=64 * 2**20)
    with TestClient(create_app(mgr, autoload=False)) as c:
        c.post("/api/models/load", json={"model_id": MODEL_ID})
        for _ in range(100):
            if c.get("/api/status").json()["state"] == "ready":
                break
            time.sleep(0.05)
        run = c.post("/api/run", json={"text": PROMPT}).json()
        assert run["token_ids"] == ids and run["model"]["n_layers"] == 28 and len(run["activations"]) == 20
        assert mgr.cache.builds == 0
        rid = run["run_id"]
        r = c.get(f"/api/run/{rid}/axis_stats", params={"act": "q_rope", "layer": 3, "axis": "channel",
                                                        "stat": "absmax", "t0": 0, "t1": len(ids), "d0": 0, "d1": 2048})
        assert r.status_code == 200 and r.json()["n"] == 2048
        assert c.get(f"/api/run/{rid}/slice", params={"act": "q_rope", "layer": 3}).status_code == 200
        assert c.get(f"/api/run/{rid}/attn", params={"layer": 3}).status_code == 200
        assert c.get(f"/api/run/{rid}/slice", params={"act": "attn_pattern", "layer": 3}).status_code == 400
        assert mgr.cache.builds == 2  # q_rope once, attn_pattern once
