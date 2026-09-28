"""Slow test on the real GPT-2 checkpoint (skipped when it is not in the local HF cache; no downloads are attempted)."""
import os

import numpy as np
import pytest

MODEL_ID = "openai-community/gpt2"
PROMPT = "The capital of France is Paris, and the capital of Italy is Rome. Attention is all you need."


def _cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return all(isinstance(try_to_load_from_cache(model_id, f), str) for f in ("config.json", "model.safetensors"))
    except Exception:
        return False


if os.environ.get("ACTLENS_SKIP_REAL") == "1":
    pytest.skip("ACTLENS_SKIP_REAL=1", allow_module_level=True)
if not _cached(MODEL_ID):
    pytest.skip(f"{MODEL_ID} is not in the local HF cache", allow_module_level=True)

import torch  # noqa: E402

from actlens.providers import NNsightProvider  # noqa: E402

from .test_archs import close  # noqa: E402

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def prov():
    p = NNsightProvider(MODEL_ID, device="cpu")
    yield p
    p.close()


@pytest.fixture(scope="module")
def ids(prov):
    return prov.tokenize(PROMPT, 64)[0]


@pytest.fixture(scope="module")
def acts(prov, ids):
    return {s.id: prov.capture(ids, s.id) for s in prov.activations()}


def test_resolves_the_gpt2_adapter_with_real_dimensions(prov):
    d = prov.dims
    assert prov.info["adapter"] == "gpt2" and (d.n_layers, d.hidden, d.inter, d.n_heads, d.head_dim) == (12, 768, 3072, 12, 64)
    assert [s.id for s in prov.activations()][:4] == ["resid_pre", "resid_mid", "resid_post", "attn_norm"]


def test_residual_stream_identities(acts):
    close(acts["resid_mid"], acts["resid_pre"] + acts["o"], rtol=1e-4, atol=1e-3)
    close(acts["resid_post"], acts["resid_mid"] + acts["down"], rtol=1e-4, atol=1e-3)
    close(acts["resid_post"][:-1], acts["resid_pre"][1:], rtol=1e-5, atol=1e-5)


def test_final_residual_stream_reproduces_the_models_logits(prov, ids, acts):
    """ln_f(resid_post[-1]) @ W_U == the model's own logits: the captured stream is the real one end to end."""
    hf = prov.model._model
    with torch.no_grad():
        want = hf(torch.tensor([ids])).logits[0]
        got = hf.lm_head(hf.transformer.ln_f(torch.from_numpy(acts["resid_post"][-1])[None]))[0]
    torch.testing.assert_close(got, want, rtol=1e-3, atol=1e-3)
    assert got.argmax(-1).tolist() == want.argmax(-1).tolist()


def test_attention_pattern_is_softmax_of_q_and_k(prov, acts):
    import math

    d = prov.dims
    Q, K = (torch.from_numpy(acts[n]).reshape(d.n_layers, -1, d.n_heads, d.head_dim).permute(0, 2, 1, 3).double() for n in "qk")
    T = Q.shape[2]
    scores = (Q @ K.transpose(-1, -2) / math.sqrt(d.head_dim)).masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool)), float("-inf"))
    np.testing.assert_allclose(acts["attn_pattern"].astype(np.float64), scores.softmax(-1).numpy(), atol=2e-3)


def test_gpt2_massive_activation_sits_on_the_first_token(acts):
    """GPT-2's known outlier: a huge residual norm on token 0 in the middle layers."""
    norms = np.linalg.norm(acts["resid_pre"][6], axis=-1)
    assert norms[0] > 10 * np.median(norms[1:])
