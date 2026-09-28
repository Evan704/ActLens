"""GPT-2 specifics on a tiny random model: the fused-QKV split and the LayerNorm / learned-position / plain-MLP paths."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from actlens.providers import NNsightProvider

from . import tiny_models as tiny
from .test_archs import IDS, close


@pytest.fixture(scope="module")
def model():
    m = tiny.gpt2()
    m.requires_grad_(False)
    return m


@pytest.fixture(scope="module")
def prov(model):
    return NNsightProvider("tiny/gpt2", device="cpu", model=model)


@pytest.fixture(scope="module")
def acts(prov):
    return {s.id: prov.capture(IDS, s.id) for s in prov.activations()}


def t(a):
    return torch.from_numpy(a).double()


def test_lists_exactly_what_gpt2_has(prov):
    ids = [s.id for s in prov.activations()]
    assert ids == ["resid_pre", "resid_mid", "resid_post", "attn_norm", "q", "k", "v", "attn_pattern", "attn_ctx",
                   "o", "mlp_norm", "up", "mlp_act", "down"]  # no gate/silu/swiglu, no QK-norm, no RoPE
    assert prov.info["adapter"] == "gpt2" and prov.info["intermediate_size"] == tiny.INTER
    labels = {s.id: s.label for s in prov.activations()}
    assert labels["attn_norm"] == "attn_norm — ln_1" and labels["q"] == "q — c_attn[q]" and labels["up"] == "up — c_fc"


def test_fused_qkv_split_matches_c_attn(model, acts):
    for l in range(tiny.LAYERS):
        w, b = model.transformer.h[l].attn.c_attn.weight.double(), model.transformer.h[l].attn.c_attn.bias.double()
        fused = (t(acts["attn_norm"][l]) @ w + b).numpy()  # Conv1D: y = x @ W + b, W is [in, out]
        got = np.concatenate([acts["q"][l], acts["k"][l], acts["v"][l]], axis=-1)
        close(got, fused, rtol=1e-4, atol=1e-5)


def test_layer0_resid_pre_is_token_plus_position_embedding(model, acts):
    tr = model.transformer
    emb = tr.wte.weight[IDS] + tr.wpe.weight[torch.arange(len(IDS))]
    close(acts["resid_pre"][0], emb.detach().numpy(), rtol=1e-6, atol=1e-6)


def test_norms_are_layernorms_of_the_residual(model, acts):
    for l in range(tiny.LAYERS):
        blk = model.transformer.h[l]
        close(acts["attn_norm"][l], F.layer_norm(t(acts["resid_pre"][l]), (tiny.D,), blk.ln_1.weight.double(),
                                                  blk.ln_1.bias.double(), blk.ln_1.eps).numpy(), rtol=1e-4, atol=1e-5)
        close(acts["mlp_norm"][l], F.layer_norm(t(acts["resid_mid"][l]), (tiny.D,), blk.ln_2.weight.double(),
                                                 blk.ln_2.bias.double(), blk.ln_2.eps).numpy(), rtol=1e-4, atol=1e-5)


def test_mlp_identities(model, acts):
    for l in range(tiny.LAYERS):
        mlp = model.transformer.h[l].mlp
        up = (t(acts["mlp_norm"][l]) @ mlp.c_fc.weight.double() + mlp.c_fc.bias.double()).numpy()
        close(acts["up"][l], up, rtol=1e-4, atol=1e-5)
        close(acts["mlp_act"][l], mlp.act(torch.from_numpy(acts["up"][l])).numpy(), rtol=1e-5, atol=1e-6)
        down = (t(acts["mlp_act"][l]) @ mlp.c_proj.weight.double() + mlp.c_proj.bias.double()).numpy()
        close(acts["down"][l], down, rtol=1e-4, atol=1e-5)


def test_attention_output_projection(model, acts):
    for l in range(tiny.LAYERS):
        proj = model.transformer.h[l].attn.c_proj
        o = (t(acts["attn_ctx"][l]) @ proj.weight.double() + proj.bias.double()).numpy()
        close(acts["o"][l], o, rtol=1e-4, atol=1e-5)


def test_tokenize_is_capped_at_the_position_table(model):
    class Tok:
        def __call__(self, text, add_special_tokens=True):
            return {"input_ids": list(range(200))}

        def decode(self, ids):
            return str(ids[0])

    p = NNsightProvider("tiny/gpt2", device="cpu", model=model)
    p.tokenizer = Tok()
    ids, tokens, truncated = p.tokenize("x", 1024)
    assert len(ids) == 64 and truncated  # n_positions=64 in the tiny config
