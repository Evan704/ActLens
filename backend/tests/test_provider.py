"""Model-free tests of NNsightProvider's registry logic and the RoPE helper."""
from types import SimpleNamespace

import pytest
import torch

from actlens.capture import ACT_IDS, REGISTRY
from actlens.providers import NNsightProvider, apply_rope


def bare_provider(qk_norm: bool, rope: bool, n_heads=4, n_kv=2, head_dim=8) -> NNsightProvider:
    p = NNsightProvider.__new__(NNsightProvider)
    p.model_id = "fake/llama"
    p.n_layers, p.hidden, p.inter = 5, 32, 96
    p.n_heads, p.n_kv_heads, p.head_dim = n_heads, n_kv, head_dim
    p.has_qk_norm, p.has_rope = qk_norm, rope
    return p


@pytest.mark.parametrize("qk_norm,rope", [(True, True), (False, True), (True, False), (False, False)])
def test_activations_only_list_what_the_model_has(qk_norm, rope):
    ids = [s.id for s in bare_provider(qk_norm, rope).activations()]
    assert ("q_norm" in ids) == qk_norm and ("k_norm" in ids) == qk_norm
    assert ("q_rope" in ids) == rope and ("k_rope" in ids) == rope
    assert ids == [a for a in ACT_IDS if a in ids]  # registry (display) order
    always = set(ACT_IDS) - {"q_norm", "k_norm", "q_rope", "k_rope"}
    assert always <= set(ids)


def test_activation_specs_carry_head_structure():
    specs = {s.id: s for s in bare_provider(True, True).activations()}
    assert specs["q"].channels == 32 and (specs["q"].n_heads, specs["q"].head_dim) == (4, 8)
    assert specs["k_rope"].channels == 16 and (specs["k_rope"].n_heads, specs["k_rope"].head_dim) == (2, 8)
    assert specs["attn_pattern"].channels is None and specs["attn_pattern"].n_heads == 4
    assert specs["gate"].channels == 96 and specs["gate"].n_heads is None
    assert all(s.n_layers == 5 and s.label == REGISTRY[s.id][1] and s.group == REGISTRY[s.id][0] for s in specs.values())
    info = specs["v"].info()
    assert info["layer_labels"] == ["0", "1", "2", "3", "4"] and info["dim"] == 16 and info["kind"] == "token"


def gpt2_like():
    ln = SimpleNamespace()
    return SimpleNamespace(transformer=SimpleNamespace(h=[SimpleNamespace(ln_1=ln, attn=SimpleNamespace(c_attn=ln))],
                                                       wte=ln))


def test_unsupported_architecture_gives_clear_error():
    p = bare_provider(False, False)
    p.model_id = "gpt2"
    with pytest.raises(ValueError, match="Unsupported architecture.*GPT-2"):
        p._check_arch(gpt2_like())


def test_apply_rope_matches_transformers_reference():
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    torch.manual_seed(0)
    T, H, Dh = 7, 3, 8
    x = torch.randn(T, H, Dh)
    ang = torch.randn(1, T, Dh // 2)
    cos, sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)
    ref, _ = apply_rotary_pos_emb(x.permute(1, 0, 2)[None], x.permute(1, 0, 2)[None], cos, sin)  # [1, H, T, Dh]
    torch.testing.assert_close(apply_rope(x, cos, sin), ref[0].permute(1, 0, 2))


def test_apply_rope_partial_rotary_leaves_tail_untouched():
    x = torch.randn(4, 2, 8)
    cos, sin = torch.randn(1, 4, 4), torch.randn(1, 4, 4)
    out = apply_rope(x, cos, sin)
    assert out.shape == x.shape
    torch.testing.assert_close(out[..., 4:], x[..., 4:])
    assert not torch.allclose(out[..., :4], x[..., :4])
