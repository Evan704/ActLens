"""Model-free tests of NNsightProvider's registry logic (on tiny random models), adapter resolution and RoPE."""
from types import SimpleNamespace

import pytest
import torch

from actlens.archs import ADAPTERS, resolve_adapter
from actlens.archs.rope import apply_rope
from actlens.capture import ACT_IDS, REGISTRY
from actlens.providers import NNsightProvider

from . import tiny_models as tiny


def provider(model) -> NNsightProvider:
    return NNsightProvider("tiny/model", device="cpu", model=model)


@pytest.mark.parametrize("make,qk_norm", [(tiny.llama, False), (tiny.qwen3, True)])
def test_activations_only_list_what_the_model_has(make, qk_norm):
    ids = [s.id for s in provider(make()).activations()]
    assert ("q_norm" in ids) == qk_norm and ("k_norm" in ids) == qk_norm
    assert "q_rope" in ids and "k_rope" in ids
    assert ids == [a for a in ACT_IDS if a in ids]  # registry (display) order
    assert set(ids) >= set(ACT_IDS) - {"q_norm", "k_norm", "mlp_act", "o_norm", "down_norm"}  # post-norm models only


def test_model_without_rotary_emb_has_no_rope_acts():
    model = tiny.llama()
    del model.model.rotary_emb
    ids = [s.id for s in provider(model).activations()]
    assert "q_rope" not in ids and "k_rope" not in ids and "q" in ids


def test_activation_specs_carry_head_structure():
    p = provider(tiny.qwen3())
    specs = {s.id: s for s in p.activations()}
    L, D, NH, NKV, DH, I = tiny.LAYERS, tiny.D, tiny.NH, tiny.NKV, tiny.DH, tiny.INTER
    assert specs["q"].channels == NH * DH and (specs["q"].n_heads, specs["q"].head_dim) == (NH, DH)
    assert specs["k_rope"].channels == NKV * DH and (specs["k_rope"].n_heads, specs["k_rope"].head_dim) == (NKV, DH)
    assert specs["attn_pattern"].channels is None and specs["attn_pattern"].n_heads == NH
    assert specs["gate"].channels == I and specs["gate"].n_heads is None
    assert all(s.n_layers == L and s.label == REGISTRY[s.id][1] and s.group == REGISTRY[s.id][0] for s in specs.values())
    info = specs["v"].info()
    assert info["layer_labels"] == ["0", "1", "2"] and info["dim"] == NKV * DH and info["kind"] == "token"
    assert p.info["adapter"] == "llama" and p.info["arch"] == "qwen3" and p.info["hidden_size"] == D


def test_resolution_by_model_type_and_by_structure():
    model = tiny.llama()
    assert resolve_adapter(model).name == "llama"
    model.config.model_type = "some-custom-llama"  # unknown type, known module layout
    assert resolve_adapter(model).name == "llama"


def test_unsupported_architecture_gives_clear_error():
    root = SimpleNamespace(config=SimpleNamespace(model_type="mamba"), backbone=SimpleNamespace(layers=[]))
    with pytest.raises(ValueError, match=r"Unsupported architecture for x/mamba.*model_type='mamba'.*Supported model types"):
        resolve_adapter(root, "x/mamba")


def test_every_adapter_declares_a_name_and_layers_path():
    assert ADAPTERS
    for a in ADAPTERS:
        assert a.name and a.layers_path and a.model_types


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
