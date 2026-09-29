"""The block dataflow shipped to UIs: `flow.resolve_flow` prunes the canonical graph to what a model exposes."""
import pytest

from actlens.capture import ACT_IDS
from actlens.flow import FLOW, resolve_flow

LLAMA = {"resid_pre", "resid_mid", "resid_post", "attn_norm", "q", "k", "v", "q_rope", "k_rope", "attn_pattern",
         "attn_ctx", "o", "mlp_norm", "gate", "up", "silu", "swiglu", "down"}


def test_the_canonical_flow_covers_exactly_the_registry():
    assert set(FLOW) == set(ACT_IDS)
    for act, slots in FLOW.items():
        assert all(i in FLOW for slot in slots for i in slot), act


def test_sequential_pre_norm_block():
    f = resolve_flow(LLAMA)
    assert f["resid_pre"] == () and f["attn_norm"] == ("resid_pre",)
    assert f["q"] == f["k"] == f["v"] == ("attn_norm",)
    assert f["q_rope"] == ("q",) and f["attn_pattern"] == ("q_rope", "k_rope")  # missing q_norm/k_norm are skipped
    assert f["attn_ctx"] == ("attn_pattern", "v") and f["o"] == ("attn_ctx",)
    assert f["resid_mid"] == ("resid_pre", "o") and f["mlp_norm"] == ("resid_mid",)
    assert f["swiglu"] == ("silu", "up") and f["down"] == ("swiglu",)
    assert f["resid_post"] == ("resid_mid", "down")


def test_plain_mlp_takes_down_from_the_activation_not_the_gate():
    f = resolve_flow((LLAMA - {"gate", "silu", "swiglu"}) | {"mlp_act"})
    assert f["up"] == ("mlp_norm",) and f["mlp_act"] == ("up",) and f["down"] == ("mlp_act",)


def test_post_norm_block_reads_the_stream_directly():
    f = resolve_flow((LLAMA - {"attn_norm", "mlp_norm"}) | {"o_norm", "down_norm", "q_norm", "k_norm"})
    assert f["q"] == f["k"] == f["v"] == ("resid_pre",)
    assert f["q_rope"] == ("q_norm",) and f["gate"] == f["up"] == ("resid_mid",)
    assert f["resid_mid"] == ("resid_pre", "o_norm") and f["resid_post"] == ("resid_mid", "down_norm")


def test_parallel_residual_has_both_branches_read_the_block_input():
    f = resolve_flow(LLAMA - {"resid_mid"})
    assert f["mlp_norm"] == ("resid_pre",)
    assert f["resid_post"] == ("resid_pre", "o", "down")


def test_an_alternative_with_no_side_exposed_invents_no_edge():
    """A plain MLP that does not expose `mlp_act`: `down` must not inherit the gated branch's inputs."""
    f = resolve_flow({"resid_pre", "resid_mid", "resid_post", "mlp_norm", "up", "down"})
    assert f["up"] == ("mlp_norm",) and f["down"] == ("resid_pre",)  # hangs off the block input, no phantom edge


def test_any_subset_of_activations_resolves_to_a_connected_acyclic_graph():
    import random
    rng = random.Random(0)
    order = list(FLOW)
    for _ in range(300):
        exposed = {a for a in order if rng.random() < 0.7} | {"resid_pre", "resid_post"}
        flow = resolve_flow(exposed)
        assert set(flow) == exposed
        for act, ins in flow.items():
            assert set(ins) <= exposed and act not in ins, (exposed, act, ins)
            assert ins or act == "resid_pre", (exposed, act)
            # FLOW is written bottom-up, so every input precedes its consumer: no cycles
            assert all(order.index(i) < order.index(act) for i in ins), (exposed, act, ins)


def test_every_model_flow_is_a_connected_acyclic_graph():
    """For each tiny model: inputs are exposed ids, the graph has no cycle, and everything derives from resid_pre."""
    from . import tiny_models as tiny
    from actlens.providers import NNsightProvider
    for name in sorted(tiny.TINY):
        specs = {s.id: s for s in NNsightProvider(f"tiny/{name}", device="cpu", model=tiny.TINY[name]()).activations()}
        seen: set[str] = set()

        def visit(a, path=()):
            assert a not in path, f"{name}: cycle through {a}"
            if a in seen:
                return
            seen.add(a)
            for i in specs[a].inputs:
                assert i in specs, f"{name}: {a} reads unexposed {i}"
                visit(i, (*path, a))

        for a in specs:
            visit(a)
            assert a == "resid_pre" or specs[a].inputs, f"{name}: {a} has no inputs"
        # every activation is downstream of the block input
        down = {"resid_pre"}
        while grown := {a for a, s in specs.items() if a not in down and set(s.inputs) & down}:
            down |= grown
        assert down == set(specs), f"{name}: not derived from resid_pre: {set(specs) - down}"
        assert specs["resid_post"].stream and not specs["q"].stream


def test_frontend_flow_fixtures_are_current():
    """frontend/src/fixtures/flows.json feeds the diagram-layout tests; it must match what the backend produces."""
    import os
    from . import flow_fixtures as fx
    fresh = fx.dumps(fx.generate())
    if os.environ.get("UPDATE_FIXTURES"):
        fx.PATH.parent.mkdir(parents=True, exist_ok=True)
        fx.PATH.write_text(fresh)
    assert fx.PATH.read_text() == fresh, "run: UPDATE_FIXTURES=1 python -m pytest tests/test_flow.py"
