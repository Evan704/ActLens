"""Real `/api/run` activation lists (as the block-diagram layout needs them) for every tiny model, shared with the
frontend tests so they exercise the dataflow the backend actually produces instead of a hand-written copy.

    UPDATE_FIXTURES=1 python -m pytest tests/test_flow.py     # regenerate after changing an adapter or flow.py
"""
import json
from pathlib import Path

from actlens.providers import NNsightProvider

from . import tiny_models as tiny

PATH = Path(__file__).resolve().parents[2] / "frontend" / "src" / "fixtures" / "flows.json"
KEYS = ("id", "kind", "dim", "n_heads", "stream", "inputs")  # only what the layout uses, so text edits do not churn it


def generate() -> dict:
    out = {}
    for name in sorted(tiny.TINY):
        p = NNsightProvider(f"tiny/{name}", device="cpu", model=tiny.TINY[name]())
        out[name] = [{k: v for k, v in s.info().items() if k in KEYS} for s in p.activations()]
    return out


def dumps(data: dict) -> str:
    return json.dumps(data, indent=1, ensure_ascii=False) + "\n"
