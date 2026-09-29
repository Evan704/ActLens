"""The named numeric building blocks of the slicing API, each a registry so a new one is added in one place.

- `STATS`: a per-slice statistic reduced along an axis (drives the across-layers map and the per-channel / per-token
  distributions). Register with `@stat("name", label=...)`.
- `AGGS`: how a block of cells is pooled into one when a window is larger than the screen budget.
- `ORDER_SCORES`: how channels are ranked for the non-natural display orders.

Every entry also carries the text the UI shows for it; `capabilities()` is what `GET /api/meta` serves, so the frontend
lists whatever is registered here instead of keeping its own copy.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

Reducer = Callable[[np.ndarray, int], np.ndarray]  # (x, axis) -> x reduced along `axis`

STATS: dict[str, Reducer] = {}
_INFO: dict[str, dict[str, dict]] = {"stats": {}, "aggs": {}, "orders": {}}  # kind -> id -> UI text


def stat(name: str, label: str, title: str = "", *, long_label: str | None = None,
         signed: bool = False, diverging: bool = False) -> Callable[[Reducer], Reducer]:
    """Register a statistic. `label` is the short name, `long_label` the one used in menus (default: `label`),
    `signed` says its values can be negative (rankings are then by |value|), `diverging` that the across-layers map
    uses a diverging colormap centred on zero (a signed statistic that is read as a level, e.g. kurtosis, does not)."""
    def add(fn: Reducer) -> Reducer:
        STATS[name] = fn
        _INFO["stats"][name] = {"label": label, "long_label": long_label or label, "title": title or label,
                                "signed": signed, "diverging": diverging}
        return fn
    return add


@stat("norm", "norm", "L2 norm", long_label="L2 norm")
def _norm(x: np.ndarray, axis: int) -> np.ndarray:
    return np.linalg.norm(x, axis=axis)


@stat("absmax", "|max|", "largest absolute value")
def _absmax(x: np.ndarray, axis: int) -> np.ndarray:
    return np.abs(x).max(axis=axis)


@stat("mean", "mean", "mean (signed)", signed=True, diverging=True)
def _mean(x: np.ndarray, axis: int) -> np.ndarray:
    return x.mean(axis=axis)


@stat("std", "std", "standard deviation")
def _std(x: np.ndarray, axis: int) -> np.ndarray:
    return x.std(axis=axis)


@stat("kurtosis", "kurtosis", "excess kurtosis (heavy-tailedness)",
      long_label="excess kurtosis", signed=True)
def _kurtosis(x: np.ndarray, axis: int) -> np.ndarray:
    """Excess kurtosis."""
    m = x.mean(axis=axis, keepdims=True)
    v = ((x - m) ** 2).mean(axis=axis)
    m4 = ((x - m) ** 4).mean(axis=axis)
    return m4 / np.maximum(v * v, 1e-20) - 3.0


# ----- pooling: [H, W, bh*bw] blocks (NaN-padded at the edges) -> [H, W] -----
Pooler = Callable[[np.ndarray], np.ndarray]


def _absmax_pool(blocks: np.ndarray) -> np.ndarray:
    """Keep the sign of the entry with the largest magnitude."""
    mag = np.where(np.isnan(blocks), -1.0, np.abs(blocks))
    idx = mag.argmax(axis=-1)
    return np.take_along_axis(blocks, idx[..., None], axis=-1)[..., 0]


AGGS: dict[str, Pooler] = {}


def agg(name: str, label: str) -> Callable[[Pooler], Pooler]:
    def add(fn: Pooler) -> Pooler:
        AGGS[name] = fn
        _INFO["aggs"][name] = {"label": label}
        return fn
    return add


agg("absmax", "abs-max (keeps outliers)")(_absmax_pool)
agg("mean", "mean")(lambda b: np.nanmean(b, axis=-1))
agg("max", "max")(lambda b: np.nanmax(b, axis=-1))
agg("min", "min")(lambda b: np.nanmin(b, axis=-1))

# ----- channel ranking: [T, C] -> a score per channel (higher first) -----
ORDER_SCORES: dict[str, Callable[[np.ndarray], np.ndarray]] = {}
_INFO["orders"]["natural"] = {"label": "natural index"}  # the identity order needs no score


def order(name: str, label: str) -> Callable[[Callable], Callable]:
    def add(fn):
        ORDER_SCORES[name] = fn
        _INFO["orders"][name] = {"label": label}
        return fn
    return add


order("absmax", "|max| over tokens ↓")(lambda x: np.abs(x).max(axis=0))
order("std", "std over tokens ↓")(lambda x: x.std(axis=0))
order("mean_abs", "|mean| over tokens ↓")(lambda x: np.abs(x.mean(axis=0)))

# statistics of the across-layers map that are not reductions (the value of one chosen channel)
OVERVIEW_EXTRA = {"dim": {"label": "single channel", "long_label": "single channel", "title": "one channel",
                          "signed": True, "diverging": True}}


def capabilities() -> dict:
    """What the slicing endpoints accept, with UI text, in display order (the body of `GET /api/meta`)."""
    listed = lambda kind: [{"id": k, **v} for k, v in _INFO[kind].items()]
    return {"stats": listed("stats"), "overview_extra": [{"id": k, **v} for k, v in OVERVIEW_EXTRA.items()],
            "aggs": listed("aggs"), "orders": listed("orders")}
