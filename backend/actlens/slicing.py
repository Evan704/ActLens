"""Windowing, pooling, ordering and statistics on captured activations (pure numpy)."""
from __future__ import annotations

import math

import numpy as np

from .capture import Capture

AGGS = ("absmax", "mean", "max", "min")
ORDERS = ("natural", "absmax", "std", "mean_abs")
OVERVIEW_STATS = ("norm", "absmax", "mean", "std", "kurtosis", "dim")


def clamp_window(lo: int, hi: int, n: int) -> tuple[int, int]:
    lo = max(0, min(int(lo), n - 1))
    hi = max(lo + 1, min(int(hi), n))
    return lo, hi


def bin_size(extent: int, cap: int) -> int:
    return max(1, math.ceil(extent / max(1, cap)))


def pool2d(a: np.ndarray, bh: int, bw: int, agg: str) -> np.ndarray:
    """Pool a 2-D array by (bh, bw) blocks. The last block may be partial."""
    if agg not in AGGS:
        raise ValueError(f"unknown agg {agg!r}")
    if bh == 1 and bw == 1:
        return np.ascontiguousarray(a, dtype=np.float32)
    h, w = a.shape
    H, W = math.ceil(h / bh), math.ceil(w / bw)
    padded = np.full((H * bh, W * bw), np.nan, dtype=np.float32)
    padded[:h, :w] = a
    blocks = padded.reshape(H, bh, W, bw).transpose(0, 2, 1, 3).reshape(H, W, bh * bw)
    if agg == "mean":
        out = np.nanmean(blocks, axis=-1)
    elif agg == "max":
        out = np.nanmax(blocks, axis=-1)
    elif agg == "min":
        out = np.nanmin(blocks, axis=-1)
    else:  # absmax: keep the sign of the entry with the largest magnitude
        mag = np.where(np.isnan(blocks), -1.0, np.abs(blocks))
        idx = mag.argmax(axis=-1)
        out = np.take_along_axis(blocks, idx[..., None], axis=-1)[..., 0]
    return out.astype(np.float32)


def dim_order(cap: Capture, layer: int, order: str) -> np.ndarray:
    """Permutation of the channel axis for one layer. Statistics span all tokens."""
    if order not in ORDERS:
        raise ValueError(f"unknown order {order!r}")
    x = cap.arr[layer]
    D = x.shape[1]
    if order == "natural":
        return np.arange(D)
    key = (layer, order)
    hit = cap.cache.get(("order", key))
    if hit is None:
        if order == "absmax":
            score = np.abs(x).max(axis=0)
        elif order == "std":
            score = x.std(axis=0)
        else:
            score = np.abs(x.mean(axis=0))
        hit = np.argsort(-score, kind="stable")
        cap.cache[("order", key)] = hit
    return hit


def global_range(cap: Capture, layer: int) -> dict:
    key = ("range", layer)
    hit = cap.cache.get(key)
    if hit is None:
        x = cap.arr[layer]
        p = np.percentile(x, [0.5, 1, 99, 99.5])
        hit = {
            "min": float(x.min()),
            "max": float(x.max()),
            "p005": float(p[0]),
            "p01": float(p[1]),
            "p99": float(p[2]),
            "p995": float(p[3]),
            "absmax": float(np.abs(x).max()),
        }
        cap.cache[key] = hit
    return hit


def token_dim_slice(cap: Capture, layer: int, t0: int, t1: int, d0: int, d1: int,
                    max_h: int, max_w: int, agg: str, order: str) -> tuple[np.ndarray, dict]:
    x = cap.arr[layer]
    T, D = x.shape
    t0, t1 = clamp_window(t0, t1, T)
    d0, d1 = clamp_window(d0, d1, D)
    perm = dim_order(cap, layer, order)
    cols = perm[d0:d1]
    win = x[t0:t1][:, cols] if order != "natural" else x[t0:t1, d0:d1]
    bh, bw = bin_size(t1 - t0, max_h), bin_size(d1 - d0, max_w)
    out = pool2d(win, bh, bw, agg)
    meta = {
        "act": cap.act, "layer": layer, "order": order, "agg": agg,
        "t0": t0, "t1": t1, "d0": d0, "d1": d1, "n_tokens": T, "n_dims": D,
        "bh": bh, "bw": bw, "rows": out.shape[0], "cols": out.shape[1],
        "dims": cols.astype(int).tolist(),
        "range": global_range(cap, layer),
    }
    return out, meta


def dim_profile(cap: Capture, layer: int, order: str, bins: int) -> tuple[np.ndarray, dict]:
    """Per-dim absmax over all tokens (in display order), pooled to <= `bins` entries."""
    x = cap.arr[layer]
    perm = dim_order(cap, layer, order)
    prof = np.abs(x).max(axis=0)[perm][None, :]
    bw = bin_size(prof.shape[1], bins)
    out = pool2d(prof, 1, bw, "max")
    return out, {"act": cap.act, "layer": layer, "order": order, "bw": bw, "n_dims": x.shape[1], "cols": out.shape[1]}


def _kurtosis(x: np.ndarray, axis: int) -> np.ndarray:
    m = x.mean(axis=axis, keepdims=True)
    v = ((x - m) ** 2).mean(axis=axis)
    m4 = ((x - m) ** 4).mean(axis=axis)
    return m4 / np.maximum(v * v, 1e-20) - 3.0


def overview(cap: Capture, stat: str, dim: int = 0) -> tuple[np.ndarray, dict]:
    """[n_layers, T] map of a per-token statistic (over D), for a token-kind activation."""
    if stat not in OVERVIEW_STATS:
        raise ValueError(f"unknown stat {stat!r}")
    x = cap.arr
    if stat == "norm":
        out = np.linalg.norm(x, axis=-1)
    elif stat == "absmax":
        out = np.abs(x).max(axis=-1)
    elif stat == "mean":
        out = x.mean(axis=-1)
    elif stat == "std":
        out = x.std(axis=-1)
    elif stat == "kurtosis":
        out = _kurtosis(x, axis=-1)
    else:
        d = max(0, min(int(dim), x.shape[-1] - 1))
        out = x[..., d]
    out = np.ascontiguousarray(out, dtype=np.float32)
    meta = {
        "act": cap.act, "stat": stat, "dim": dim, "rows": out.shape[0], "cols": out.shape[1],
        "layer_labels": cap.layer_labels(),
        "range": {"min": float(out.min()), "max": float(out.max()),
                  "p01": float(np.percentile(out, 1)), "p99": float(np.percentile(out, 99))},
    }
    return out, meta


# ---------- attention ----------

def _attn_metrics(cap: Capture) -> np.ndarray:
    """[L, H, 3]: mean entropy (nats), mean mass on key 0 (queries >= 1), mean attention distance."""
    hit = cap.cache.get("attn_metrics")
    if hit is not None:
        return hit
    L, H, T, _ = cap.arr.shape
    out = np.zeros((L, H, 3), dtype=np.float32)
    dist = (np.arange(T)[:, None] - np.arange(T)[None, :]).astype(np.float32)
    for l in range(L):
        p = cap.arr[l].astype(np.float32)
        ent = -(np.where(p > 0, p * np.log(np.maximum(p, 1e-30)), 0.0)).sum(-1)  # [H, T]
        out[l, :, 0] = ent.mean(-1)
        out[l, :, 1] = p[:, 1:, 0].mean(-1) if T > 1 else 1.0
        out[l, :, 2] = (p * np.maximum(dist, 0)).sum(-1).mean(-1)
    cap.cache["attn_metrics"] = out
    return out


ATTN_STATS = ("entropy", "first_token", "distance")


def attn_overview(cap: Capture, stat: str) -> tuple[np.ndarray, dict]:
    if stat not in ATTN_STATS:
        raise ValueError(f"unknown attention stat {stat!r}")
    m = _attn_metrics(cap)[:, :, ATTN_STATS.index(stat)]
    out = np.ascontiguousarray(m, dtype=np.float32)
    return out, {"stat": stat, "rows": out.shape[0], "cols": out.shape[1],
                 "range": {"min": float(out.min()), "max": float(out.max())}}


def attn_slice(cap: Capture, layer: int, head: int, q0: int, q1: int, k0: int, k1: int,
               max_q: int, max_k: int, agg: str) -> tuple[np.ndarray, dict]:
    """head == -1 -> all heads, returned as [H, Q, K]; else [1, Q, K]."""
    L, H, T, _ = cap.arr.shape
    layer = max(0, min(layer, L - 1))
    q0, q1 = clamp_window(q0, q1, T)
    k0, k1 = clamp_window(k0, k1, T)
    heads = list(range(H)) if head < 0 else [max(0, min(head, H - 1))]
    bq, bk = bin_size(q1 - q0, max_q), bin_size(k1 - k0, max_k)
    pooled = [pool2d(cap.arr[layer, h, q0:q1, k0:k1].astype(np.float32), bq, bk, agg) for h in heads]
    out = np.stack(pooled, axis=0)
    metrics = _attn_metrics(cap)[layer]
    meta = {
        "layer": layer, "heads": heads, "q0": q0, "q1": q1, "k0": k0, "k1": k1, "n_tokens": T,
        "bq": bq, "bk": bk, "rows": out.shape[1], "cols": out.shape[2], "agg": agg,
        "metrics": {name: metrics[heads, i].tolist() for i, name in enumerate(ATTN_STATS)},
    }
    return out, meta


# ---------- region statistics ----------

PCTS = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9]


def summarize(values: np.ndarray, bins: int = 64, clip: bool = False) -> dict:
    v = values.astype(np.float64).ravel()
    n = v.size
    if n == 0:
        raise ValueError("empty region")
    mean = float(v.mean())
    std = float(v.std())
    kurt = float(((v - mean) ** 4).mean() / (std ** 4) - 3.0) if std > 0 else 0.0
    skew = float(((v - mean) ** 3).mean() / (std ** 3)) if std > 0 else 0.0
    pct = np.percentile(v, PCTS)
    lo, hi = (float(pct[0]), float(pct[-1])) if clip else (float(v.min()), float(v.max()))
    if hi <= lo:
        hi = lo + 1e-9
    counts, edges = np.histogram(v, bins=bins, range=(lo, hi))
    return {
        "n": int(n), "mean": mean, "std": std, "min": float(v.min()), "max": float(v.max()),
        "absmax": float(np.abs(v).max()), "l2": float(np.linalg.norm(v)),
        "skew": skew, "kurtosis": kurt,
        "frac_pos": float((v > 0).mean()),
        "frac_3sigma": float((np.abs(v - mean) > 3 * std).mean()) if std > 0 else 0.0,
        "percentiles": {str(p): float(x) for p, x in zip(PCTS, pct)},
        "hist": {"counts": counts.astype(int).tolist(), "edges": edges.tolist(),
                 "n_clipped": int(n - counts.sum())},
    }


def token_region_stats(cap: Capture, layer: int, t0: int, t1: int, d0: int, d1: int,
                       order: str, clip: bool, top: int = 8) -> dict:
    x = cap.arr[layer]
    T, D = x.shape
    t0, t1 = clamp_window(t0, t1, T)
    d0, d1 = clamp_window(d0, d1, D)
    perm = dim_order(cap, layer, order)
    cols = perm[d0:d1]
    region = x[t0:t1][:, cols]
    out = summarize(region, clip=clip)
    flat = np.abs(region).ravel()
    k = min(top, flat.size)
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx])]
    rr, cc = np.unravel_index(idx, region.shape)
    out["top"] = [{"token": int(t0 + r), "dim": int(cols[c]), "value": float(region[r, c])} for r, c in zip(rr, cc)]
    out["region"] = {"t0": t0, "t1": t1, "d0": d0, "d1": d1}
    return out


def attn_region_stats(cap: Capture, layer: int, head: int, q0: int, q1: int, k0: int, k1: int,
                      clip: bool) -> dict:
    L, H, T, _ = cap.arr.shape
    q0, q1 = clamp_window(q0, q1, T)
    k0, k1 = clamp_window(k0, k1, T)
    p = cap.arr[max(0, min(layer, L - 1)), max(0, min(head, H - 1)), q0:q1, k0:k1].astype(np.float32)
    # causal mask: keys after the query are exactly zero and would distort every statistic
    valid = (np.arange(k0, k1)[None, :] <= np.arange(q0, q1)[:, None])
    if not valid.any():
        raise ValueError("region contains only masked (future) positions")
    out = summarize(p[valid], clip=clip)
    out["region"] = {"q0": q0, "q1": q1, "k0": k0, "k1": k1}
    return out


# ---------- per-channel / per-token statistic distributions ----------

AXES = ("channel", "token")
AXIS_STATS = ("norm", "absmax", "mean", "std", "kurtosis")


def _strict_window(lo: int, hi: int, n: int, name: str) -> tuple[int, int]:
    """Clamp [lo, hi) to [0, n); unlike `clamp_window`, an empty result is an error."""
    lo, hi = max(0, min(int(lo), n)), max(0, min(int(hi), n))
    if hi <= lo:
        raise ValueError(f"empty {name} range")
    return lo, hi


def _check_axis_stat(axis: str, stat: str) -> None:
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}; expected one of {list(AXES)}")
    if stat not in AXIS_STATS:
        raise ValueError(f"unknown stat {stat!r}; expected one of {list(AXIS_STATS)}")


def axis_stat_values(region: np.ndarray, axis: str, stat: str) -> np.ndarray:
    """Reduce a [T, C] region to one statistic per channel (axis='channel', over tokens)
    or per token (axis='token', over channels). `norm` is L2, `kurtosis` is excess kurtosis."""
    _check_axis_stat(axis, stat)
    x = region.astype(np.float64)
    red = 0 if axis == "channel" else 1
    if stat == "norm":
        return np.sqrt((x * x).sum(axis=red))
    if stat == "absmax":
        return np.abs(x).max(axis=red)
    if stat == "mean":
        return x.mean(axis=red)
    if stat == "std":
        return x.std(axis=red)
    return _kurtosis(x, axis=red)


def axis_stats(cap: Capture, layer: int, axis: str, stat: str, t0: int, t1: int, d0: int, d1: int,
               order: str, clip: bool, top: int = 10, bins: int = 64) -> dict:
    """Distribution of a per-channel / per-token statistic over the region [t0,t1) x [d0,d1)
    (d in `order`-rank coordinates). Returns `summarize()` of the per-index values plus the top-|value| indices."""
    _check_axis_stat(axis, stat)
    if not 0 <= top <= 1000 or not 1 <= bins <= 4096:
        raise ValueError("top must be in 0..1000 and bins in 1..4096")
    x = cap.arr[layer]
    T, C = x.shape
    t0, t1 = _strict_window(t0, t1, T, "token")
    d0, d1 = _strict_window(d0, d1, C, "channel")
    perm = dim_order(cap, layer, order)
    cols = perm[d0:d1]
    region = x[t0:t1, d0:d1] if order == "natural" else x[t0:t1][:, cols]
    vals = axis_stat_values(region, axis, stat)
    out = summarize(vals, bins=bins, clip=clip)
    k = min(top, vals.size)
    if k:
        idx = np.argpartition(-np.abs(vals), k - 1)[:k]
        idx = idx[np.argsort(-np.abs(vals[idx]), kind="stable")]
    else:
        idx = np.empty(0, dtype=int)
    if axis == "channel":
        out["top"] = [{"index": int(d0 + i), "id": int(cols[i]), "value": float(vals[i])} for i in idx]
    else:
        out["top"] = [{"index": int(t0 + i), "id": int(t0 + i), "value": float(vals[i])} for i in idx]
    out["axis"], out["stat"] = axis, stat
    out["region"] = {"t0": t0, "t1": t1, "d0": d0, "d1": d1}
    return out
