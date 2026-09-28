import numpy as np
import pytest

from actlens import slicing
from actlens.capture import Capture


def make_capture(act="resid_post", L=3, T=10, C=16, seed=0) -> Capture:
    """Random [L, T, C] token capture."""
    rng = np.random.default_rng(seed)
    return Capture(act, rng.normal(size=(L, T, C)).astype(np.float32))


def make_attn_capture(L=3, T=10, H=2, seed=0) -> Capture:
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(L, H, T, T))
    logits = np.where(np.tril(np.ones((T, T), bool)), logits, -np.inf)
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return Capture("attn_pattern", p.astype(np.float16))


def test_pool_absmax_keeps_sign_and_partial_blocks():
    a = np.array([[1, -5, 2], [0, 3, -1]], dtype=np.float32)
    out = slicing.pool2d(a, 2, 2, "absmax")
    assert out.shape == (1, 2)
    assert out[0, 0] == -5 and out[0, 1] == 2  # partial last column block ([2, -1]) ignores the NaN padding


@pytest.mark.parametrize("agg", slicing.AGGS)
def test_pool_matches_reference(agg):
    rng = np.random.default_rng(1)
    a = rng.normal(size=(7, 9)).astype(np.float32)
    out = slicing.pool2d(a, 3, 4, agg)
    blk = a[3:6, 4:8]
    ref = {"mean": blk.mean(), "max": blk.max(), "min": blk.min(), "absmax": blk.flat[np.abs(blk).argmax()]}[agg]
    assert out.shape == (3, 3)
    assert out[1, 1] == pytest.approx(ref, rel=1e-6)


def test_slice_window_and_dims():
    cap = make_capture()
    out, meta = slicing.token_dim_slice(cap, 2, 2, 6, 4, 12, 512, 1024, "absmax", "natural")
    assert out.shape == (4, 8) and (meta["bh"], meta["bw"]) == (1, 1)
    np.testing.assert_array_equal(out, cap.arr[2, 2:6, 4:12])
    assert meta["dims"] == list(range(4, 12)) and meta["act"] == "resid_post"


def test_slice_bins_when_window_exceeds_cap():
    cap = make_capture(C=64)
    out, meta = slicing.token_dim_slice(cap, 0, 0, 10, 0, 64, 4, 16, "mean", "natural")
    assert (meta["bh"], meta["bw"]) == (3, 4)
    assert out.shape == (4, 16) == (meta["rows"], meta["cols"])


def test_slice_clamps_out_of_range_window():
    cap = make_capture()
    out, meta = slicing.token_dim_slice(cap, 0, -5, 999, 10, 5, 512, 1024, "mean", "natural")
    assert meta["t0"] == 0 and meta["t1"] == 10
    assert meta["d0"] == 10 and meta["d1"] == 11  # inverted range collapses to the smallest valid window


def test_absmax_order_puts_outlier_dim_first():
    cap = make_capture()
    cap.arr[1, 3, 11] = 100.0
    out, meta = slicing.token_dim_slice(cap, 1, 0, 10, 0, 4, 512, 1024, "absmax", "absmax")
    assert meta["dims"][0] == 11
    assert out[3, 0] == 100.0


def test_overview_shapes_and_dim_stat():
    cap = make_capture()
    for stat in ("norm", "absmax", "mean", "std", "kurtosis"):
        out, meta = slicing.overview(cap, stat)
        assert out.shape == (3, 10) and meta["layer_labels"] == ["0", "1", "2"] and meta["act"] == "resid_post"
    out, _ = slicing.overview(cap, "dim", dim=5)
    np.testing.assert_array_equal(out, cap.arr[..., 5])
    with pytest.raises(ValueError):
        slicing.overview(cap, "nope")


def test_summarize_basic():
    v = np.arange(1, 101, dtype=np.float32)
    s = slicing.summarize(v, bins=10)
    assert s["mean"] == pytest.approx(50.5) and s["min"] == 1 and s["max"] == 100
    assert sum(s["hist"]["counts"]) == 100
    assert s["percentiles"]["50"] == pytest.approx(50.5)


def test_summarize_clip_reports_dropped():
    v = np.concatenate([np.zeros(999), [1e6]]).astype(np.float32)
    s = slicing.summarize(v, clip=True)
    assert s["hist"]["n_clipped"] >= 0 and sum(s["hist"]["counts"]) + s["hist"]["n_clipped"] == 1000


def test_summarize_constant_input():
    s = slicing.summarize(np.full(10, 3.0, dtype=np.float32))
    assert s["std"] == 0 and s["kurtosis"] == 0


def test_token_region_top_outliers():
    cap = make_capture(C=64)
    cap.arr[1, 4, 33] = -50.0
    s = slicing.token_region_stats(cap, 1, 0, 10, 0, 64, "natural", False)
    assert s["top"][0] == {"token": 4, "dim": 33, "value": -50.0}


def test_attn_slice_all_heads_and_pooling():
    cap = make_attn_capture(T=10)
    out, meta = slicing.attn_slice(cap, 1, -1, 0, 10, 0, 10, 4, 4, "max")
    assert out.shape == (2, 4, 4) and meta["heads"] == [0, 1] and (meta["bq"], meta["bk"]) == (3, 3)
    out, meta = slicing.attn_slice(cap, 1, 1, 2, 5, 0, 5, 512, 512, "max")
    assert out.shape == (1, 3, 5) and meta["heads"] == [1]


def test_attn_metrics_sane():
    cap = make_attn_capture(T=10)
    m = slicing._attn_metrics(cap)
    assert m.shape == (3, 2, 3)
    assert np.all(m[..., 0] >= 0) and np.all(m[..., 0] <= np.log(10) + 1e-3)
    assert np.all((m[..., 1] >= 0) & (m[..., 1] <= 1))
    # a head that puts all mass on key 0 has first-token mass 1 and entropy 0
    cap.arr[0, 0] = 0
    cap.arr[0, 0, :, 0] = 1
    cap.cache.clear()
    m = slicing._attn_metrics(cap)
    assert m[0, 0, 0] == pytest.approx(0, abs=1e-6) and m[0, 0, 1] == pytest.approx(1)


def test_attn_region_stats_ignores_masked_future():
    cap = make_attn_capture(T=10)
    s = slicing.attn_region_stats(cap, 0, 0, 0, 10, 0, 10, False)
    assert s["n"] == 55  # lower triangle incl. diagonal
    with pytest.raises(ValueError):
        slicing.attn_region_stats(cap, 0, 0, 0, 2, 5, 8, False)


# ---------- axis stats ----------

def _ref_kurt(x, axis):
    x = x.astype(np.float64)
    m = x.mean(axis=axis, keepdims=True)
    v = ((x - m) ** 2).mean(axis=axis)
    return ((x - m) ** 4).mean(axis=axis) / v**2 - 3


@pytest.mark.parametrize("axis", ["channel", "token"])
def test_axis_stat_values_match_numpy(axis):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(7, 5)).astype(np.float32)
    red = 0 if axis == "channel" else 1
    ref = {"norm": np.linalg.norm(x, axis=red), "absmax": np.abs(x).max(axis=red), "mean": x.mean(axis=red),
           "std": x.std(axis=red), "kurtosis": _ref_kurt(x, red)}
    for stat, r in ref.items():
        out = slicing.axis_stat_values(x, axis, stat)
        assert out.shape == ((5,) if axis == "channel" else (7,))
        np.testing.assert_allclose(out, r, rtol=1e-5, atol=1e-6)


def test_axis_stats_channel_natural_region_and_top():
    cap = make_capture(C=32, T=10)
    cap.arr[1, 4, 20] = 90.0
    cap.arr[1, 2, 7] = -70.0
    s = slicing.axis_stats(cap, 1, "channel", "absmax", 0, 10, 4, 24, "natural", False, top=2, bins=8)
    assert s["n"] == 20 and s["axis"] == "channel" and s["stat"] == "absmax"
    assert s["region"] == {"t0": 0, "t1": 10, "d0": 4, "d1": 24}
    assert s["top"][0] == {"index": 20, "id": 20, "value": 90.0}
    assert s["top"][1] == {"index": 7, "id": 7, "value": 70.0}  # per-channel |max| of channel 7 (tokens 0..10)
    assert len(s["hist"]["counts"]) == 8 and sum(s["hist"]["counts"]) == 20


def test_axis_stats_channel_with_order_reports_rank_and_id():
    cap = make_capture(C=32, T=10)
    cap.arr[0, 3, 25] = 50.0
    cap.arr[0, 6, 9] = 40.0
    s = slicing.axis_stats(cap, 0, "channel", "absmax", 0, 10, 0, 4, "absmax", False, top=2)
    # order=absmax ranks channel 25 first, then 9; the region is the top-4 ranks
    assert s["top"][0]["index"] == 0 and s["top"][0]["id"] == 25 and s["top"][0]["value"] == pytest.approx(50.0)
    assert s["top"][1]["index"] == 1 and s["top"][1]["id"] == 9
    assert s["n"] == 4


def test_axis_stats_token_axis_over_channel_subrange():
    cap = make_capture(C=16, T=10)
    cap.arr[2, 5, 3] = 200.0
    s = slicing.axis_stats(cap, 2, "token", "norm", 2, 9, 0, 8, "natural", False, top=3)
    assert s["n"] == 7 and s["region"] == {"t0": 2, "t1": 9, "d0": 0, "d1": 8}
    assert s["top"][0]["index"] == 5 == s["top"][0]["id"]
    ref = np.linalg.norm(cap.arr[2, 5, 0:8].astype(np.float64))
    assert s["top"][0]["value"] == pytest.approx(ref)
    vals = [t["value"] for t in s["top"]]
    assert vals == sorted(vals, reverse=True)


def test_axis_stats_token_axis_with_order_uses_rank_columns():
    cap = make_capture(C=16, T=6)
    cap.arr[0, :, 12] += 30.0  # channel 12 becomes rank 0 under order=absmax
    s = slicing.axis_stats(cap, 0, "token", "mean", 0, 6, 0, 1, "absmax", False, top=6)
    np.testing.assert_allclose(sorted(t["value"] for t in s["top"]), sorted(cap.arr[0, :, 12]), rtol=1e-6)


@pytest.mark.parametrize("kwargs", [
    {"axis": "bogus"}, {"stat": "bogus"}, {"t0": 5, "t1": 5}, {"t0": 20, "t1": 30}, {"d0": 3, "d1": 2}, {"bins": 0},
])
def test_axis_stats_bad_params_raise(kwargs):
    cap = make_capture()
    args = dict(axis="channel", stat="norm", t0=0, t1=10, d0=0, d1=16, order="natural", clip=False, top=3, bins=8)
    args.update(kwargs)
    with pytest.raises(ValueError):
        slicing.axis_stats(cap, 0, **args)


def test_axis_stats_single_index_and_clip():
    cap = make_capture()
    s = slicing.axis_stats(cap, 0, "token", "std", 0, 1, 0, 16, "natural", True, top=5)
    assert s["n"] == 1 and len(s["top"]) == 1
