import json
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from fastapi.testclient import TestClient

from actlens import slicing
from actlens.app import PRESET_MODELS, ModelManager, create_app
from actlens.capture import Capture, make_spec

L, T_MAX, D, NH, DH, INTER = 3, 64, 16, 2, 4, 64


class FakeProvider:
    """Deterministic stand-in for NNsightProvider that records every capture() call."""

    delay = 0.0  # seconds each capture takes
    fail_acts: set = set()

    def __init__(self, model_id, device="auto", dtype="float32"):
        if model_id == "bad/model":
            raise OSError("not found")
        self.model_id = model_id
        self.info = {"model_id": model_id, "n_layers": L, "hidden_size": D, "intermediate_size": INTER,
                     "n_heads": NH, "n_kv_heads": NH, "head_dim": DH}
        self.calls: list[str] = []
        self.threads: set[str] = set()

    def tokenize(self, text, max_tokens):
        if not text.strip():
            raise ValueError("Empty prompt")
        ids = [ord(c) for c in text]
        return ids[:max_tokens], list(text[:max_tokens]), len(ids) > max_tokens

    def activations(self):
        return [
            make_spec("resid_pre", L, D), make_spec("resid_post", L, D),
            make_spec("q", L, NH * DH, NH, DH), make_spec("gate", L, INTER),
            make_spec("attn_pattern", L, None, NH, None),
        ]

    def capture(self, token_ids, act):
        self.calls.append(act)
        self.threads.add(threading.current_thread().name)
        time.sleep(self.delay)
        if act in self.fail_acts:
            raise RuntimeError("boom")
        T = len(token_ids)
        rng = np.random.default_rng(sum(token_ids) + len(act))
        if act == "attn_pattern":
            logits = np.where(np.tril(np.ones((T, T), bool)), rng.normal(size=(L, NH, T, T)), -np.inf)
            p = np.exp(logits - logits.max(-1, keepdims=True))
            return (p / p.sum(-1, keepdims=True)).astype(np.float16)
        c = {s.id: s.channels for s in self.activations()}[act]
        return rng.normal(size=(L, T, c)).astype(np.float32)

    def close(self):
        pass


def decode(content: bytes):
    n = struct.unpack("<I", content[:4])[0]
    meta = json.loads(content[4:4 + n])
    off = 4 + n + (-(4 + n)) % 4
    arr = np.frombuffer(content, dtype="<f4", offset=off).reshape(meta["shape"])
    return meta, arr


def wait_ready(c):
    for _ in range(200):
        if c.get("/api/status").json()["state"] == "ready":
            return
        time.sleep(0.02)
    raise AssertionError("model did not become ready")


@pytest.fixture(autouse=True)
def reset_fake():
    FakeProvider.delay, FakeProvider.fail_acts = 0.0, set()


@pytest.fixture()
def make_client():
    clients = []

    def make(cache_bytes=None):
        mgr = ModelManager(factory=FakeProvider, cache_bytes=cache_bytes)
        c = TestClient(create_app(mgr, autoload=False))
        c.__enter__()
        clients.append(c)
        c.post("/api/models/load", json={"model_id": "fake/model"})
        wait_ready(c)
        return c

    yield make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture()
def client(make_client):
    return make_client()


def provider(c) -> FakeProvider:
    return c.app.state.manager.provider


def start_run(c, text="hello world!"):
    r = c.post("/api/run", json={"text": text})
    assert r.status_code == 200, r.text
    return r.json()


# ---------- models / run registration ----------

def test_status_and_corpus(client):
    s = client.get("/api/status").json()
    assert s["state"] == "ready" and s["model_id"] == "fake/model" and s["presets"]
    assert len({p["id"] for p in PRESET_MODELS}) == len(PRESET_MODELS)
    corpus = client.get("/api/corpus").json()
    assert len(corpus) >= 10 and all({"id", "title", "text"} <= set(c) for c in corpus)


def test_run_response_shape_and_no_capture(client):
    run = start_run(client)
    assert set(run) == {"run_id", "model_id", "tokens", "token_ids", "truncated", "elapsed_ms", "model", "activations"}
    assert run["tokens"] == list("hello world!") and run["token_ids"][0] == ord("h") and run["truncated"] is False
    assert run["model"] == {"n_layers": L, "hidden_size": D, "intermediate_size": INTER, "n_heads": NH,
                            "n_kv_heads": NH, "head_dim": DH}
    acts = {a["id"]: a for a in run["activations"]}
    assert list(acts) == ["resid_pre", "resid_post", "q", "gate", "attn_pattern"]  # display order, only supported ids
    assert acts["q"] == {
        "id": "q", "label": "q — q_proj", "group": "Attention", "kind": "token", "n_layers": L, "dim": NH * DH,
        "layer_labels": ["0", "1", "2"], "n_heads": NH, "head_dim": DH, "description": acts["q"]["description"]}
    assert acts["resid_pre"]["group"] == "Residual" and acts["resid_pre"]["n_heads"] is None
    assert acts["gate"]["group"] == "MLP" and acts["gate"]["head_dim"] is None
    ap = acts["attn_pattern"]
    assert ap["kind"] == "attn" and ap["dim"] is None and ap["n_heads"] == NH and ap["head_dim"] is None
    assert all(a["description"] for a in acts.values())
    assert "sites" not in run and "attn" not in run and "size_mb" not in run
    assert provider(client).calls == []  # registering a run captures nothing


def test_run_truncation(client):
    run = client.post("/api/run", json={"text": "abcdefghij", "max_tokens": 4}).json()
    assert run["tokens"] == list("abcd") and run["truncated"] is True


# ---------- lazy capture ----------

def test_capture_happens_once_per_act_across_endpoints(client):
    rid = start_run(client)["run_id"]
    base = f"/api/run/{rid}"
    p = {"act": "resid_pre", "layer": 1}
    for url, extra in [("slice", {}), ("slice", {"t0": 2}), ("stats", {}), ("profile", {}), ("axis_stats", {}),
                       ("overview", {"stat": "norm"})]:
        assert client.get(f"{base}/{url}", params={**p, **extra}).status_code == 200
    assert provider(client).calls == ["resid_pre"]
    assert client.get(f"{base}/slice", params={"act": "gate", "layer": 0}).status_code == 200
    assert provider(client).calls == ["resid_pre", "gate"]
    # attn endpoints implicitly use attn_pattern, captured once
    for url, extra in [("attn", {"layer": 0}), ("attn_overview", {}), ("attn_stats", {"layer": 0, "head": 0})]:
        assert client.get(f"{base}/{url}", params=extra).status_code == 200
    assert provider(client).calls == ["resid_pre", "gate", "attn_pattern"]
    # every capture ran on the single model thread
    assert provider(client).threads and all(t.startswith("model") for t in provider(client).threads)


def test_capture_is_per_run(client):
    a, b = start_run(client, "aaa")["run_id"], start_run(client, "bbb")["run_id"]
    for rid in (a, b, a, b):
        client.get(f"/api/run/{rid}/stats", params={"act": "q", "layer": 0})
    assert provider(client).calls == ["q", "q"]


def test_concurrent_requests_share_one_capture(make_client):
    c = make_client()
    FakeProvider.delay = 0.3
    rid = start_run(c)["run_id"]

    def hit(i):
        params = {"act": "gate", "layer": i % L}
        return c.get(f"/api/run/{rid}/{('slice', 'stats', 'axis_stats')[i % 3]}", params=params).status_code

    with ThreadPoolExecutor(8) as ex:
        codes = list(ex.map(hit, range(8)))
    assert codes == [200] * 8
    assert provider(c).calls == ["gate"]


def test_lru_eviction_under_tiny_budget(make_client):
    nbytes = L * 12 * D * 4  # one resid capture of a 12-token prompt
    c = make_client(cache_bytes=int(nbytes * 2.2))  # room for two resid captures, not two plus q
    rid = start_run(c)["run_id"]
    get = lambda act: c.get(f"/api/run/{rid}/stats", params={"act": act, "layer": 0}).status_code
    assert [get("resid_pre"), get("resid_post")] == [200, 200]
    assert provider(c).calls == ["resid_pre", "resid_post"]
    assert get("resid_pre") == 200  # hit; makes resid_post the LRU entry
    assert get("q") == 200  # 2*L*12*D*4 + L*12*8*4 > budget -> evicts resid_post
    keys = {k[1] for k in c.app.state.manager.cache.keys()}
    assert keys == {"resid_pre", "q"}
    assert get("resid_pre") == 200 and provider(c).calls == ["resid_pre", "resid_post", "q"]
    assert get("resid_post") == 200 and provider(c).calls[-1] == "resid_post"  # re-captured after eviction
    assert c.app.state.manager.cache.nbytes <= c.app.state.manager.cache.budget


def test_entry_larger_than_budget_is_kept_alone(make_client):
    c = make_client(cache_bytes=1)
    rid = start_run(c)["run_id"]
    for _ in range(2):
        assert c.get(f"/api/run/{rid}/stats", params={"act": "resid_pre", "layer": 0}).status_code == 200
    assert provider(c).calls == ["resid_pre"]  # the entry just built is never evicted
    assert c.get(f"/api/run/{rid}/stats", params={"act": "resid_post", "layer": 0}).status_code == 200
    assert {k[1] for k in c.app.state.manager.cache.keys()} == {"resid_post"}


def test_run_eviction_drops_cached_activations(client):
    first = start_run(client, "prompt 0")["run_id"]
    client.get(f"/api/run/{first}/stats", params={"act": "q", "layer": 0})
    for i in range(1, 5):
        start_run(client, f"prompt {i}")
    assert client.get(f"/api/run/{first}/stats", params={"act": "q", "layer": 0}).status_code == 404
    assert all(k[0] != first for k in client.app.state.manager.cache.keys())


def test_capture_error_is_clean_500_and_not_cached(client):
    FakeProvider.fail_acts = {"gate"}
    rid = start_run(client)["run_id"]
    r = client.get(f"/api/run/{rid}/slice", params={"act": "gate", "layer": 0})
    assert r.status_code == 500 and "gate" in r.json()["detail"] and "boom" in r.json()["detail"]
    # the model thread survived, other acts still work, and the failure is retried (not cached)
    assert client.get(f"/api/run/{rid}/slice", params={"act": "q", "layer": 0}).status_code == 200
    FakeProvider.fail_acts = set()
    assert client.get(f"/api/run/{rid}/slice", params={"act": "gate", "layer": 0}).status_code == 200
    assert provider(client).calls == ["gate", "q", "gate"]


def test_concurrent_waiters_all_get_the_capture_error(make_client):
    c = make_client()
    FakeProvider.delay, FakeProvider.fail_acts = 0.2, {"gate"}
    rid = start_run(c)["run_id"]

    def hit(_):
        return c.get(f"/api/run/{rid}/stats", params={"act": "gate", "layer": 0}).status_code

    with ThreadPoolExecutor(4) as ex:
        codes = list(ex.map(hit, range(4)))
    assert codes == [500] * 4 and provider(c).calls == ["gate"]


# ---------- endpoints ----------

def test_run_then_slice_roundtrip(client):
    run = start_run(client)
    r = client.get(f"/api/run/{run['run_id']}/slice",
                   params={"act": "resid_pre", "layer": 1, "t0": 0, "t1": 4, "d0": 2, "d1": 8})
    assert r.status_code == 200
    meta, arr = decode(r.content)
    assert arr.shape == (4, 6) == (meta["rows"], meta["cols"]) and meta["dims"] == list(range(2, 8))
    assert meta["act"] == "resid_pre" and "site" not in meta


def test_all_endpoints_ok(client):
    rid = start_run(client)["run_id"]
    base = f"/api/run/{rid}"
    for url, params in [
        ("overview", {"act": "gate", "stat": "norm"}),
        ("profile", {"act": "resid_pre", "layer": 0, "order": "absmax"}),
        ("attn", {"layer": 0}),
        ("attn", {"layer": 1, "head": 1, "q0": 2, "q1": 6}),
        ("attn_overview", {"stat": "first_token"}),
    ]:
        r = client.get(f"{base}/{url}", params=params)
        assert r.status_code == 200, (url, r.text)
        decode(r.content)
    assert client.get(f"{base}/stats", params={"act": "resid_post", "layer": 2}).json()["n"] > 0
    assert client.get(f"{base}/attn_stats", params={"layer": 0, "head": 0}).json()["n"] > 0


def axis_stats(c, rid, **params):
    return c.get(f"/api/run/{rid}/axis_stats", params={"act": "gate", "layer": 1, **params})


@pytest.mark.parametrize("axis", ["channel", "token"])
@pytest.mark.parametrize("order", ["natural", "absmax"])
def test_axis_stats_endpoint_matches_direct_computation(client, axis, order):
    rid = start_run(client)["run_id"]
    q = dict(axis=axis, stat="std", t0=1, t1=9, d0=2, d1=30, order=order, top=5, bins=16)
    r = axis_stats(client, rid, **q)
    assert r.status_code == 200, r.text
    js = r.json()
    cap = Capture("gate", provider(client).capture([ord(c) for c in "hello world!"], "gate"))
    ref = slicing.axis_stats(cap, 1, axis, "std", 1, 9, 2, 30, order, False, 5, 16)
    assert js == ref
    assert js["axis"] == axis and js["stat"] == "std" and js["region"] == {"t0": 1, "t1": 9, "d0": 2, "d1": 30}
    assert js["n"] == (28 if axis == "channel" else 8) and len(js["hist"]["counts"]) == 16 and len(js["top"]) == 5
    assert set(js["top"][0]) == {"index", "id", "value"}
    vals = [abs(t["value"]) for t in js["top"]]
    assert vals == sorted(vals, reverse=True)
    assert {"mean", "std", "percentiles", "kurtosis", "hist"} <= set(js)
    if axis == "token":
        assert all(t["index"] == t["id"] and 1 <= t["index"] < 9 for t in js["top"])
    elif order == "natural":
        assert all(t["index"] == t["id"] and 2 <= t["index"] < 30 for t in js["top"])
    else:  # ranks in [2, 30); ids are original channel ids (a permutation)
        assert all(2 <= t["index"] < 30 for t in js["top"]) and any(t["index"] != t["id"] for t in js["top"])


def test_axis_stats_defaults_and_all_stats(client):
    rid = start_run(client)["run_id"]
    for stat in slicing.AXIS_STATS:
        for axis in slicing.AXES:
            r = axis_stats(client, rid, axis=axis, stat=stat)
            assert r.status_code == 200 and len(r.json()["top"]) <= 10
    assert axis_stats(client, rid).json()["axis"] == "channel"


@pytest.mark.parametrize("url,params", [
    ("slice", {"act": "nope", "layer": 0}),
    ("slice", {"act": "resid_pre", "layer": 99}),
    ("slice", {"act": "resid_pre", "layer": -1}),
    ("slice", {"act": "resid_pre", "layer": 0, "agg": "bogus"}),
    ("slice", {"act": "resid_pre", "layer": 0, "order": "bogus"}),
    ("slice", {"act": "q_rope", "layer": 0}),  # valid registry id but not exposed by this model
    ("overview", {"act": "resid_pre", "stat": "bogus"}),
    ("overview", {"act": "nope"}),
    ("attn", {"layer": 99}),
    ("attn_stats", {"layer": 0, "head": 99}),
    ("axis_stats", {"act": "gate", "layer": 0, "axis": "bogus"}),
    ("axis_stats", {"act": "gate", "layer": 0, "stat": "bogus"}),
    ("axis_stats", {"act": "gate", "layer": 0, "order": "bogus"}),
    ("axis_stats", {"act": "gate", "layer": 99}),
    ("axis_stats", {"act": "nope", "layer": 0}),
    ("axis_stats", {"act": "gate", "layer": 0, "t0": 5, "t1": 5}),
    ("axis_stats", {"act": "gate", "layer": 0, "d0": 900, "d1": 999}),
    ("axis_stats", {"act": "gate", "layer": 0, "bins": 0}),
    ("axis_stats", {"act": "gate", "layer": 0, "top": -1}),
])
def test_bad_requests_are_400(client, url, params):
    rid = start_run(client)["run_id"]
    r = client.get(f"/api/run/{rid}/{url}", params=params)
    assert r.status_code == 400, r.text
    assert isinstance(r.json()["detail"], str)


@pytest.mark.parametrize("url,extra", [
    ("slice", {"layer": 0}), ("stats", {"layer": 0}), ("profile", {"layer": 0}), ("overview", {}),
    ("axis_stats", {"layer": 0}),
])
def test_attn_pattern_rejected_on_token_endpoints(client, url, extra):
    rid = start_run(client)["run_id"]
    r = client.get(f"/api/run/{rid}/{url}", params={"act": "attn_pattern", **extra})
    assert r.status_code == 400 and "attn" in r.json()["detail"]
    assert provider(client).calls == []  # rejected before any model work


def test_old_site_param_is_gone(client):
    rid = start_run(client)["run_id"]
    assert client.get(f"/api/run/{rid}/slice", params={"site": "resid", "layer": 0}).status_code == 422


def test_unknown_run_404_and_empty_prompt_400(client):
    assert client.get("/api/run/deadbeef/overview").status_code == 404
    assert client.get("/api/run/deadbeef/axis_stats", params={"act": "gate", "layer": 0}).status_code == 404
    assert client.post("/api/run", json={"text": "   "}).status_code == 400


def test_run_eviction(client):
    ids = [start_run(client, f"prompt {i}")["run_id"] for i in range(5)]
    assert client.get(f"/api/run/{ids[0]}/overview", params={"act": "gate"}).status_code == 404
    assert client.get(f"/api/run/{ids[-1]}/overview", params={"act": "gate"}).status_code == 200


def test_run_before_model_ready_is_409():
    with TestClient(create_app(ModelManager(factory=FakeProvider), autoload=False)) as c:
        assert c.post("/api/run", json={"text": "hi"}).status_code == 409


def test_load_failure_reported_in_status(client):
    client.post("/api/models/load", json={"model_id": "bad/model"})
    for _ in range(100):
        s = client.get("/api/status").json()
        if s["state"] != "loading":
            break
        time.sleep(0.02)
    assert s["state"] == "error" and "not found" in s["error"]


def test_model_reload_invalidates_runs_and_cache(client):
    rid = start_run(client)["run_id"]
    client.get(f"/api/run/{rid}/stats", params={"act": "q", "layer": 0})
    client.post("/api/models/load", json={"model_id": "fake/other"})
    wait_ready(client)
    assert client.get(f"/api/run/{rid}/stats", params={"act": "q", "layer": 0}).status_code == 404
    assert client.app.state.manager.cache.keys() == []


def test_cache_budget_env(monkeypatch):
    monkeypatch.setenv("ACTLENS_CACHE_MB", "3")
    assert ModelManager(factory=FakeProvider).cache.budget == 3 * 2**20
    monkeypatch.delenv("ACTLENS_CACHE_MB")
    assert ModelManager(factory=FakeProvider).cache.budget == 2048 * 2**20


# ----- access token (used when the server is exposed through a tunnel) -----
def test_token_blocks_requests_without_it():
    c = TestClient(create_app(ModelManager(FakeProvider), autoload=False, token="s3cret"), follow_redirects=False)
    assert c.get("/api/status").status_code == 401
    assert c.get("/api/status", headers={"Authorization": "Bearer nope"}).status_code == 401
    assert c.get("/api/status?token=nope").status_code == 401


def test_token_bearer_header_works():
    c = TestClient(create_app(ModelManager(FakeProvider), autoload=False, token="s3cret"))
    assert c.get("/api/status", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_token_query_sets_cookie_and_redirects():
    c = TestClient(create_app(ModelManager(FakeProvider), autoload=False, token="s3cret"), follow_redirects=False)
    r = c.get("/api/status?token=s3cret")
    assert r.status_code == 303 and r.headers["location"] == "/api/status"
    cookie = r.headers["set-cookie"]
    assert "actlens_token=s3cret" in cookie and "HttpOnly" in cookie
    assert "samesite=lax" in cookie.lower()  # Strict would be dropped on the redirect after a cross-site click
    assert "secure" not in cookie.lower()
    fresh = TestClient(create_app(ModelManager(FakeProvider), autoload=False, token="s3cret"), follow_redirects=False)
    r = fresh.get("/api/status?token=s3cret", headers={"x-forwarded-proto": "https"})  # behind a TLS-terminating tunnel
    assert "secure" in r.headers["set-cookie"].lower()
    assert c.get("/api/status").status_code == 200  # the cookie is now sent


def test_no_token_means_open_access():
    c = TestClient(create_app(ModelManager(FakeProvider), autoload=False))
    assert c.get("/api/status").status_code == 200


def test_load_progress_and_error_hint():
    import threading
    from actlens.loading import LoadProgress

    gate = threading.Event()

    def slow_prefetch(model_id, progress):
        progress.set_stage("download", 100)
        progress.add(40)
        gate.wait(5)

    mgr = ModelManager(factory=FakeProvider, prefetch=slow_prefetch)
    with TestClient(create_app(mgr, autoload=False)) as c:
        c.post("/api/models/load", json={"model_id": "fake/model"})
        for _ in range(100):
            s = c.get("/api/status").json()
            if s["progress"] and s["progress"]["stage"] == "download":
                break
            time.sleep(0.02)
        assert s["state"] == "loading" and s["progress"]["done"] == 40 and s["progress"]["total"] == 100
        gate.set()
        wait_ready(c)
        assert c.get("/api/status").json()["progress"] is None

    from actlens.loading import explain
    assert "gated" in explain("a/b", type("GatedRepoError", (OSError,), {})("401"))[0]
    nf = type("RepositoryNotFoundError", (OSError,), {})("401 Repository Not Found ... private or gated repo, make sure you are authenticated")
    assert "not found" in explain("a/b", nf)[0]
    assert "memory" in explain("a/b", RuntimeError("MPS backend out of memory"))[0]
    assert "not found" in explain("a/b", OSError("a/b is not a local folder and is not a valid model identifier"))[0]
    msg, hint = explain("a/b", ValueError("Unsupported architecture for a/b (model_type='x')"))
    assert "architecture" in msg and hint
    assert LoadProgress().snapshot()["stage"] == "idle"
