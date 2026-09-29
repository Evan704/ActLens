"""FastAPI app: model management, runs, and windowed activation slices."""
from __future__ import annotations

import asyncio
import gc
import hmac
import json
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import slicing
from .cache import ActivationCache
from .capture import ATTN_ACT, Capture, Run
from .loading import LoadProgress, explain as explain_load_error, prefetch as prefetch_model
from .providers import ActivationProvider, NNsightProvider
from .wire import frame_response

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
PRESET_MODELS = [
    {"id": "Qwen/Qwen3-0.6B", "label": "Qwen3-0.6B"},
    {"id": "Qwen/Qwen2.5-0.5B", "label": "Qwen2.5-0.5B"},
    {"id": "HuggingFaceTB/SmolLM2-360M", "label": "SmolLM2-360M"},
    {"id": "openai-community/gpt2", "label": "GPT-2"},
]
MAX_RUNS = 3
TOKEN_COOKIE = "actlens_token"
DEFAULT_CACHE_MB = 2048
CORPUS = json.loads((Path(__file__).parent / "corpus.json").read_text())
_PACKAGED_DIST = Path(__file__).parent / "static"  # bundled into the wheel
_DEV_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"  # source checkout
FRONTEND_DIST = _PACKAGED_DIST if (_PACKAGED_DIST / "index.html").is_file() else _DEV_DIST


def cache_budget_bytes() -> int:
    try:
        mb = float(os.environ.get("ACTLENS_CACHE_MB", DEFAULT_CACHE_MB))
    except ValueError:
        mb = DEFAULT_CACHE_MB
    return int(mb * 2**20)


class ModelManager:
    """Owns the single loaded model. All model work runs on one dedicated thread."""

    def __init__(self, factory: Callable[[str, str, str], ActivationProvider] = NNsightProvider,
                 cache_bytes: int | None = None, prefetch: Callable[[str, LoadProgress], None] | None = None):
        self.factory = factory
        # Only the real provider loads from the Hub; injected factories (tests, notebooks) opt in explicitly.
        self.prefetch = prefetch if prefetch is not None else (prefetch_model if factory is NNsightProvider else None)
        self.progress = LoadProgress()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model")
        self.provider: ActivationProvider | None = None
        self.state = "idle"  # idle | loading | ready | error
        self.target: str | None = None
        self.error: str | None = None
        self.error_hint: str | None = None
        self.error_detail: str | None = None
        self.runs: OrderedDict[str, Run] = OrderedDict()
        self.cache = ActivationCache(cache_budget_bytes() if cache_bytes is None else cache_bytes)
        self._lock = threading.Lock()

    def status(self) -> dict:
        return {
            "state": self.state,
            "model_id": self.provider.model_id if self.provider else self.target,
            "target": self.target,
            "error": self.error,
            "error_hint": self.error_hint,
            "error_detail": self.error_detail,
            "progress": self.progress.snapshot() if self.state == "loading" else None,
            "info": self.provider.info if self.state == "ready" and self.provider else None,
        }

    def load(self, model_id: str, device: str = "auto", dtype: str = "float32") -> None:
        with self._lock:
            if self.state == "loading":
                raise HTTPException(409, "A model is already loading")
            self.state, self.target = "loading", model_id
            self.error = self.error_hint = self.error_detail = None
            self.progress.start()
        self.executor.submit(self._load, model_id, device, dtype)

    def _load(self, model_id: str, device: str, dtype: str) -> None:
        try:
            if self.provider is not None:
                self.provider.close()
                self.provider = None
            self.runs.clear()
            self.cache.clear()
            gc.collect()
            if self.prefetch is not None:
                self.prefetch(model_id, self.progress)
            self.progress.set_stage("load")
            self.provider = self.factory(model_id, device, dtype)
            self.state = "ready"
        except Exception as e:  # surfaced to the UI via /api/status
            self.error, self.error_hint = explain_load_error(model_id, e)
            self.error_detail = f"{type(e).__name__}: {e}"
            self.state = "error"

    def _register(self, provider: ActivationProvider, text: str, max_tokens: int) -> Run:
        t0 = time.perf_counter()
        ids, tokens, truncated = provider.tokenize(text, max_tokens)
        specs = {s.id: s for s in provider.activations()}
        keys = ("n_layers", "hidden_size", "intermediate_size", "n_heads", "n_kv_heads", "head_dim")
        return Run(run_id=uuid.uuid4().hex[:12], model_id=provider.model_id, text=text, token_ids=list(ids),
                   tokens=list(tokens), truncated=truncated, specs=specs,
                   model={k: provider.info.get(k) for k in keys}, elapsed_ms=(time.perf_counter() - t0) * 1000)

    async def run(self, text: str, max_tokens: int) -> Run:
        """Tokenize and register a run. No activations are captured yet."""
        if self.state != "ready" or self.provider is None:
            raise HTTPException(409, f"Model not ready (state: {self.state})")
        loop = asyncio.get_running_loop()
        try:
            run = await loop.run_in_executor(self.executor, self._register, self.provider, text, max_tokens)
        except ValueError as e:
            raise HTTPException(400, str(e))
        self.runs[run.run_id] = run
        while len(self.runs) > MAX_RUNS:
            old, _ = self.runs.popitem(last=False)
            self.cache.drop_run(old)
        return run

    def get(self, run_id: str) -> Run:
        run = self.runs.get(run_id)
        if run is None:
            raise HTTPException(404, "Unknown run (it may have been evicted); run the prompt again")
        self.runs.move_to_end(run_id)
        return run

    def capture(self, run: Run, act: str) -> Capture:
        """Blocking (call from a worker thread): return the cached capture, building it on the model thread if needed.
        Concurrent callers for the same (run, act) share one forward pass."""
        provider = self.provider
        if provider is None or run.run_id not in self.runs or provider.model_id != run.model_id:
            raise HTTPException(404, "Unknown run (it may have been evicted); run the prompt again")

        def build() -> Capture:
            arr = self.executor.submit(provider.capture, run.token_ids, act).result()
            return Capture(act=act, arr=arr)

        try:
            cap = self.cache.get((run.run_id, act), build)
        except HTTPException:
            raise
        except ValueError as e:
            raise HTTPException(400, str(e))
        except Exception as e:
            detail = f"Capture of {act!r} failed: {type(e).__name__}: {e}"
            if "out of memory" in detail.lower():
                detail = (f"Out of memory while capturing {act!r}. Use a shorter prompt (lower max tokens), a smaller "
                          f"model, or a lower-precision --dtype. ({type(e).__name__})")
            raise HTTPException(500, detail)
        if run.run_id not in self.runs:  # run was evicted while we were capturing
            self.cache.drop_run(run.run_id)
        return cap


class LoadRequest(BaseModel):
    model_id: str
    device: str = "auto"
    dtype: str = "float32"


class RunRequest(BaseModel):
    text: str
    max_tokens: int = 512


def create_app(manager: ModelManager | None = None, autoload: bool = True, model_id: str = DEFAULT_MODEL,
               device: str = "auto", dtype: str = "float32", token: str | None = None) -> FastAPI:
    mgr = manager or ModelManager()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if autoload and mgr.state == "idle":
            mgr.load(model_id, device, dtype)
        yield

    app = FastAPI(title="ActLens", lifespan=lifespan)
    app.state.manager = mgr

    if token:
        # Access token for a server that is reachable from outside (e.g. a tunnel from Colab). Open
        # `/?token=...` once: the token is stored in an HttpOnly cookie and the URL is cleaned up.
        # Scripts can send `Authorization: Bearer <token>` instead.
        def valid(candidate: str | None) -> bool:
            return bool(candidate) and hmac.compare_digest(candidate.encode(), token.encode())

        @app.middleware("http")
        async def require_token(request: Request, call_next):
            bearer = request.headers.get("authorization", "")
            if valid(request.cookies.get(TOKEN_COOKIE)) or valid(bearer.removeprefix("Bearer ").strip()):
                return await call_next(request)
            if request.method == "GET" and valid(request.query_params.get("token")):
                url = request.url.remove_query_params("token")
                resp = RedirectResponse(url.path + (f"?{url.query}" if url.query else ""), status_code=303)
                # Lax, not Strict: the link is usually opened from another site (Colab), and a Strict cookie is
                # not sent on the redirect that follows a cross-site navigation.
                https = "https" in (request.url.scheme, request.headers.get("x-forwarded-proto", ""))
                resp.set_cookie(TOKEN_COOKIE, token, httponly=True, samesite="lax", secure=https)
                return resp
            return PlainTextResponse("ActLens: missing or invalid access token. Open the full link printed "
                                     "by the server (it ends in ?token=...).", status_code=401)

    def guarded(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except ValueError as e:
            raise HTTPException(400, str(e))

    def token_capture(run_id: str, act: str, layer: int | None = None) -> Capture:
        """Validate `act` (a token-kind activation of this run's model) and capture it lazily."""
        run = mgr.get(run_id)
        spec = run.specs.get(act)
        if spec is None:
            raise HTTPException(400, f"unknown act {act!r}; expected one of {list(run.specs)}")
        if spec.kind != "token":
            raise HTTPException(400, f"act {act!r} is an attention pattern; use the /attn endpoints")
        if layer is not None and not 0 <= layer < spec.n_layers:
            raise HTTPException(400, f"layer {layer} out of range for {act} (0..{spec.n_layers - 1})")
        return mgr.capture(run, act)

    def attn_capture(run_id: str) -> Capture:
        run = mgr.get(run_id)
        if ATTN_ACT not in run.specs:
            raise HTTPException(400, "this model does not expose attention patterns")
        return mgr.capture(run, ATTN_ACT)

    # ----- models / corpus -----
    @app.get("/api/status")
    def status():
        return {**mgr.status(), "presets": PRESET_MODELS}

    @app.post("/api/models/load", status_code=202)
    def load_model(req: LoadRequest):
        mgr.load(req.model_id, req.device, req.dtype)
        return mgr.status()

    @app.get("/api/corpus")
    def corpus():
        return CORPUS

    # ----- runs -----
    @app.post("/api/run")
    async def run(req: RunRequest):
        r = await mgr.run(req.text, max(1, min(req.max_tokens, 1024)))
        return {
            "run_id": r.run_id, "model_id": r.model_id, "tokens": r.tokens, "token_ids": r.token_ids,
            "truncated": r.truncated, "elapsed_ms": round(r.elapsed_ms, 1), "model": r.model,
            "activations": [sp.info() for sp in r.specs.values()],
        }

    @app.get("/api/run/{run_id}/overview")
    def overview(run_id: str, act: str = "resid_post", stat: str = "norm", dim: int = 0):
        cap = token_capture(run_id, act)
        arr, meta = guarded(slicing.overview, cap, stat, dim)
        return frame_response(meta, arr)

    @app.get("/api/run/{run_id}/slice")
    def slice_(run_id: str, act: str, layer: int, t0: int = 0, t1: int = 64, d0: int = 0, d1: int = 128,
               max_h: int = Query(512, ge=1, le=2048), max_w: int = Query(1024, ge=1, le=4096),
               agg: str = "absmax", order: str = "natural"):
        cap = token_capture(run_id, act, layer)
        arr, meta = guarded(slicing.token_dim_slice, cap, layer, t0, t1, d0, d1, max_h, max_w, agg, order)
        return frame_response(meta, arr)

    @app.get("/api/run/{run_id}/profile")
    def profile(run_id: str, act: str, layer: int, order: str = "natural", bins: int = Query(512, ge=8, le=4096)):
        cap = token_capture(run_id, act, layer)
        arr, meta = guarded(slicing.dim_profile, cap, layer, order, bins)
        return frame_response(meta, arr)

    @app.get("/api/run/{run_id}/stats")
    def stats(run_id: str, act: str, layer: int, t0: int = 0, t1: int = 64, d0: int = 0, d1: int = 128,
              order: str = "natural", clip: bool = False):
        cap = token_capture(run_id, act, layer)
        return guarded(slicing.token_region_stats, cap, layer, t0, t1, d0, d1, order, clip)

    @app.get("/api/run/{run_id}/axis_stats")
    def axis_stats(run_id: str, act: str, layer: int, axis: str = "channel", stat: str = "norm",
                   t0: int = 0, t1: int = 64, d0: int = 0, d1: int = 128, order: str = "natural",
                   clip: bool = False, top: int = 10, bins: int = 64):
        cap = token_capture(run_id, act, layer)
        return guarded(slicing.axis_stats, cap, layer, axis, stat, t0, t1, d0, d1, order, clip, top, bins)

    @app.get("/api/run/{run_id}/attn")
    def attn(run_id: str, layer: int, head: int = -1, q0: int = 0, q1: int = 100000, k0: int = 0, k1: int = 100000,
             max_q: int = Query(512, ge=1, le=1024), max_k: int = Query(512, ge=1, le=1024), agg: str = "max"):
        cap = attn_capture(run_id)
        if not 0 <= layer < cap.n_layers:
            raise HTTPException(400, f"layer {layer} out of range (0..{cap.n_layers - 1})")
        arr, meta = guarded(slicing.attn_slice, cap, layer, head, q0, q1, k0, k1, max_q, max_k, agg)
        return frame_response(meta, arr)

    @app.get("/api/run/{run_id}/attn_overview")
    def attn_overview(run_id: str, stat: str = "entropy"):
        cap = attn_capture(run_id)
        arr, meta = guarded(slicing.attn_overview, cap, stat)
        return frame_response(meta, arr)

    @app.get("/api/run/{run_id}/attn_stats")
    def attn_stats(run_id: str, layer: int, head: int, q0: int = 0, q1: int = 100000, k0: int = 0, k1: int = 100000,
                   clip: bool = False):
        cap = attn_capture(run_id)
        if not (0 <= layer < cap.n_layers and 0 <= head < cap.n_heads):
            raise HTTPException(400, "layer/head out of range")
        return guarded(slicing.attn_region_stats, cap, layer, head, q0, q1, k0, k1, clip)

    # ----- static frontend (production build) -----
    if (FRONTEND_DIST / "index.html").is_file():
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{path:path}")
        def spa(path: str):
            if path.startswith("api/"):
                raise HTTPException(404)
            f = FRONTEND_DIST / path
            return FileResponse(f if path and f.is_file() else FRONTEND_DIST / "index.html")

    return app


app = create_app()
