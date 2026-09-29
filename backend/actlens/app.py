"""FastAPI app: model management, runs, and windowed activation slices."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import slicing
from .reducers import capabilities
from .auth import TOKEN_COOKIE, install_token_auth  # noqa: F401  (TOKEN_COOKIE is re-exported)
from .capture import Capture
from .errors import ActLensError, BadRequest
from .manager import DEFAULT_CACHE_MB, MAX_RUNS, ModelManager, cache_budget_bytes  # noqa: F401  (re-exported)
from .wire import frame_response

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
PRESET_MODELS = [
    {"id": "Qwen/Qwen3-0.6B", "label": "Qwen3-0.6B"},
    {"id": "Qwen/Qwen2.5-0.5B", "label": "Qwen2.5-0.5B"},
    {"id": "HuggingFaceTB/SmolLM2-360M", "label": "SmolLM2-360M"},
    {"id": "openai-community/gpt2", "label": "GPT-2"},
]
CORPUS = json.loads((Path(__file__).parent / "corpus.json").read_text())
_PACKAGED_DIST = Path(__file__).parent / "static"  # bundled into the wheel
_DEV_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"  # source checkout
FRONTEND_DIST = _PACKAGED_DIST if (_PACKAGED_DIST / "index.html").is_file() else _DEV_DIST


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
        install_token_auth(app, token)

    @app.exception_handler(ActLensError)
    async def domain_error(_: Request, e: ActLensError):
        return JSONResponse({"detail": e.detail}, status_code=e.status_code)

    def guarded(fn, *a, **kw):
        try:
            return fn(*a, **kw)
        except ValueError as e:
            raise BadRequest(str(e))

    def token_capture(run_id: str, act: str, layer: int | None = None) -> Capture:
        """Validate `act` (a token-kind activation of this run's model) and capture it lazily."""
        run = mgr.get(run_id)
        spec = run.specs.get(act)
        if spec is None:
            raise BadRequest(f"unknown act {act!r}; expected one of {list(run.specs)}")
        if spec.kind != "token":
            raise BadRequest(f"act {act!r} is an attention pattern; use the /attn endpoints")
        if layer is not None and not 0 <= layer < spec.n_layers:
            raise BadRequest(f"layer {layer} out of range for {act} (0..{spec.n_layers - 1})")
        return mgr.capture(run, act)

    def attn_capture(run_id: str) -> Capture:
        run = mgr.get(run_id)
        spec = run.spec_of_kind("attn")
        if spec is None:
            raise BadRequest("this model does not expose attention patterns")
        return mgr.capture(run, spec.id)

    # ----- models / corpus -----
    @app.get("/api/status")
    def status():
        return {**mgr.status(), "presets": PRESET_MODELS}

    @app.post("/api/models/load", status_code=202)
    def load_model(req: LoadRequest):
        mgr.load(req.model_id, req.device, req.dtype)
        return mgr.status()

    @app.get("/api/meta")
    def meta():
        """The statistics, pooling modes and channel orders the slice endpoints accept, with their UI text."""
        return capabilities()

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
            raise BadRequest(f"layer {layer} out of range (0..{cap.n_layers - 1})")
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
            raise BadRequest("layer/head out of range")
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
