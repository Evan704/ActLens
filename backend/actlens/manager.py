"""The single loaded model, the runs registered against it, and lazy activation capture."""
from __future__ import annotations

import asyncio
import gc
import os
import threading
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable

from .cache import ActivationCache
from .capture import Capture, Run
from .errors import ActLensError, BadRequest, Conflict, NotFound
from .loading import LoadProgress, explain as explain_load_error, prefetch as prefetch_model
from .providers import ActivationProvider, NNsightProvider

MAX_RUNS = 3
DEFAULT_CACHE_MB = 2048


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
        self.prefetch = prefetch if prefetch is not None else (prefetch_model if getattr(factory, "hub_backed", False) else None)
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
                raise Conflict("A model is already loading")
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
            raise Conflict(f"Model not ready (state: {self.state})")
        loop = asyncio.get_running_loop()
        try:
            run = await loop.run_in_executor(self.executor, self._register, self.provider, text, max_tokens)
        except ValueError as e:
            raise BadRequest(str(e))
        self.runs[run.run_id] = run
        while len(self.runs) > MAX_RUNS:
            old, _ = self.runs.popitem(last=False)
            self.cache.drop_run(old)
        return run

    def get(self, run_id: str) -> Run:
        run = self.runs.get(run_id)
        if run is None:
            raise NotFound("Unknown run (it may have been evicted); run the prompt again")
        self.runs.move_to_end(run_id)
        return run

    def capture(self, run: Run, act: str) -> Capture:
        """Blocking (call from a worker thread): return the cached capture, building it on the model thread if needed.
        Concurrent callers for the same (run, act) share one forward pass."""
        provider = self.provider
        if provider is None or run.run_id not in self.runs or provider.model_id != run.model_id:
            raise NotFound("Unknown run (it may have been evicted); run the prompt again")

        def build() -> Capture:
            arr = self.executor.submit(provider.capture, run.token_ids, act).result()
            return Capture(act=act, arr=arr)

        try:
            cap = self.cache.get((run.run_id, act), build)
        except ActLensError:
            raise
        except ValueError as e:
            raise BadRequest(str(e))
        except Exception as e:
            detail = f"Capture of {act!r} failed: {type(e).__name__}: {e}"
            if "out of memory" in detail.lower():
                detail = (f"Out of memory while capturing {act!r}. Use a shorter prompt (lower max tokens), a smaller "
                          f"model, or a lower-precision --dtype. ({type(e).__name__})")
            raise ActLensError(detail)
        if run.run_id not in self.runs:  # run was evicted while we were capturing
            self.cache.drop_run(run.run_id)
        return cap

    def logit_lens(self, run: Run, cap: Capture, pos: int, k: int, target: int | None) -> dict:
        """Blocking: unembed one token position of a residual capture at every layer, on the model thread."""
        provider = self.provider
        if provider is None or run.run_id not in self.runs or provider.model_id != run.model_id:
            raise NotFound("Unknown run (it may have been evicted); run the prompt again")
        resid = cap.arr[:, pos, :]
        try:
            return self.executor.submit(provider.logit_lens, resid, k, target).result()
        except ValueError as e:
            raise BadRequest(str(e))
