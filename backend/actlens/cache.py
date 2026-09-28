"""Byte-budgeted LRU of captured activations with single-flight builds."""
from __future__ import annotations

import threading
from collections import OrderedDict
from concurrent.futures import Future
from typing import Callable, Hashable

from .capture import Capture


class ActivationCache:
    """Maps (run_id, act) -> Capture.

    - Least-recently-used entries are evicted while the total size exceeds `budget_bytes`; the entry that was
      just built is never evicted (so a single oversized entry still works, it just lives alone).
    - `get` is single-flight: concurrent requests for a key that is being built wait for the one build instead
      of starting another. A failed build propagates its exception to every waiter and is not cached.
    """

    def __init__(self, budget_bytes: int):
        self.budget = int(budget_bytes)
        self._entries: OrderedDict[Hashable, Capture] = OrderedDict()
        self._inflight: dict[Hashable, Future] = {}
        self._lock = threading.Lock()
        self.builds = 0  # number of completed builds (for tests / diagnostics)

    @property
    def nbytes(self) -> int:
        with self._lock:
            return sum(c.nbytes() for c in self._entries.values())

    def keys(self) -> list:
        with self._lock:
            return list(self._entries)

    def get(self, key: Hashable, build: Callable[[], Capture]) -> Capture:
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None:
                self._entries.move_to_end(key)
                return hit
            fut = self._inflight.get(key)
            owner = fut is None
            if owner:
                fut = self._inflight[key] = Future()
        if not owner:
            return fut.result()
        try:
            cap = build()
        except BaseException as e:
            with self._lock:
                self._inflight.pop(key, None)
            fut.set_exception(e)
            raise
        with self._lock:
            self._entries[key] = cap
            self._entries.move_to_end(key)
            self._evict(keep=key)
            self._inflight.pop(key, None)
            self.builds += 1
        fut.set_result(cap)
        return cap

    def _evict(self, keep: Hashable) -> None:
        total = sum(c.nbytes() for c in self._entries.values())
        for k in list(self._entries):
            if total <= self.budget:
                break
            if k == keep:
                continue
            total -= self._entries.pop(k).nbytes()

    def drop_run(self, run_id: str) -> None:
        with self._lock:
            for k in [k for k in self._entries if k[0] == run_id]:
                del self._entries[k]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
