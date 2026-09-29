"""Model loading helpers: hub prefetch with byte-level progress, and readable load errors."""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

CONTRIBUTING_ARCH = "https://github.com/Evan704/ActLens/blob/main/CONTRIBUTING.md#adding-an-architecture"


class LoadProgress:
    """Thread-safe progress of the model load in flight; read by /api/status."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.stage = "idle"  # resolve | download | load
        self.done = 0  # bytes written to disk
        self.transfer = 0  # bytes received from the network (Xet); runs ahead of `done`
        self.total = 0
        self.started = time.monotonic()

    def start(self) -> None:
        with self._lock:
            self.stage, self.done, self.transfer, self.total, self.started = "resolve", 0, 0, 0, time.monotonic()

    def set_stage(self, stage: str, total: int = 0) -> None:
        with self._lock:
            self.stage, self.done, self.transfer, self.total = stage, 0, 0, total

    def add(self, n: int, transfer: bool = False) -> None:
        with self._lock:
            if transfer:
                self.transfer += n
            else:
                self.done += n

    def snapshot(self) -> dict:
        with self._lock:
            done = max(self.done, self.transfer)
            return {"stage": self.stage, "done": min(done, self.total) if self.total else done, "total": self.total or None,
                    "elapsed_s": round(time.monotonic() - self.started, 1)}


def _bar_class(progress: LoadProgress):
    from tqdm.auto import tqdm

    class ByteBar(tqdm):
        """Feeds the bytes of the per-file bars huggingface_hub creates into `progress`; the "Fetching N files"
        bar counts files, not bytes, and is left alone. Xet downloads add a network-bytes bar ("Downloading bytes")
        that moves continuously while the file-bytes bar only advances as chunks are flushed to disk, so both are
        tracked and the larger one is reported."""

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            desc = str(kw.get("desc", ""))
            self._bytes = kw.get("unit") == "B" and not desc.startswith("Download complete")
            self._transfer = desc.startswith("Downloading bytes")
            if self._bytes and self.n:
                progress.add(int(self.n), self._transfer)

        def update(self, n=1):
            if self._bytes and n:
                progress.add(int(n), self._transfer)
            return super().update(n)

    return ByteBar


def prefetch(model_id: str, progress: LoadProgress) -> None:
    """Download a Hub model's weights ahead of `from_pretrained` so the UI can show real byte progress.

    Local paths and offline mode are skipped. Only errors that mean "this model is not usable" (unknown id, gated,
    invalid id) propagate; network hiccups are left to `from_pretrained`, which can still use a cached copy."""
    if Path(model_id).exists() or os.environ.get("HF_HUB_OFFLINE") == "1":
        return
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

    try:
        siblings = HfApi().model_info(model_id).siblings or []
    except (RepositoryNotFoundError, HFValidationError):
        raise
    except Exception:
        return
    names = [s.rfilename for s in siblings]
    patterns = ["*.json", "*.txt", "*.model", "*.tiktoken"]
    patterns += ["*.safetensors"] if any(n.endswith(".safetensors") for n in names) else ["*.bin"]
    try:
        plan = snapshot_download(model_id, allow_patterns=patterns, dry_run=True)
        total = sum(f.file_size for f in plan if f.will_download)
        if not total:
            return  # everything is cached
        progress.set_stage("download", total)
        snapshot_download(model_id, allow_patterns=patterns, tqdm_class=_bar_class(progress))
    except Exception:
        return


def explain(model_id: str, exc: BaseException) -> tuple[str, str | None]:
    """(headline, hint) for a failed model load. The raw exception text is kept separately for debugging."""
    text = f"{type(exc).__name__}: {exc}"
    low = text.lower()
    names = {c.__name__ for c in type(exc).__mro__}

    # The Hub's "not found" text also says "private or gated repo", so only match the gated-specific phrasing.
    if "GatedRepoError" in names or "you are trying to access a gated repo" in low or "access to model" in low and "restricted" in low:
        return (f"{model_id} is a gated model and needs access.",
                "Accept its license on huggingface.co, then run `huggingface-cli login` (or set HF_TOKEN) and load it again.")
    if names & {"RepositoryNotFoundError", "HFValidationError"} or "not a local folder" in low or "not a valid model identifier" in low:
        return (f"Model {model_id!r} was not found.",
                "Use a Hugging Face model id like `owner/name` or a local folder that contains config.json. "
                "Private or gated models need HF_TOKEN.")
    if "OutOfMemoryError" in names or "out of memory" in low or "invalid buffer size" in low or isinstance(exc, MemoryError):
        return (f"Not enough memory to load {model_id}.",
                "Try a smaller model or `--dtype float16` / `bfloat16`, or `--device cpu` if the GPU is small.")
    if "unsupported architecture" in low:
        return (f"{model_id} uses an architecture ActLens does not support yet.",
                f"See the list of supported model types in the message below, or add an adapter: {CONTRIBUTING_ARCH}")
    if "trust_remote_code" in low:
        return (f"{model_id} needs custom code (trust_remote_code), which ActLens does not run.",
                "Pick a model with native transformers support.")
    if names & {"ConnectionError", "Timeout", "ConnectTimeout", "ReadTimeout", "ConnectError", "TimeoutError", "LocalEntryNotFoundError"} \
            or any(s in low for s in ("connection", "timed out", "max retries", "offline", "outgoing traffic")):
        return (f"Could not reach huggingface.co to download {model_id}.",
                "Check your network or proxy. Once a model has been downloaded once, it also loads offline "
                "with HF_HUB_OFFLINE=1.")
    first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    return (f"Failed to load {model_id}: {first[:200]}", None)
