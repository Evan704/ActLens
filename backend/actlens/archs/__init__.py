"""Architecture adapters: how each model family exposes its activations. See `base.ArchAdapter`."""
from . import llama  # noqa: F401  (importing registers the built-in adapters)
from .base import ActDef, ArchAdapter, Dims, PreNormBlockAdapter, TraceCtx, get, has
from .registry import ADAPTERS, load_plugins, register, resolve_adapter, supported_model_types

__all__ = ["ActDef", "ArchAdapter", "Dims", "PreNormBlockAdapter", "TraceCtx", "ADAPTERS", "get", "has",
           "load_plugins", "register", "resolve_adapter", "supported_model_types"]
