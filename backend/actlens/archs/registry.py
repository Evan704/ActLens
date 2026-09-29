"""Adapter registry and lookup.

Resolution order: exact `config.model_type` match, then any adapter whose `probe` finds no missing modules
(so a custom model with a known module layout works unlisted). Third-party adapters are picked up from the
`actlens.archs` entry-point group and from modules named in `$ACTLENS_ARCH_MODULES` (comma separated dotted
module names or .py paths, also settable with `actlens --arch-module`); either just needs to `@register` a class.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from importlib.metadata import entry_points
from pathlib import Path

from .base import ArchAdapter

ADAPTERS: list[type[ArchAdapter]] = []
_plugins_loaded = False


def register(cls: type[ArchAdapter]) -> type[ArchAdapter]:
    """Class decorator. Later registrations take precedence over earlier ones with the same `name`."""
    ADAPTERS[:] = [a for a in ADAPTERS if a.name != cls.name]
    ADAPTERS.append(cls)
    return cls


def _import_module(spec: str) -> None:
    if spec.endswith(".py") or os.sep in spec:
        path = Path(spec).expanduser().resolve()
        mod = importlib.util.spec_from_file_location(f"actlens_arch_{path.stem}", path)
        if mod is None or mod.loader is None:
            raise ImportError(f"cannot load architecture module {spec}")
        module = importlib.util.module_from_spec(mod)
        sys.modules[mod.name] = module
        mod.loader.exec_module(module)
    else:
        importlib.import_module(spec)


def load_plugins() -> None:
    """Import entry-point and $ACTLENS_ARCH_MODULES adapters once (idempotent)."""
    global _plugins_loaded
    if _plugins_loaded:
        return
    _plugins_loaded = True
    for ep in entry_points(group="actlens.archs"):
        ep.load()
    for spec in filter(None, (s.strip() for s in os.environ.get("ACTLENS_ARCH_MODULES", "").split(","))):
        _import_module(spec)


def supported_model_types() -> list[str]:
    return sorted({t for a in ADAPTERS for t in a.model_types})


# model types that look close to a supported one but are deliberately not: why, and what to use instead
KNOWN_UNSUPPORTED = {
    "gemma3": "this is the multimodal Gemma-3 (blocks live under model.language_model); load a text-only Gemma-3 "
              "checkpoint instead (model_type gemma3_text, e.g. gemma-3-270m or gemma-3-1b)",
}


def resolve_adapter(root, model_id: str = "") -> ArchAdapter:
    """The adapter for a loaded HF model, or a ValueError explaining what is missing."""
    load_plugins()
    model_type = getattr(getattr(root, "config", None), "model_type", None)
    for cls in reversed(ADAPTERS):  # newest registration first, so plugins can override built-ins
        if model_type in cls.model_types and not cls().probe(root):
            return cls()
    misses = []
    for cls in reversed(ADAPTERS):
        adapter = cls()
        missing = adapter.probe(root)
        if not missing:
            return adapter
        misses.append((len(missing), adapter.name, missing))
    _, best, missing = min(misses, key=lambda m: m[0])
    if model_type in KNOWN_UNSUPPORTED:
        raise ValueError(f"Unsupported architecture for {model_id or 'this model'} (model_type={model_type!r}): "
                         f"{KNOWN_UNSUPPORTED[model_type]}.")
    raise ValueError(
        f"Unsupported architecture for {model_id or 'this model'} (model_type={model_type!r}). Supported model types: "
        f"{', '.join(supported_model_types())}. Closest adapter {best!r} is missing: {', '.join(missing)}. "
        "To add an architecture, subclass actlens.archs.ArchAdapter and @register it "
        "(see actlens/archs/gpt2.py; load it with --arch-module).")
