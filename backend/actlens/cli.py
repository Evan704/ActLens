"""`actlens` command: load a model and serve the UI + API from one process."""
from __future__ import annotations

import argparse
import os
import sys
import threading
import webbrowser

from . import __version__

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="actlens", description="Interactive viewer for the activations of a Hugging Face model.")
    p.add_argument("-m", "--model", default=None, help="HF model id or local path to load on startup "
                   "(default: Qwen/Qwen3-0.6B). More models can be loaded from the UI.")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps (default: auto)")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                   help="weight dtype (default: float32)")
    p.add_argument("--host", default="127.0.0.1", help="bind address (default: 127.0.0.1)")
    p.add_argument("-p", "--port", type=int, default=int(os.environ.get("PORT", 8000)),
                   help="port (default: 8000, or $PORT)")
    p.add_argument("--cache-mb", type=float, default=None,
                   help="activation cache budget in MB (default: $ACTLENS_CACHE_MB or 2048)")
    p.add_argument("--open", action="store_true", help="open the UI in a browser once the server is up")
    p.add_argument("--version", action="version", version=f"actlens {__version__}")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.cache_mb is not None:
        os.environ["ACTLENS_CACHE_MB"] = str(args.cache_mb)

    # Heavy imports (torch, transformers) only after argument parsing so --help/--version stay instant.
    import uvicorn

    from . import app as app_module

    if not (app_module.FRONTEND_DIST / "index.html").is_file():
        print("actlens: no built frontend found; serving the API only. Run `npm run build` in frontend/ "
              "(source checkout) or install a release wheel.", file=sys.stderr)
    if args.host not in LOOPBACK:
        print(f"actlens: WARNING: binding to {args.host}. The API can load any model and has no authentication; "
              "only expose it on a network you trust.", file=sys.stderr)

    app = app_module.create_app(model_id=args.model or app_module.DEFAULT_MODEL,
                                device=args.device, dtype=args.dtype)
    if args.open:
        url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
