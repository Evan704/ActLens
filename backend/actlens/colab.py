"""Run ActLens on a Colab GPU and view it in your own browser.

    from actlens.colab import launch
    launch()   # starts the server + a Cloudflare quick tunnel, prints a private link

The server needs the Colab VM, but the UI is opened from your local browser through the tunnel. The link carries a
random access token (see `--token`); anyone who has the full link can use the server, so do not share it.
"""
from __future__ import annotations

import os
import platform
import re
import secrets
import stat
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CLOUDFLARED_URL = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-{arch}"
TUNNEL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")

_procs: list[subprocess.Popen] = []


def find_tunnel_url(text: str) -> str | None:
    m = TUNNEL_RE.search(text)
    return m.group(0) if m else None


def _cloudflared() -> str:
    path = Path.home() / ".cache" / "actlens" / "cloudflared"
    if not path.is_file():
        arch = "arm64" if platform.machine() in ("aarch64", "arm64") else "amd64"
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CLOUDFLARED_URL.format(arch=arch), path)
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _wait_ready(base: str, token: str, proc: subprocess.Popen, timeout: float) -> None:
    req = urllib.request.Request(f"{base}/api/status", headers={"Authorization": f"Bearer {token}"})
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"actlens server exited early (code {proc.returncode}); see actlens.log")
        try:
            urllib.request.urlopen(req, timeout=2).close()
            return
        except OSError:
            time.sleep(1)
    raise TimeoutError("actlens server did not start; see actlens.log")


def stop() -> None:
    """Stop the server and the tunnel."""
    while _procs:
        p = _procs.pop()
        p.terminate()


def launch(model: str = "Qwen/Qwen3-0.6B", *, dtype: str = "float16", device: str = "auto", port: int = 8000,
           log: str = "actlens.log", timeout: float = 180) -> str:
    """Start ActLens and a tunnel; return (and display) the private URL to open in your browser."""
    stop()
    token = secrets.token_urlsafe(24)
    server = subprocess.Popen(
        [sys.executable, "-m", "actlens.cli", "-m", model, "--device", device, "--dtype", dtype,
         "--port", str(port), "--token", token],
        stdout=open(log, "w"), stderr=subprocess.STDOUT)
    _procs.append(server)
    _wait_ready(f"http://127.0.0.1:{port}", token, server, timeout)

    tunnel = subprocess.Popen([_cloudflared(), "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    _procs.append(tunnel)
    base, seen, deadline = None, "", time.time() + 60
    while base is None and time.time() < deadline:
        line = tunnel.stdout.readline()
        if not line and tunnel.poll() is not None:
            break
        seen += line
        base = find_tunnel_url(line)
    if base is None:
        stop()
        raise RuntimeError("could not open a tunnel:\n" + seen[-1000:])
    # drain the tunnel's output so its pipe never fills up
    import threading
    threading.Thread(target=lambda: [None for _ in tunnel.stdout], daemon=True).start()

    url = f"{base}/?token={token}"
    print("ActLens is running. Open this private link in your browser (keep this Colab tab open):\n" + url)
    try:
        from IPython.display import HTML, display
        display(HTML(f'<a href="{url}" target="_blank" style="font-size:1.2em">Open ActLens ↗</a>'))
    except ImportError:
        pass
    return url
