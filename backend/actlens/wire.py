"""Binary framing: [u32 json_len][json][pad to 4 bytes][float32 little-endian payload]."""
from __future__ import annotations

import json
import struct

import numpy as np
from fastapi import Response


def frame(meta: dict, arr: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(arr, dtype="<f4")
    meta = {**meta, "shape": list(arr.shape)}
    body = json.dumps(meta, separators=(",", ":")).encode()
    pad = (-(4 + len(body))) % 4
    return struct.pack("<I", len(body)) + body + b"\0" * pad + arr.tobytes()


def frame_response(meta: dict, arr: np.ndarray) -> Response:
    return Response(frame(meta, arr), media_type="application/octet-stream")
