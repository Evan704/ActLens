#!/usr/bin/env bash
# Build the frontend if needed and serve everything from one process: http://127.0.0.1:8000
set -euo pipefail
cd "$(dirname "$0")"
# Requires the conda env to be activated first: conda activate actlens
if [ ! -d frontend/dist ] || [ -n "$(find frontend/src -newer frontend/dist/index.html -print -quit 2>/dev/null)" ]; then
  (cd frontend && npm run build)
fi
cd backend
exec python -m uvicorn actlens.app:app --host 127.0.0.1 --port "${PORT:-8000}"
