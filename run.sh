#!/usr/bin/env bash
# Dev server with hot reload:
#   frontend  Vite on http://127.0.0.1:5173 (open this one; HMR, /api is proxied to the backend)
#   backend   uvicorn --reload on http://127.0.0.1:8000 (restarts on .py changes, which reloads the model)
# `./run.sh --prod` builds the frontend if needed and serves everything from one process on :8000, without reload.
# Requires the conda env to be activated first: conda activate actlens
set -euo pipefail
cd "$(dirname "$0")"
API_PORT="${PORT:-8000}"

if [ "${1:-}" = "--prod" ]; then
  if [ ! -d frontend/dist ] || [ -n "$(find frontend/src -newer frontend/dist/index.html -print -quit 2>/dev/null)" ]; then
    (cd frontend && npm run build)
  fi
  cd backend
  exec python -m uvicorn actlens.app:app --host 127.0.0.1 --port "$API_PORT"
fi

[ -d frontend/node_modules ] || (cd frontend && npm install)
(cd frontend && PORT="${VITE_PORT:-5173}" ACTLENS_API="http://127.0.0.1:$API_PORT" exec npx vite --host 127.0.0.1) &
VITE_PID=$!
trap 'kill "$VITE_PID" 2>/dev/null || true' EXIT INT TERM
cd backend
python -m uvicorn actlens.app:app --host 127.0.0.1 --port "$API_PORT" --reload --reload-dir actlens
