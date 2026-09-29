# Contributing to ActLens

## Development setup

The dev environment is a conda env named `actlens` (Python 3.12 + Node 22); activate it before running anything below.

```bash
conda create -y -n actlens -c conda-forge python=3.12 nodejs=22
conda activate actlens
pip install -e ".[dev]"             # editable install: `actlens` command + pytest, httpx, build
(cd frontend && npm install)
./run.sh                            # dev server with hot reload: open http://127.0.0.1:5173
```

`./run.sh` starts Vite (HMR, port 5173, `/api` proxied to the backend) and uvicorn with `--reload` on :8000. Frontend
edits apply instantly; a backend `.py` edit restarts the server, which reloads the model. `./run.sh --prod` builds the
frontend if needed and serves UI + API from :8000 without reload (what `actlens` does in a source checkout).

Tests: `cd backend && python -m pytest`, `cd frontend && npx vitest run`.
Browser e2e (needs `playwright-core` and Chrome; see the header of each script): `frontend/e2e/real.e2e.mjs` drives the real
backend (every activation, head jump, distribution modes, exports), `ui.e2e.mjs` and `distribution.mjs` run against mocks.

## Releasing

The wheel bundles the built UI as `actlens/static` (`hatch_build.py` copies `frontend/dist`; the build fails if it is
missing). The version lives in `backend/actlens/__init__.py`.

```bash
(cd frontend && npm ci && npm run build)
python -m build                     # dist/actlens-<version>.tar.gz and .whl
twine upload dist/*                 # or: twine upload --repository testpypi dist/*
```


## Adding an architecture

An adapter (`backend/actlens/archs/`) tells ActLens which activations a model family has and how to read each one from a
block; everything else (tracing, caching, API, UI) is architecture-neutral.

1. Subclass `ArchAdapter` (or `PreNormBlockAdapter` for sequential pre-norm blocks, which already covers the residual
   stream, norms, attention pattern/context/output and MLP output) and `@register` it. Set `layers_path`, `model_types`,
   and implement `probe` (which modules must exist), `dims` and `acts`. `acts` returns `{activation id: ActDef}`:
   leave out what the model lacks and it simply does not appear in the picker. `archs/gpt2.py` is a complete small example
   (fused QKV, no RoPE, plain MLP); `archs/llama.py` shows QK-norm and RoPE (`setup` runs once before the trace and computes the RoPE tables).
2. Add a tiny random model to `tests/tiny_models.py::TINY`. `tests/test_archs.py` then checks residual sums,
   `attn_pattern == softmax(q k^T)`, `attn_ctx == pattern @ v`, shapes and specs for it, with no download.
3. Ship it in your own package with the `actlens.archs` entry-point group, or try it without packaging:
   `actlens --arch-module ./my_arch.py` (also `$ACTLENS_ARCH_MODULES`).


## Implementation notes

- Model loading: `device_map="mps"` hangs with transformers 5.x, so the model is loaded on CPU and then moved.
- Attention capture needs `attn_implementation="eager"` (SDPA doesn't return probabilities).
- Activations (`backend/actlens/capture.py` registry): residual stream
  (`resid_pre/mid/post`), attention (`attn_norm, q, k, v, q_norm, k_norm, q_rope, k_rope, attn_pattern, attn_ctx, o`),
  MLP (`mlp_norm, gate, up, silu, swiglu, mlp_act, down`). `/api/run` lists only what the loaded model's adapter exposes
  (Llama-style: `q_norm`/`k_norm` need QK-norm modules and `q_rope`/`k_rope` need a `rotary_emb` module; RoPE is applied
  from `rotary_emb`'s cos/sin, since `apply_rotary_pos_emb` is a function, not a hookable module).
- Lazy capture: `POST /api/run` only tokenizes. The first data request for an activation runs one forward pass on the
  model thread that captures exactly that activation for all layers (~60-400 ms for Qwen3-0.6B at 64 tokens); results
  are kept in a byte-budgeted LRU keyed by `(run_id, act)` (`ACTLENS_CACHE_MB`, default 2048; the entry just built is never
  evicted). Concurrent requests for the same activation share one capture. Only the last 3 runs are registered.
- `GET /api/run/{id}/axis_stats` gives the distribution of a per-channel or per-token statistic over a region;
  the maths is in `slicing.py` (`axis_stats`, `axis_stat_values`).
- Tests: `cd backend && python -m pytest` runs fast fake-provider tests, the per-architecture contract tests on tiny
  random models (`tests/test_archs.py`), and the real-model tests `tests/test_real_model.py` (Qwen3-0.6B identities incl.
  recomputing attention probabilities from `q_rope`/`k_rope`) and `tests/test_real_gpt2.py` (~10 s). Real-model modules
  are skipped when the checkpoint is not in the local HF cache or `ACTLENS_SKIP_REAL=1`.

