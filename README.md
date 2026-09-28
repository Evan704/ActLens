# ActLens

Interactive viewer for the activations of a Hugging Face model during a forward pass.
Python backend (FastAPI + nnsight) captures activations; a React + TypeScript frontend browses them as heatmaps.

## Install

```bash
pip install actlens          # bundles the web UI; needs Python >= 3.10
actlens                      # loads Qwen/Qwen3-0.6B and serves UI + API on http://127.0.0.1:8000
```

The first run downloads the model weights from the Hugging Face Hub. Only the versions the project is developed
against are tested (torch 2, transformers 5, nnsight 0.7); use a fresh virtual environment.

```bash
actlens -m Qwen/Qwen2.5-0.5B            # another HF model id or a local path
actlens --device cuda --dtype bfloat16  # device: auto | cpu | cuda | mps; dtype: float32 | float16 | bfloat16
actlens --port 9000 --open              # custom port, open the browser when the server is up
actlens --cache-mb 4096                 # activation cache budget (also $ACTLENS_CACHE_MB)
```

More models can be loaded from the UI at any time. The server binds to `127.0.0.1` by default: the API can load any
model and has no authentication, so `--host 0.0.0.0` prints a warning and should only be used on a trusted network.

## Development

The dev environment is a conda env named `actlens` (Python 3.12 + Node 22); activate it before running anything below.

```bash
conda create -y -n actlens -c conda-forge python=3.12 nodejs=22
conda activate actlens
pip install -e ".[dev]"             # editable install: `actlens` command + pytest, httpx, build
(cd frontend && npm install)
./run.sh                            # builds the frontend if needed, then serves UI + API (same as `actlens`)
```

Hot reload: run `python -m uvicorn actlens.app:app --port 8000` in `backend/` and `npx vite` in `frontend/`
(proxies `/api` to :8000, UI on :5173). In a source checkout the server serves `frontend/dist` directly.

Tests: `cd backend && python -m pytest`, `cd frontend && npx vitest run`.
Browser e2e (needs `playwright-core` and Chrome; see the header of each script): `frontend/e2e/real.e2e.mjs` drives the real
backend (every activation, head jump, distribution modes, exports), `ui.e2e.mjs` and `distribution.mjs` run against mocks.

### Releasing

The wheel bundles the built UI as `actlens/static` (`hatch_build.py` copies `frontend/dist`; the build fails if it is
missing). The version lives in `backend/actlens/__init__.py`.

```bash
(cd frontend && npm ci && npm run build)
python -m build                     # dist/actlens-<version>.tar.gz and .whl
twine upload dist/*                 # or: twine upload --repository testpypi dist/*
```

## Views

One page. Pick an **activation** and a **layer** in the selection bar (`[` `]` step the layer); the mode toggle
switches between two views of that activation:

- **Layer** — token × channel heatmap of the chosen layer (window, minimap of per-channel |max|, ordering, pooling,
  colormap/range/scale, brush/cursor, export). For `attn_pattern` it is the attention view instead: a grid of all heads,
  click one for the full query × key map, plus a layer × head statistic map (entropy / sink mass / distance).
- **Across layers** — token × layer map of a per-token statistic (L2 norm, |max|, mean, std, kurtosis, or one channel).
  Click a cell to jump to that layer with the token cursor placed.

Activations (Llama-style blocks; the picker only lists what the loaded model has):

| Group | Activations |
|---|---|
| Residual | `resid_pre` (attn_norm input), `resid_mid` (mlp_norm input), `resid_post` |
| Attention | `attn_norm`, `q`, `k`, `v`, `q_norm`/`k_norm` (models with QK-norm), `q_rope`/`k_rope` (after RoPE), `attn_pattern`, `attn_ctx` (o_proj input), `o` |
| MLP | `mlp_norm`, `gate`, `up`, `silu`, `swiglu` (= down_proj input), `down` |

Activations are captured lazily: the first time you open one, a forward pass records it for every layer (~0.1–0.4 s
for Qwen3-0.6B) and it is cached (`ACTLENS_CACHE_MB`, default 2048).

Heatmaps only render a window (default 64 tokens × 128 channels). The server pools anything larger than the pixel
budget, so zooming all the way out is cheap. Channels can be ranked by |max| / std / |mean| to surface outlier channels.
For head-structured activations (`q k v q_norm k_norm q_rope k_rope attn_ctx`) the channel axis is `head × head_dim`:
axis ticks read `h3·17`, faint separators mark head boundaries, and the **Head** control jumps the window to one head.

Interaction: drag = pan · pinch or ⌘/Ctrl + scroll = zoom · scroll = pan · Shift + drag = select region ·
click = cursor · double-click = reset · `[` `]` = previous/next layer · Esc = clear selection.

### Distribution panel

- **Values** — histogram and summary (mean/std/percentiles/kurtosis/skew) of the raw values; scope = visible window,
  brushed selection, the cursor's **channel** (over all tokens), the cursor's **token** (over all channels), or the whole layer.
- **Per channel** — pick a statistic (|max|, std, mean, norm, kurtosis): one number per channel (computed over tokens),
  shown as a histogram across channels with a clickable top-channels list.
- **Per token** — the same, one number per token (computed over channels), with a clickable top-tokens list.
- In both per-axis modes, an **Inside the selected channel / token** section shows the histogram (and summary) of the values
  inside the cursor's channel (over the region's tokens) or token (over the region's channels) — the values that its
  statistic was computed from. The overall distribution stays above it, with the selected item's statistic marked in blue.
  Click a heatmap cell or a top-list entry to change the selection.

## Export

PNG and PDF buttons render the current view (title, prompt, window, colormap settings, axes, colorbar) at 1–4×.
The histogram and the attention head grid have their own PNG export. The PDF embeds the figure as a high-resolution
image rather than vector graphics, so CJK tokens render correctly.

## Notes

- Model loading: `device_map="mps"` hangs with transformers 5.x, so the model is loaded on CPU and then moved.
- Attention capture needs `attn_implementation="eager"` (SDPA doesn't return probabilities).
- Supported architectures: Llama-style decoders (`model.layers[i].{input_layernorm, self_attn.{q,k,v,o}_proj,
  post_attention_layernorm, mlp.{gate,up,down}_proj, act_fn}`: Qwen2/2.5/3, Llama, Mistral, SmolLM). GPT-2 style models
  are rejected with a clear error. Only Qwen3-0.6B has been tested on a real checkpoint (a tiny random Llama without
  QK-norm is also tested).
- Activations (`backend/actlens/capture.py` registry): residual stream
  (`resid_pre/mid/post`), attention (`attn_norm, q, k, v, q_norm, k_norm, q_rope, k_rope, attn_pattern, attn_ctx, o`),
  MLP (`mlp_norm, gate, up, silu, swiglu, down`). `/api/run` lists only what the loaded model has: `q_norm`/`k_norm` need
  QK-norm modules and `q_rope`/`k_rope` need a `rotary_emb` module (RoPE is applied from `rotary_emb`'s cos/sin, since
  `apply_rotary_pos_emb` is a function, not a hookable module).
- Lazy capture: `POST /api/run` only tokenizes. The first data request for an activation runs one forward pass on the
  model thread that captures exactly that activation for all layers (~60-400 ms for Qwen3-0.6B at 64 tokens); results
  are kept in a byte-budgeted LRU keyed by `(run_id, act)` (`ACTLENS_CACHE_MB`, default 2048; the entry just built is never
  evicted). Concurrent requests for the same activation share one capture. Only the last 3 runs are registered.
- `GET /api/run/{id}/axis_stats` gives the distribution of a per-channel or per-token statistic over a region;
  the maths is in `slicing.py` (`axis_stats`, `axis_stat_values`).
- Tests: `cd backend && python -m pytest` runs fast fake-provider tests plus `tests/test_real_model.py` (Qwen3-0.6B
  identities incl. recomputing attention probabilities from `q_rope`/`k_rope`; ~10 s). The real-model module is skipped
  when the model is not in the local HF cache or `ACTLENS_SKIP_REAL=1`.

## License

[MIT](LICENSE)
