# ActLens

**See inside a language model while it reads your prompt.**

ActLens is an interactive viewer for the activations of any supported Hugging Face model. Type a prompt, pick an
activation and a layer, and browse the result as a token × channel heatmap: residual stream, attention patterns, Q/K/V,
MLP internals and more. It runs locally, in one command.

[![PyPI](https://img.shields.io/pypi/v/actlens)](https://pypi.org/project/actlens/)
[![Python](https://img.shields.io/pypi/pyversions/actlens)](https://pypi.org/project/actlens/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Evan704/ActLens/blob/main/LICENSE)

![ActLens showing the residual stream of Qwen3-0.6B](https://raw.githubusercontent.com/Evan704/ActLens/main/docs/images/layer-view.png)

## Features

- **Every activation, every layer.** Residual stream, attention (Q, K, V, RoPE, patterns, context, output) and MLP
  (gate, up, SwiGLU, down), captured lazily and cached.
- **Two views.** *Layer* shows a token × channel heatmap for one layer; *Across layers* shows a per-token statistic
  (L2 norm, |max|, mean, std, kurtosis, or a single channel) for every layer at once.
- **Attention explorer.** A grid of all heads, a full query × key map per head, and a layer × head map of entropy,
  sink mass and attention distance.
- **Find outlier channels.** Rank channels by |max|, std or |mean|; jump straight to a head.
- **Distributions.** Histograms and summary statistics over a region, a channel, a token, or the whole layer.
- **Fast to navigate.** Pan, zoom, brush and inspect; large activations are pooled server-side so zooming out stays cheap.
- **Publication-ready export.** PNG and PDF with title, prompt, axes and colorbar at up to 4×.
- **Extensible.** Add support for a new architecture with a small adapter.

See [Examples](https://github.com/Evan704/ActLens/blob/main/docs/examples.md) for a massive activation on the first
token, attention sinks and previous-token heads, each a few clicks away.

## Quick start

```bash
pip install actlens      # Python >= 3.10; use a fresh virtual environment
actlens                  # loads Qwen/Qwen3-0.6B and serves the UI at http://127.0.0.1:8000
```

The first run downloads the model weights from the Hugging Face Hub. ActLens is developed and tested against
torch 2, transformers 5 and nnsight 0.7.

```bash
actlens -m Qwen/Qwen2.5-0.5B             # another Hugging Face model id or a local path
actlens --device cuda --dtype bfloat16   # device: auto | cpu | cuda | mps; dtype: float32 | float16 | bfloat16
actlens --port 9000 --open               # custom port, open the browser when ready
actlens --cache-mb 4096                  # activation cache budget (or $ACTLENS_CACHE_MB)
```

You can load more models from the UI at any time.

> **Security note.** The server binds to `127.0.0.1` by default. The API can load any model and has no
> authentication, so only use `--host 0.0.0.0` on a network you trust.

### Google Colab

No local GPU? Run the model on a free Colab GPU and view the visualization in your own browser.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Evan704/ActLens/blob/main/colab/ActLens.ipynb)

1. Open the notebook and pick a GPU runtime (Runtime → Change runtime type → T4 GPU).
2. Run the first cell. It installs ActLens, starts the server and opens a Cloudflare tunnel.
3. Click the **Open ActLens** link it prints. The UI opens in your browser; keep the Colab tab open while you use it.

Or in any notebook:

```python
!pip install -q actlens
from actlens.colab import launch
launch(model="Qwen/Qwen3-0.6B", dtype="float16")
```

The link contains a random access token, and the server rejects requests without it. Anyone who has the full link can use
your session, so do not share it. Use `actlens.colab.stop()` to shut everything down. If the page does not load, check
`actlens.log`.

You can also protect a server of your own with `actlens --token` (generates a token and prints the URL) or `--token VALUE`.

## Using the viewer

Pick an **activation** and a **layer** in the selection bar (`[` and `]` step through layers), then choose a view.

| Interaction | Action |
|---|---|
| Drag | Pan |
| Pinch, or ⌘/Ctrl + scroll | Zoom |
| Shift + drag | Select a region |
| Click | Place the cursor |
| Double-click | Reset the view |
| `[` / `]` | Previous / next layer |
| Esc | Clear the selection |

For head-structured activations (`q k v q_norm k_norm q_rope k_rope attn_ctx`) the channel axis is `head × head_dim`,
with ticks like `h3·17` and faint separators between heads; the **Head** control jumps the window to one head.

### Available activations

The picker lists only what the loaded model provides.

| Group | Activations |
|---|---|
| Residual | `resid_pre`, `resid_mid`, `resid_post` |
| Attention | `attn_norm`, `q`, `k`, `v`, `q_norm`, `k_norm`, `q_rope`, `k_rope`, `attn_pattern`, `attn_ctx`, `o` |
| MLP | `mlp_norm`, `gate`, `up`, `silu`, `swiglu`, `mlp_act`, `down` |

### Distribution panel

- **Values**: histogram and summary (mean, std, percentiles, kurtosis, skew) of the raw values for the visible window,
  a brushed selection, the cursor's channel or token, or the whole layer.
- **Per channel / Per token**: one statistic per channel or token, shown as a histogram with a clickable list of the
  top outliers.

### Export

The PNG and PDF buttons render the current view at 1–4×. The histogram and the attention head grid have their own PNG
export. PDFs embed the figure as a high-resolution image so CJK tokens render correctly.

## Supported models

| Adapter | `model_type` | Models |
|---|---|---|
| `llama` | `llama`, `qwen2`, `qwen3`, `mistral`, and models with the same module layout | Llama, Qwen2/2.5/3, Mistral, SmolLM |
| `gpt2` | `gpt2` | GPT-2, DistilGPT-2 |

Qwen3-0.6B and GPT-2 are tested on real checkpoints; every adapter is also tested on a tiny random model. An
unsupported model is rejected at load time with a message naming the missing modules.

Want another architecture? See [Adding an architecture](https://github.com/Evan704/ActLens/blob/main/CONTRIBUTING.md#adding-an-architecture).

## Troubleshooting

- **Attention patterns need eager attention.** ActLens loads models with `attn_implementation="eager"` because SDPA does
  not return attention probabilities.
- **Slow first open of an activation.** The first time you open an activation, one forward pass captures it for every
  layer (about 0.1–0.4 s for Qwen3-0.6B); after that it comes from the cache.
- **Running out of memory.** Use a smaller model, `--dtype float16` or `bfloat16`, or lower `--cache-mb`.

## Contributing

Development setup, tests, architecture notes and the release process are in
[CONTRIBUTING.md](https://github.com/Evan704/ActLens/blob/main/CONTRIBUTING.md).

## License

[MIT](https://github.com/Evan704/ActLens/blob/main/LICENSE)
