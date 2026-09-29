# Changelog

## 0.2.0

### Added

- **Logit lens view.** Unembeds the residual stream of one token after every layer with the model's own final norm,
  output head and final logit soft-capping. Shows the top-k tokens per layer, follows a tracked token (by default the
  next prompt token) through the layers with its rank and probability, and plots the entropy of the next-token
  distribution. Backed by `GET /api/run/{run_id}/logit_lens`.
- **Architecture view.** A clickable decoder-block diagram drawn from the dataflow the backend ships for each model;
  click a node to open its activation.
- **More architectures.** Gemma, Gemma-2 and Gemma-3 (text), OLMo, OLMo-2, GPT-NeoX (Pythia), Phi-3, Granite and StableLM,
  in addition to Llama, Qwen, Mistral, SmolLM and GPT-2. Gemma-2/3 sandwich norms, attention soft-capping, sliding
  windows and per-layer RoPE tables are modelled. Any other model with the same module layout works unlisted.
- **Model loading feedback.** Download and load progress in the UI, and readable explanations of common load errors.
- `run.sh` is now a hot-reload dev server; `run.sh --prod` serves the built UI.
- README: remote server (SSH) section, screenshots and worked examples (`docs/examples.md`).

### Changed

- The backend is split into layers: the statistics, pooling modes, channel orders and attention head metrics that the
  slice endpoints accept live in a registry and are served to the UI from `/api/meta`, so adding one no longer means
  touching the frontend.
- The adapter contract only grew optional pieces (`Dims` fields with defaults, an `ArchAdapter.setup` hook that runs
  before the trace, `PreNormBlockAdapter.core_acts` / `branch_norm_acts`, `TraceCtx.layer`), so adapters written for
  0.1.0 keep working. One behaviour change: `has()` now treats an attribute set to `None` as absent.

### Notes

- Logit lens on early layers is often noise by nature; it is not a tuned lens.
