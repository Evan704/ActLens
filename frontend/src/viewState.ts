/**
 * Pure view-state logic for the unified activation view (no React, no zustand), so the transitions
 * can be unit-tested: per-activation memory, defaults, switching activations, opening a cell.
 */
import type { ActId, ActivationInfo, AggKind, AttnStat, OrderKind, OverviewStat, RunInfo } from "./api";
import { headLayout, headWindow } from "./channels";
import type { Display } from "./colormaps";
import type { Cell, Region } from "./figure";
import { clampViewport, windowViewport, type Viewport } from "./viewport";

export type Mode = "layer" | "across" | "arch" | "lens";

export const ACT_DISPLAY: Display = { cmap: "RdBu", range: "robust", scale: "linear", symmetric: true, manual: [-1, 1] };
export const SEQ_DISPLAY: Display = { cmap: "viridis", range: "robust", scale: "symlog", symmetric: false, manual: [0, 1] };
export const ATTN_DISPLAY: Display = { cmap: "magma", range: "manual", scale: "sqrt", symmetric: false, manual: [0, 1] };

/** Default window: a modest slice, never the whole (potentially huge) matrix. */
export const DEFAULT_TOKENS = 64;
export const DEFAULT_CHANNELS = 128;

/** Cursor in the token x channel view. `col` is a rank coordinate; `id` the original channel when known. */
export interface TokenCursor extends Cell {
  id?: number;
}

/** What each activation remembers between visits. */
export interface ActSettings {
  display: Display;
  order: OrderKind;
  agg: AggKind;
  vp: Viewport;
}
export type ActMemory = Record<ActId, ActSettings>;

/** The live state of the token x channel view (belongs to the most recently selected token activation). */
export interface TokenView extends ActSettings {
  brush: Region | null;
  cursor: TokenCursor | null;
}

export interface AcrossState {
  stat: OverviewStat;
  channel: number;
  display: Display;
  vp: Viewport;
}
export interface AttnState {
  head: number | null;
  stat: AttnStat;
  sortBy: "index" | AttnStat;
  display: Display;
  vp: Viewport;
  brush: Region | null;
  cursor: Cell | null;
}

export const findAct = (run: RunInfo | null, id: ActId): ActivationInfo | undefined => run?.activations.find((a) => a.id === id);

/** Signed activations use a symmetric diverging map, attention probabilities a manual [0,1] sqrt magma. */
export const defaultDisplay = (info: ActivationInfo): Display => (info.kind === "attn" ? ATTN_DISPLAY : ACT_DISPLAY);

/** Wide MLP activations (gate/up/silu/swiglu) are sparse-ish, so rank channels by |max| as the old MLP tab did. */
export function defaultOrder(info: ActivationInfo): OrderKind {
  return info.kind === "token" && info.group === "MLP" && (info.dim ?? 0) > 1024 ? "absmax" : "natural";
}

/** 64 tokens x min(C,128) channels; head-structured activations start on exactly head 0. */
export function defaultWindow(info: ActivationInfo, nTokens: number): Viewport {
  const nDims = info.dim ?? 0;
  const heads = headLayout(info);
  const span = heads ? heads.headDim : Math.min(nDims, DEFAULT_CHANNELS);
  return windowViewport(0, span, 0, Math.min(nTokens, DEFAULT_TOKENS), { nx: nDims, ny: nTokens });
}

export function defaultSettings(info: ActivationInfo, nTokens: number): ActSettings {
  return { display: defaultDisplay(info), order: defaultOrder(info), agg: "absmax", vp: defaultWindow(info, nTokens) };
}

export const homeViewport = (nx: number, ny: number): Viewport => ({ x0: 0, x1: nx, y0: 0, y1: ny });

export const clampLayer = (layer: number, info: Pick<ActivationInfo, "n_layers">) => Math.max(0, Math.min(info.n_layers - 1, Math.round(layer)));

export interface Switched {
  tok: TokenView;
  memory: ActMemory;
}

/**
 * Switch the selected activation. The old token activation's settings are remembered; the new one restores its own
 * display/order/pooling/channel window (or defaults) while the token window carries over. Selection is cleared.
 * `from` may be an attention activation (its state lives elsewhere) or undefined.
 */
export function switchActivation(
  tok: TokenView,
  memory: ActMemory,
  from: ActivationInfo | undefined,
  to: ActivationInfo,
  nTokens: number,
): Switched {
  const mem = { ...memory };
  if (from?.kind === "token") mem[from.id] = { display: tok.display, order: tok.order, agg: tok.agg, vp: tok.vp };
  if (to.kind !== "token") return { tok, memory: mem };
  const saved = mem[to.id] ?? defaultSettings(to, nTokens);
  const ext = { nx: to.dim ?? 0, ny: nTokens };
  const vp = clampViewport({ x0: saved.vp.x0, x1: saved.vp.x1, y0: tok.vp.y0, y1: tok.vp.y1 }, ext, 1);
  return { tok: { ...saved, vp, brush: null, cursor: null }, memory: mem };
}

/**
 * Jump from the across-layers map to the layer view: centre the token window on `token` (keeping its span) and place
 * the cursor there. The cursor column stays where it was; `channel` (a rank coordinate) is given for the
 * single-channel statistic and is brought into the window.
 */
export function openCell(tok: TokenView, info: ActivationInfo, nTokens: number, token: number, channel: number | null): TokenView {
  const nDims = info.dim ?? 0;
  const ext = { nx: nDims, ny: nTokens };
  const span = Math.max(8, tok.vp.y1 - tok.vp.y0);
  const y0 = Math.max(0, Math.min(nTokens - span, token - span / 2));
  let vp = windowViewport(tok.vp.x0, tok.vp.x1 - tok.vp.x0, y0, span, ext);
  let col = tok.cursor?.col ?? Math.floor(vp.x0);
  if (channel !== null) {
    col = Math.max(0, Math.min(nDims - 1, channel));
    if (col < vp.x0 || col >= vp.x1) vp = windowViewport(col - (vp.x1 - vp.x0) / 2, vp.x1 - vp.x0, vp.y0, vp.y1 - vp.y0, ext);
  }
  col = Math.max(0, Math.min(nDims - 1, col));
  return { ...tok, vp, brush: null, cursor: { row: token, col, id: channel !== null ? channel : undefined } };
}

/** Move the cursor (and, if needed, the window) so a token row is visible. Used by the distribution panel. */
export function pickToken(tok: TokenView, nTokens: number, nDims: number, token: number): TokenView {
  const ext = { nx: nDims, ny: nTokens };
  const t = Math.max(0, Math.min(nTokens - 1, token));
  const span = tok.vp.y1 - tok.vp.y0;
  const vp = t >= tok.vp.y0 && t < tok.vp.y1 ? tok.vp : windowViewport(tok.vp.x0, tok.vp.x1 - tok.vp.x0, t - span / 2, span, ext);
  const keep = tok.cursor;
  return { ...tok, vp, cursor: { row: t, col: keep?.col ?? Math.floor(vp.x0), id: keep?.id } };
}

/** Same for a channel rank. */
export function pickChannel(tok: TokenView, nTokens: number, nDims: number, rank: number): TokenView {
  const ext = { nx: nDims, ny: nTokens };
  const c = Math.max(0, Math.min(nDims - 1, rank));
  const span = tok.vp.x1 - tok.vp.x0;
  const vp = c >= tok.vp.x0 && c < tok.vp.x1 ? tok.vp : windowViewport(c - span / 2, span, tok.vp.y0, tok.vp.y1 - tok.vp.y0, ext);
  return { ...tok, vp, cursor: { row: tok.cursor?.row ?? Math.floor(vp.y0), col: c } };
}

/** Jump the channel window to one head. Heads only exist in natural channel order, so ranking is switched off. */
export function jumpToHead(tok: TokenView, info: ActivationInfo, nTokens: number, head: number): TokenView {
  const heads = headLayout(info);
  if (!heads) return tok;
  return { ...tok, order: "natural", vp: headWindow(tok.vp, head, heads, { nx: info.dim ?? 0, ny: nTokens }), brush: null, cursor: null };
}

export interface Core {
  act: ActId;
  layer: number;
  memory: ActMemory;
  tok: TokenView;
  attn: AttnState;
  across: AcrossState;
}

/**
 * State for a freshly registered run. Same model: keep activation, layer and per-activation settings, but restart the
 * token windows (the token axis changed). New model: start from defaults, mid-network.
 */
export function resetForRun(run: RunInfo, prev: Core & { run: RunInfo | null }): Core {
  const T = run.tokens.length;
  const sameModel = prev.run !== null && prev.run.model_id === run.model_id;
  const act = run.activations.find((a) => a.id === prev.act) ?? run.activations.find((a) => a.id === "resid_post") ?? run.activations[0];
  // first run of a model: start mid-network, where activations are informative (layer 0 is just the embedding)
  const layer = sameModel ? clampLayer(prev.layer, act) : Math.floor(act.n_layers / 2);

  const memory: ActMemory = {};
  if (sameModel) {
    for (const a of run.activations) {
      const m = prev.memory[a.id];
      if (a.kind === "token" && m) memory[a.id] = { ...m, vp: resetTokenWindow(m.vp, a, T) };
    }
  }
  const liveAct = act.kind === "token" ? act : run.activations.find((a) => a.kind === "token") ?? act;
  const saved: ActSettings = sameModel && act.kind === "token" && act.id === prev.act ? prev.tok : memory[liveAct.id] ?? defaultSettings(liveAct, T);
  const tok: TokenView = { ...saved, vp: resetTokenWindow(saved.vp, liveAct, T), brush: null, cursor: null };

  const attnInfo = run.activations.find((a) => a.kind === "attn");
  const head = sameModel && prev.attn.head !== null ? Math.min(prev.attn.head, (attnInfo?.n_heads ?? 1) - 1) : null;
  return {
    act: act.id,
    layer,
    memory,
    tok,
    attn: { ...prev.attn, head, vp: homeViewport(T, T), brush: null, cursor: null },
    across: { ...prev.across, vp: homeViewport(act.n_layers, T) },
  };
}

/** Keep the channel window (clamped), put the token window back to the top. */
function resetTokenWindow(vp: Viewport, info: ActivationInfo, nTokens: number): Viewport {
  const top = defaultWindow(info, nTokens);
  return clampViewport({ x0: vp.x0, x1: vp.x1, y0: top.y0, y1: top.y1 }, { nx: info.dim ?? 0, ny: nTokens }, 1);
}

export function initialCore(): Core {
  return {
    act: "resid_post",
    layer: 0,
    memory: {},
    tok: { display: ACT_DISPLAY, order: "natural", agg: "absmax", vp: { x0: 0, x1: DEFAULT_CHANNELS, y0: 0, y1: DEFAULT_TOKENS }, brush: null, cursor: null },
    attn: { head: null, stat: "entropy", sortBy: "index", display: ATTN_DISPLAY, vp: homeViewport(1, 1), brush: null, cursor: null },
    across: { stat: "norm", channel: 0, display: SEQ_DISPLAY, vp: homeViewport(1, 1) },
  };
}
