/**
 * Types, fetchers and pure helpers for the distribution panel.
 *
 * `/stats` and `/axis_stats` both take `act` (not `site`). Regions are `[t0,t1) x [d0,d1)` with `d` in
 * `order`-rank coordinates, exactly like `/slice`.
 */
import { getJSON, runPath, type OrderKind, type Stats } from "./api";
import type { Region4 } from "./components/DistributionPanel";
import { DEFAULT_META, statOfId, type StatInfo } from "./meta";
import { fmtToken } from "./tokens";

// ---------------------------------------------------------------------------------------------
// axis_stats API
// ---------------------------------------------------------------------------------------------

export type AxisKind = "channel" | "token";
export type AxisStat = string;

/** Short name of a statistic; `stats` comes from `useMeta()`. */
export const axisStatLabel = (s: AxisStat, stats: StatInfo[] = DEFAULT_META.stats): string => statOfId(stats, s)?.label ?? s;

export interface AxisTop {
  /** rank in `order` for channels, token position for tokens */
  index: number;
  /** original channel id (== index for tokens) */
  id: number;
  value: number;
}

/** `Stats` computed over the per-index statistic values, plus the axis bookkeeping. */
export interface AxisStatsResponse extends Omit<Stats, "top" | "region"> {
  axis: AxisKind;
  stat: AxisStat;
  top: AxisTop[];
  region: Region4;
}

export interface AxisStatsQuery {
  runId: string;
  act: string;
  layer: number;
  axis: AxisKind;
  stat: AxisStat;
  region: Region4;
  order: OrderKind;
  clip: boolean;
  top?: number;
  bins?: number;
}

/** Position of `axis` inside {@link axisStatsKey}; used to avoid showing channel data under "Per token". */
const AXIS_KEY_POS = 4;

export function axisStatsKey(q: AxisStatsQuery) {
  const r = q.region;
  return ["axis_stats", q.runId, q.act, q.layer, q.axis, q.stat, r.t0, r.t1, r.d0, r.d1, q.order, q.clip, q.top ?? 10, q.bins ?? 64] as const;
}

export function fetchAxisStats(q: AxisStatsQuery, signal?: AbortSignal): Promise<AxisStatsResponse> {
  return getJSON<AxisStatsResponse>(
    runPath(q.runId, "axis_stats"),
    {
      act: q.act,
      layer: q.layer,
      axis: q.axis,
      stat: q.stat,
      ...q.region,
      order: q.order,
      clip: q.clip,
      top: q.top ?? 10,
      bins: q.bins ?? 64,
    },
    signal,
  );
}

/** `placeholderData` for react-query: keep the previous result while refetching, but only within one axis. */
export function keepSameAxis<T>(axis: AxisKind) {
  return (prev: T | undefined, prevQuery: { queryKey: readonly unknown[] } | undefined): T | undefined =>
    prevQuery && prevQuery.queryKey[AXIS_KEY_POS] === axis ? prev : undefined;
}

// ---------------------------------------------------------------------------------------------
// /stats (raw values)
// ---------------------------------------------------------------------------------------------

export interface ValueStatsQuery {
  runId: string;
  act: string;
  layer: number;
  region: Region4;
  order: OrderKind;
  clip: boolean;
}

export function valueStatsKey(q: ValueStatsQuery) {
  const r = q.region;
  return ["stats", q.runId, q.act, q.layer, r.t0, r.t1, r.d0, r.d1, q.order, q.clip] as const;
}

export function fetchValueStats(q: ValueStatsQuery, signal?: AbortSignal): Promise<Stats> {
  return getJSON<Stats>(runPath(q.runId, "stats"), { act: q.act, layer: q.layer, ...q.region, order: q.order, clip: q.clip }, signal);
}

// ---------------------------------------------------------------------------------------------
// scopes -> regions
// ---------------------------------------------------------------------------------------------

export type ValueScope = "window" | "selection" | "channel" | "token" | "layer";
/** Scope of the per-channel / per-token statistics. */
export type RegionScope = "window" | "selection" | "layer";

export const VALUE_SCOPES: { id: ValueScope; label: string; title: string }[] = [
  { id: "window", label: "window", title: "the visible window of the heatmap" },
  { id: "selection", label: "selection", title: "the brushed selection (drag on the heatmap)" },
  { id: "channel", label: "channel", title: "the cursor's channel, over all tokens (click a cell to set the cursor)" },
  { id: "token", label: "token", title: "the cursor's token, over all channels (click a cell to set the cursor)" },
  { id: "layer", label: "layer", title: "the whole layer: all tokens and all channels" },
];
export const REGION_SCOPES = VALUE_SCOPES.filter((s) => s.id === "window" || s.id === "selection" || s.id === "layer") as {
  id: RegionScope;
  label: string;
  title: string;
}[];

export interface ScopeContext {
  nTokens: number;
  nChannels: number;
  window: Region4;
  selection: Region4 | null;
  cursor: { token: number; channel: number } | null;
}

const clampInt = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, Math.round(v)));

/** Integer region inside `[0,nTokens) x [0,nChannels)`, at least one cell wide on each axis (when the tensor is non-empty). */
export function clampRegion(r: Region4, nTokens: number, nChannels: number): Region4 {
  const t0 = clampInt(Math.floor(r.t0), 0, Math.max(0, nTokens - 1));
  const d0 = clampInt(Math.floor(r.d0), 0, Math.max(0, nChannels - 1));
  const t1 = clampInt(Math.ceil(r.t1), t0 + 1, Math.max(t0 + 1, nTokens));
  const d1 = clampInt(Math.ceil(r.d1), d0 + 1, Math.max(d0 + 1, nChannels));
  return { t0, t1, d0, d1 };
}

/** Is the prerequisite of `scope` (a brushed selection, a cursor) present? */
export function scopeAvailable(scope: ValueScope, ctx: Pick<ScopeContext, "selection" | "cursor">): boolean {
  if (scope === "selection") return ctx.selection !== null;
  if (scope === "channel" || scope === "token") return ctx.cursor !== null;
  return true;
}

export function scopeDisabledReason(scope: ValueScope): string | null {
  if (scope === "selection") return "brush a region on the heatmap first";
  if (scope === "channel" || scope === "token") return "click a cell on the heatmap to place the cursor first";
  return null;
}

/** Region queried for a value scope; `null` when its prerequisite is missing (callers fall back to `window`). */
export function resolveValueRegion(scope: ValueScope, ctx: ScopeContext): Region4 | null {
  const { nTokens: T, nChannels: C } = ctx;
  switch (scope) {
    case "window":
      return clampRegion(ctx.window, T, C);
    case "selection":
      return ctx.selection ? clampRegion(ctx.selection, T, C) : null;
    case "channel":
      return ctx.cursor ? clampRegion({ t0: 0, t1: T, d0: ctx.cursor.channel, d1: ctx.cursor.channel + 1 }, T, C) : null;
    case "token":
      return ctx.cursor ? clampRegion({ t0: ctx.cursor.token, t1: ctx.cursor.token + 1, d0: 0, d1: C }, T, C) : null;
    case "layer":
      return clampRegion({ t0: 0, t1: T, d0: 0, d1: C }, T, C);
  }
}

/** Same for the per-axis statistics (window / selection / layer only). */
export function resolveAxisRegion(scope: RegionScope, ctx: ScopeContext): Region4 | null {
  return resolveValueRegion(scope, ctx);
}

/**
 * Region of the cursor's own channel / token *inside* an axis-stats region: the values that the per-channel (per-token)
 * statistic was computed from. `null` without a cursor or when the cursor lies outside `region`.
 */
export function insideRegion(axis: AxisKind, region: Region4, cursor: { token: number; channel: number } | null): Region4 | null {
  if (!cursor) return null;
  if (axis === "channel") {
    const c = cursor.channel;
    return c >= region.d0 && c < region.d1 ? { t0: region.t0, t1: region.t1, d0: c, d1: c + 1 } : null;
  }
  const t = cursor.token;
  return t >= region.t0 && t < region.t1 ? { t0: t, t1: t + 1, d0: region.d0, d1: region.d1 } : null;
}

/** The per-axis statistic `stat` recovered from the summary of one channel's / token's values (what `axis_stats` reports for it).
 * The summary only carries these five; a statistic registered later on the backend has no value here (null). */
export function statOf(stat: AxisStat, s: Pick<Stats, "absmax" | "std" | "mean" | "l2" | "kurtosis">): number | null {
  switch (stat) {
    case "absmax":
      return s.absmax;
    case "std":
      return s.std;
    case "mean":
      return s.mean;
    case "norm":
      return s.l2;
    case "kurtosis":
      return s.kurtosis;
    default:
      return null;
  }
}

/** The scope actually used: `wanted` if available, otherwise the visible window. */
export function effectiveScope<S extends ValueScope>(wanted: S, ctx: Pick<ScopeContext, "selection" | "cursor">): S | "window" {
  return scopeAvailable(wanted, ctx) ? wanted : "window";
}

// ---------------------------------------------------------------------------------------------
// labels & formatting
// ---------------------------------------------------------------------------------------------

/** `h3·17` for head-structured channel axes, else the plain id. */
export function channelLabel(id: number, headDim: number | null): string {
  if (headDim && headDim > 0) return `h${Math.floor(id / headDim)}·${id % headDim}`;
  return String(id);
}

/** Long form used in tooltips / exports: `head 3, dim 17 (channel 401)` or `channel 401`. */
export function channelName(id: number, headDim: number | null): string {
  if (headDim && headDim > 0) return `head ${Math.floor(id / headDim)}, dim ${id % headDim} (channel ${id})`;
  return `channel ${id}`;
}

/** `#12 ·capital`. */
export function tokenLabel(index: number, tokens: string[], max = 14): string {
  const t = tokens[index];
  return t === undefined ? `#${index}` : `#${index} ${fmtToken(t, max)}`;
}

/** A top-list entry's label for the axis. `index` is a rank for channels (differs from `id` when `order != natural`). */
export function topLabel(axis: AxisKind, t: AxisTop, tokens: string[], headDim: number | null): { main: string; sub: string } {
  if (axis === "token") return { main: tokenLabel(t.index, tokens), sub: "" };
  const ranked = t.index !== t.id;
  const main = headDim ? `${t.id} · ${channelLabel(t.id, headDim)}` : `ch ${t.id}`;
  return { main, sub: ranked ? `rank ${t.index}` : "" };
}

/** Number formatting tolerant of missing / non-finite values (JSON has no NaN; the backend may send null). */
export function fmtNum(v: number | null | undefined): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e5 || a < 1e-3) return v.toExponential(2);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return String(Number(v.toPrecision(3)));
}

export function fmtPct(frac: number | null | undefined, digits = 1): string {
  if (frac === null || frac === undefined || !Number.isFinite(frac)) return "–";
  const p = frac * 100;
  // never show a non-zero fraction as 0.0%
  if (p > 0 && p < Math.pow(10, -digits)) return `<${Math.pow(10, -digits)}%`;
  return `${p.toFixed(digits)}%`;
}

/** `tokens 0–63 · channels 0–127`, with `(ranked)` when channels are re-ordered. */
export function regionText(r: Region4, order: OrderKind, headDim: number | null = null): string {
  const oneHead = headDim && r.d1 - r.d0 === headDim && order === "natural" && r.d0 % headDim === 0;
  const ch = oneHead ? `head ${r.d0 / headDim}` : r.d1 - r.d0 === 1 ? `channel ${r.d0}` : `channels ${r.d0}–${r.d1 - 1}`;
  const tk = r.t1 - r.t0 === 1 ? `token ${r.t0}` : `tokens ${r.t0}–${r.t1 - 1}`;
  return `${tk} · ${ch}${order !== "natural" && r.d1 - r.d0 > 1 ? " (ranked)" : ""}`;
}

/** Human description of a value scope: what the numbers are about. */
export function scopeText(
  scope: ValueScope,
  o: { cursor: { token: number; channel: number } | null; cursorChannelId: number | null; tokens: string[]; headDim: number | null },
): string {
  switch (scope) {
    case "window":
      return "visible window";
    case "selection":
      return "brushed selection";
    case "layer":
      return "whole layer";
    case "channel":
      return o.cursor
        ? `channel ${o.cursorChannelId !== null ? channelLabel(o.cursorChannelId, o.headDim) : `rank ${o.cursor.channel}`}, all tokens`
        : "channel";
    case "token":
      return o.cursor ? `${tokenLabel(o.cursor.token, o.tokens, 12)}, all channels` : "token";
  }
}

/** What the per-axis histogram is a histogram *of*. */
export function axisStatTitle(axis: AxisKind, stat: AxisStat, stats?: StatInfo[]): string {
  return `${axisStatLabel(stat, stats)} per ${axis}`;
}

/** Share of values outside the (clipped) histogram range. */
export function clippedShare(nClipped: number, n: number): number {
  return n > 0 ? nClipped / n : 0;
}
