/** Pure geometry for the histogram canvas (kept free of DOM / canvas so it can be unit-tested). */

export interface HistMargins {
  l: number;
  r: number;
  t: number;
  b: number;
}

/** Plot-area margins in CSS px. `headerLines` is the number of title / meta lines drawn above (export only). */
export function histMargins(opts: { headerLines?: number; xLabel?: boolean }): HistMargins {
  const lines = opts.headerLines ?? 0;
  return { l: 44, r: 10, t: lines > 0 ? 8 + 6 + lines * 15 : 8, b: opts.xLabel ? 40 : 24 };
}

export const plotWidth = (w: number, m: HistMargins) => Math.max(1, w - m.l - m.r);
export const plotHeight = (h: number, m: HistMargins) => Math.max(1, h - m.t - m.b);

/** Bin under a pointer at `px` (CSS px from the canvas left edge), or null outside the plot. */
export function binAtX(px: number, w: number, m: HistMargins, nBins: number): number | null {
  const f = (px - m.l) / plotWidth(w, m);
  if (!(f >= 0 && f < 1) || nBins <= 0) return null;
  return Math.min(nBins - 1, Math.floor(f * nBins));
}

/** Screen x of data value `v` for an axis spanning `[lo, hi]` starting at `x0` and `pw` wide. */
export function valueToX(v: number, lo: number, hi: number, x0: number, pw: number): number {
  return x0 + ((v - lo) / (hi - lo || 1)) * pw;
}

/** Bar x-extent of bin `i` (a 1 px gap between neighbours when they are wide enough). */
export function barX(i: number, edges: number[], lo: number, hi: number, x0: number, pw: number): { x: number; w: number } {
  const xa = valueToX(edges[i], lo, hi, x0, pw);
  const xb = valueToX(edges[i + 1], lo, hi, x0, pw);
  const gap = xb - xa > 3 ? 1 : 0;
  return { x: xa + gap / 2, w: Math.max(1, xb - xa - gap) };
}

/** Height scale: linear counts, or log10(1 + count) so empty bins stay at 0 and singletons are visible. */
export const yValue = (count: number, logY: boolean) => (logY ? Math.log10(1 + count) : count);

/** Bar height in px; any non-empty bin gets at least 1.5 px so rare outlier bins never vanish. */
export function barHeight(count: number, maxCount: number, logY: boolean, ph: number): number {
  if (count <= 0) return 0;
  const top = yValue(Math.max(1, maxCount), logY);
  return Math.max(1.5, (yValue(count, logY) / top) * ph);
}

function niceStep(raw: number): number {
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const f = raw / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
}

/** Y axis ticks: `count` value and `frac` (0 = baseline, 1 = top of plot). */
export function yTicks(maxCount: number, logY: boolean): { count: number; frac: number }[] {
  const mx = Math.max(1, maxCount);
  const top = yValue(mx, logY);
  const out: { count: number; frac: number }[] = [{ count: 0, frac: 0 }];
  if (logY) {
    for (let p = 0; Math.pow(10, p) <= mx; p++) out.push({ count: Math.pow(10, p), frac: yValue(Math.pow(10, p), true) / top });
    // few decades: add 3x steps so the axis is not just 0 and 1
    if (out.length < 3) out.push({ count: mx, frac: 1 });
    return out;
  }
  const step = niceStep(mx / 3);
  for (let c = step; c <= mx * 1.0001; c += step) out.push({ count: c, frac: c / mx });
  return out;
}

/** Clip fraction helper for the "x outside" note: percentage string with sensible precision. */
export function outsideNote(nClipped: number, n: number): string {
  if (nClipped <= 0) return "none outside";
  const p = (nClipped / Math.max(1, n)) * 100;
  return `${nClipped.toLocaleString()} outside (${p < 0.1 ? "<0.1" : p.toFixed(p < 10 ? 1 : 0)}%)`;
}
