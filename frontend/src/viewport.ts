/** Viewport math in axis units (x = columns, y = rows). All functions are pure. */
export interface Viewport {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}
export interface Extent {
  nx: number;
  ny: number;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Keep a [lo, hi) span inside [0, n] without changing its size (unless it exceeds n). */
function fit(lo: number, hi: number, n: number, minSpan: number): [number, number] {
  const span = clamp(hi - lo, Math.min(minSpan, n), n);
  const start = clamp(lo, 0, n - span);
  return [start, start + span];
}

export function clampViewport(vp: Viewport, ext: Extent, minSpan = 2): Viewport {
  const [x0, x1] = fit(vp.x0, vp.x1, ext.nx, minSpan);
  const [y0, y1] = fit(vp.y0, vp.y1, ext.ny, minSpan);
  return { x0, x1, y0, y1 };
}

/** Zoom keeping the axis value under (fx, fy) — fractions of the plot, 0..1 — fixed. */
export function zoomAt(
  vp: Viewport,
  ext: Extent,
  fx: number,
  fy: number,
  factor: number,
  axes: "both" | "x" | "y" = "both",
  minSpan = 2,
): Viewport {
  const zoom = (lo: number, hi: number, f: number, n: number): [number, number] => {
    const span = hi - lo;
    const next = clamp(span * factor, Math.min(minSpan, n), n);
    const anchor = lo + f * span;
    return fit(anchor - f * next, anchor - f * next + next, n, minSpan);
  };
  const [x0, x1] = axes === "y" ? [vp.x0, vp.x1] : zoom(vp.x0, vp.x1, fx, ext.nx);
  const [y0, y1] = axes === "x" ? [vp.y0, vp.y1] : zoom(vp.y0, vp.y1, fy, ext.ny);
  return { x0, x1, y0, y1 };
}

export function pan(vp: Viewport, ext: Extent, dx: number, dy: number): Viewport {
  return clampViewport({ x0: vp.x0 + dx, x1: vp.x1 + dx, y0: vp.y0 + dy, y1: vp.y1 + dy }, ext, 0);
}

export function sameViewport(a: Viewport, b: Viewport, eps = 1e-9): boolean {
  return (
    Math.abs(a.x0 - b.x0) < eps && Math.abs(a.x1 - b.x1) < eps && Math.abs(a.y0 - b.y0) < eps && Math.abs(a.y1 - b.y1) < eps
  );
}

/** Integer window covering the viewport, grown by `pad` (fraction of span) each side and clamped. */
export function fetchWindow(vp: Viewport, ext: Extent, pad = 0.2) {
  const px = (vp.x1 - vp.x0) * pad;
  const py = (vp.y1 - vp.y0) * pad;
  return {
    d0: clamp(Math.floor(vp.x0 - px), 0, ext.nx - 1),
    d1: clamp(Math.ceil(vp.x1 + px), 1, ext.nx),
    t0: clamp(Math.floor(vp.y0 - py), 0, ext.ny - 1),
    t1: clamp(Math.ceil(vp.y1 + py), 1, ext.ny),
  };
}

/** Viewport that starts at (x, y) with the given spans, clamped into the extent. */
export function windowViewport(x0: number, xSpan: number, y0: number, ySpan: number, ext: Extent): Viewport {
  return clampViewport({ x0, x1: x0 + xSpan, y0, y1: y0 + ySpan }, ext, 1);
}

/** Smallest "1/2/5 x 10^k" step >= raw. */
export function niceStep(raw: number): number {
  if (raw <= 1) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 5, 10]) if (m * p >= raw) return m * p;
  return 10 * p;
}
