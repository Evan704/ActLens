import {
  interpolateBlues,
  interpolateCividis,
  interpolateGreys,
  interpolateInferno,
  interpolateMagma,
  interpolatePuOr,
  interpolateRdBu,
  interpolateViridis,
} from "d3-scale-chromatic";
import type { ValueRange } from "./api";

export type ScaleKind = "linear" | "sqrt" | "symlog";
export type RangeMode = "robust" | "window" | "full" | "manual";

export interface Colormap {
  id: string;
  label: string;
  diverging: boolean;
  fn: (t: number) => string;
}

export const COLORMAPS: Colormap[] = [
  { id: "RdBu", label: "RdBu (diverging)", diverging: true, fn: (t) => interpolateRdBu(1 - t) },
  { id: "PuOr", label: "PuOr (diverging)", diverging: true, fn: (t) => interpolatePuOr(1 - t) },
  { id: "viridis", label: "Viridis", diverging: false, fn: interpolateViridis },
  { id: "magma", label: "Magma", diverging: false, fn: interpolateMagma },
  { id: "inferno", label: "Inferno", diverging: false, fn: interpolateInferno },
  { id: "cividis", label: "Cividis", diverging: false, fn: interpolateCividis },
  { id: "blues", label: "Blues", diverging: false, fn: interpolateBlues },
  { id: "greys", label: "Greys", diverging: false, fn: interpolateGreys },
];

export const colormapById = (id: string): Colormap => COLORMAPS.find((c) => c.id === id) ?? COLORMAPS[0];

/** d3 returns "rgb(r, g, b)" or "#rrggbb". */
export function parseColor(s: string): [number, number, number] {
  const m = /rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/.exec(s);
  if (m) return [Math.round(+m[1]), Math.round(+m[2]), Math.round(+m[3])];
  const h = s.replace("#", "");
  const full = h.length === 3 ? h.replace(/./g, "$&$&") : h;
  return [parseInt(full.slice(0, 2), 16), parseInt(full.slice(2, 4), 16), parseInt(full.slice(4, 6), 16)];
}

const lutCache = new Map<string, Uint8ClampedArray>();
export function buildLut(cm: Colormap): Uint8ClampedArray {
  let lut = lutCache.get(cm.id);
  if (!lut) {
    lut = new Uint8ClampedArray(256 * 4);
    for (let i = 0; i < 256; i++) {
      const [r, g, b] = parseColor(cm.fn(i / 255));
      lut.set([r, g, b, 255], i * 4);
    }
    lutCache.set(cm.id, lut);
  }
  return lut;
}

const K = 100; // symlog strength: 2 decades of dynamic range are spread across the colormap
const fwd: Record<ScaleKind, (x: number) => number> = {
  linear: (x) => x,
  sqrt: (x) => Math.sqrt(x),
  symlog: (x) => Math.log1p(K * x) / Math.log1p(K),
};
const inv: Record<ScaleKind, (x: number) => number> = {
  linear: (x) => x,
  sqrt: (x) => x * x,
  symlog: (x) => Math.expm1(x * Math.log1p(K)) / K,
};

export interface ColorSpec {
  lut: Uint8ClampedArray;
  vmin: number;
  vmax: number;
  scale: ScaleKind;
  /** Symmetric around zero: vmin = -vmax and the colormap midpoint is 0. */
  symmetric: boolean;
}

/** Map a value to a position t in [0, 1] on the colormap. NaN stays NaN. */
export function valueToT(v: number, c: Pick<ColorSpec, "vmin" | "vmax" | "scale" | "symmetric">): number {
  if (Number.isNaN(v)) return NaN;
  const f = fwd[c.scale];
  if (c.symmetric) {
    const m = Math.max(Math.abs(c.vmax), 1e-30);
    return 0.5 + 0.5 * Math.sign(v) * f(Math.min(1, Math.abs(v) / m));
  }
  const span = c.vmax - c.vmin || 1e-30;
  return f(Math.min(1, Math.max(0, (v - c.vmin) / span)));
}

export function tToValue(t: number, c: Pick<ColorSpec, "vmin" | "vmax" | "scale" | "symmetric">): number {
  const g = inv[c.scale];
  if (c.symmetric) {
    const u = 2 * t - 1;
    return Math.sign(u) * g(Math.abs(u)) * Math.abs(c.vmax);
  }
  return c.vmin + g(t) * (c.vmax - c.vmin);
}

function percentile(sorted: Float32Array, p: number): number {
  const i = (sorted.length - 1) * p;
  const lo = Math.floor(i);
  const hi = Math.ceil(i);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo);
}

/** Value range for a colour scale. `values` is the currently displayed matrix. */
export function resolveRange(
  mode: RangeMode,
  symmetric: boolean,
  global: ValueRange | null,
  values: Float32Array | null,
  manual: [number, number],
): [number, number] {
  let lo: number;
  let hi: number;
  if (mode === "manual") {
    [lo, hi] = manual;
  } else if (mode === "window" && values && values.length) {
    // subsample so sorting stays cheap for large windows
    const step = Math.max(1, Math.floor(values.length / 20000));
    const sample = Float32Array.from({ length: Math.ceil(values.length / step) }, (_, i) => values[i * step]).filter(
      (v) => !Number.isNaN(v),
    );
    sample.sort();
    lo = percentile(sample, 0.01);
    hi = percentile(sample, 0.99);
  } else if (global) {
    [lo, hi] = mode === "full" ? [global.min, global.max] : [global.p01, global.p99];
  } else {
    [lo, hi] = [0, 1];
  }
  if (symmetric) {
    const m = Math.max(Math.abs(lo), Math.abs(hi));
    [lo, hi] = [-m, m];
  }
  if (!(hi > lo)) hi = lo + 1e-9;
  return [lo, hi];
}

export function formatNumber(v: number): string {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e5 || a < 1e-3) return v.toExponential(2);
  if (a >= 100) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  return String(Number(v.toPrecision(3)));
}

export interface Display {
  cmap: string;
  range: RangeMode;
  scale: ScaleKind;
  symmetric: boolean;
  manual: [number, number];
}

/** Resolve a Display (user settings) into concrete colour parameters for the given data. */
export function makeColorSpec(display: Display, global: ValueRange | null, values: Float32Array | null): ColorSpec {
  const cm = colormapById(display.cmap);
  const symmetric = display.symmetric && cm.diverging;
  const [vmin, vmax] = resolveRange(display.range, symmetric, global, values, display.manual);
  return { lut: buildLut(cm), vmin, vmax, scale: display.scale, symmetric };
}
