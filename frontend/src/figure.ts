/**
 * Canvas rendering of a heatmap figure (plot, axes, colorbar). The same function draws the
 * interactive view and the PNG/PDF export, so what you save is what you see.
 */
import { formatNumber, tToValue, valueToT, type ColorSpec } from "./colormaps";
import { niceStep, type Viewport } from "./viewport";

export interface Theme {
  bg: string;
  plotBg: string;
  fg: string;
  muted: string;
  grid: string;
  accent: string;
}
export const LIGHT: Theme = { bg: "#ffffff", plotBg: "#e8e8ea", fg: "#16181d", muted: "#5b6270", grid: "#c9ccd3", accent: "#d9480f" };
export const DARK: Theme = { bg: "#15171c", plotBg: "#0d0e11", fg: "#e6e8ee", muted: "#9aa1b0", grid: "#3a3f4b", accent: "#ffa94d" };

/** A rasterised matrix (1 px per cell) placed on the axes: cell (r, c) starts at (row0 + r*bh, col0 + c*bw). */
export interface CellImage {
  canvas: CanvasImageSource;
  rows: number;
  cols: number;
  row0: number;
  col0: number;
  bh: number;
  bw: number;
}

export interface FigureSpec {
  image: CellImage | null;
  viewport: Viewport;
  /** Rows increase upwards (used when rows are layers). The image must already be flipped. */
  flipY: boolean;
  color: ColorSpec;
  rowLabel: (i: number) => string | null;
  colLabel: (i: number) => string | null;
  /** True if row labels are words (tokens) rather than numbers. */
  rowTextual: boolean;
  colTextual: boolean;
  rowTitle: string;
  colTitle: string;
  colorTitle: string;
  /**
   * Columns come in consecutive groups of this many (e.g. attention heads x head_dim). A faint separator is
   * drawn at every group boundary once a group is at least GROUP_MIN_PX wide.
   */
  colGroupSize?: number | null;
  /** Minimum pixels per column tick label; defaults to COL_PX_NUM. Raise it for long labels such as "h15·127". */
  colTickPx?: number;
  title?: string;
  /** Extra lines shown under the title (export only). */
  meta?: string[];
}

export interface Region {
  x0: number;
  x1: number;
  y0: number;
  y1: number;
}
export interface Cell {
  row: number;
  col: number;
}
export interface Overlay {
  hover?: Cell | null;
  hoverRow?: number | null;
  brush?: Region | null;
  cursor?: Cell | null;
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  h: number;
}
export interface Layout {
  plot: Rect;
  bar: Rect;
}

const MONO = "11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
const SANS = "12px -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Noto Sans CJK SC', sans-serif";
const BAR_W = 84; // colorbar + tick labels
const ROW_PX = 12; // min pixel height per row label
const COL_PX_NUM = 38; // min pixel width per numeric column label
const COL_PX_ROT = 12; // min pixel width per rotated column label
export const GROUP_MIN_PX = 6; // groups narrower than this get no separators

/** Axis positions of the group boundaries strictly inside (x0, x1); empty when a group is narrower than `minPx`. */
export function groupBoundaries(x0: number, x1: number, size: number, plotW: number, minPx = GROUP_MIN_PX): number[] {
  if (!(size > 0) || !(x1 > x0) || (size * plotW) / (x1 - x0) < minPx) return [];
  const out: number[] = [];
  for (let k = Math.floor(x0 / size) + 1; k * size < x1; k++) if (k * size > x0) out.push(k * size);
  return out;
}

export const screenX = (spec: Pick<FigureSpec, "viewport">, plot: Rect, x: number) =>
  plot.x + ((x - spec.viewport.x0) / (spec.viewport.x1 - spec.viewport.x0)) * plot.w;

export function screenY(spec: Pick<FigureSpec, "viewport" | "flipY">, plot: Rect, y: number) {
  const f = (y - spec.viewport.y0) / (spec.viewport.y1 - spec.viewport.y0);
  return spec.flipY ? plot.y + plot.h - f * plot.h : plot.y + f * plot.h;
}

/** Axis coordinates under a screen point. */
export function axisAt(spec: Pick<FigureSpec, "viewport" | "flipY">, plot: Rect, px: number, py: number) {
  const { x0, x1, y0, y1 } = spec.viewport;
  const fx = (px - plot.x) / plot.w;
  const fy = spec.flipY ? 1 - (py - plot.y) / plot.h : (py - plot.y) / plot.h;
  return { x: x0 + fx * (x1 - x0), y: y0 + fy * (y1 - y0) };
}

function fit(ctx: CanvasRenderingContext2D, text: string, maxW: number): string {
  if (ctx.measureText(text).width <= maxW) return text;
  const chars = Array.from(text);
  while (chars.length > 1 && ctx.measureText(chars.join("") + "…").width > maxW) chars.pop();
  return chars.join("") + "…";
}

interface Tick {
  i: number;
  label: string;
}
function ticks(lo: number, hi: number, px: number, minPx: number, label: (i: number) => string | null): Tick[] {
  const cellPx = px / (hi - lo);
  const step = Math.max(1, Math.round(niceStep(minPx / cellPx)));
  const out: Tick[] = [];
  for (let i = Math.ceil((lo - 0.5) / step) * step; i + 0.5 <= hi + 1e-9; i += step) {
    if (i + 0.5 < lo) continue;
    const l = label(i);
    if (l !== null) out.push({ i, label: l });
  }
  return out;
}

export function computeLayout(ctx: CanvasRenderingContext2D, spec: FigureSpec, width: number, height: number): Layout {
  const { viewport: vp } = spec;
  const headerH = (spec.title ? 22 : 6) + (spec.meta?.length ?? 0) * 15;
  let left = 46;
  ctx.font = MONO;
  if (spec.rowTextual) {
    // same 42px bottom margin the plot will get, so the tick step matches what is drawn later
    const ts = ticks(vp.y0, vp.y1, height - headerH - 42, ROW_PX, spec.rowLabel);
    if (ts.length) left = Math.min(150, Math.max(...ts.map((t) => ctx.measureText(t.label).width))) + 40;
  }
  let bottom = 42;
  if (spec.colTextual) {
    const ts = ticks(vp.x0, vp.x1, width - left - BAR_W, COL_PX_ROT, spec.colLabel);
    if (ts.length) bottom = Math.min(110, Math.max(...ts.map((t) => ctx.measureText(t.label).width))) + 40;
  }
  const plot: Rect = { x: left, y: headerH, w: Math.max(40, width - left - BAR_W), h: Math.max(40, height - headerH - bottom) };
  return { plot, bar: { x: plot.x + plot.w + 14, y: plot.y, w: 12, h: plot.h } };
}

export function drawFigure(
  ctx: CanvasRenderingContext2D,
  spec: FigureSpec,
  overlay: Overlay,
  theme: Theme,
  width: number,
  height: number,
  scale: number,
): Layout {
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, width, height);
  const L = computeLayout(ctx, spec, width, height);
  const { plot, bar } = L;
  const vp = spec.viewport;

  // header
  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";
  if (spec.title) {
    ctx.font = `600 13px ${SANS.split("px ")[1]}`;
    ctx.fillStyle = theme.fg;
    ctx.fillText(fit(ctx, spec.title, width - 20), 10, 16);
  }
  ctx.font = SANS;
  ctx.fillStyle = theme.muted;
  spec.meta?.forEach((m, k) => ctx.fillText(fit(ctx, m, width - 20), 10, 32 + k * 15));

  // plot area + cells
  ctx.fillStyle = theme.plotBg;
  ctx.fillRect(plot.x, plot.y, plot.w, plot.h);
  ctx.save();
  ctx.beginPath();
  ctx.rect(plot.x, plot.y, plot.w, plot.h);
  ctx.clip();
  const img = spec.image;
  if (img) {
    const xa = screenX(spec, plot, img.col0);
    const xb = screenX(spec, plot, img.col0 + img.cols * img.bw);
    const ya = screenY(spec, plot, img.row0);
    const yb = screenY(spec, plot, img.row0 + img.rows * img.bh);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(img.canvas, 0, 0, img.cols, img.rows, xa, Math.min(ya, yb), xb - xa, Math.abs(yb - ya));
  }

  if (spec.colGroupSize) {
    ctx.strokeStyle = theme.fg;
    ctx.globalAlpha = 0.3;
    ctx.lineWidth = 1;
    for (const b of groupBoundaries(vp.x0, vp.x1, spec.colGroupSize, plot.w)) {
      const x = Math.round(screenX(spec, plot, b)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(x, plot.y);
      ctx.lineTo(x, plot.y + plot.h);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  // overlays
  const cellRect = (row: number, col: number): Rect => {
    const xa = screenX(spec, plot, col);
    const xb = screenX(spec, plot, col + 1);
    const ya = screenY(spec, plot, row);
    const yb = screenY(spec, plot, row + 1);
    return { x: xa, y: Math.min(ya, yb), w: xb - xa, h: Math.abs(yb - ya) };
  };
  if (overlay.hoverRow != null) {
    const r = cellRect(overlay.hoverRow, vp.x0);
    ctx.fillStyle = theme.fg;
    ctx.globalAlpha = 0.14;
    ctx.fillRect(plot.x, r.y, plot.w, Math.max(r.h, 1));
    ctx.globalAlpha = 1;
  }
  if (overlay.brush) {
    const b = overlay.brush;
    const xa = screenX(spec, plot, b.x0);
    const xb = screenX(spec, plot, b.x1);
    const ya = screenY(spec, plot, b.y0);
    const yb = screenY(spec, plot, b.y1);
    ctx.fillStyle = theme.accent;
    ctx.globalAlpha = 0.12;
    ctx.fillRect(xa, Math.min(ya, yb), xb - xa, Math.abs(yb - ya));
    ctx.globalAlpha = 1;
    ctx.strokeStyle = theme.accent;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([5, 3]);
    ctx.strokeRect(xa, Math.min(ya, yb), xb - xa, Math.abs(yb - ya));
    ctx.setLineDash([]);
  }
  for (const [c, color, w] of [
    [overlay.cursor, theme.accent, 2],
    [overlay.hover, theme.fg, 1],
  ] as const) {
    if (!c) continue;
    const r = cellRect(c.row, c.col);
    ctx.strokeStyle = color;
    ctx.lineWidth = w;
    ctx.strokeRect(r.x + 0.5, r.y + 0.5, Math.max(r.w - 1, 1), Math.max(r.h - 1, 1));
  }
  ctx.restore();

  ctx.strokeStyle = theme.grid;
  ctx.lineWidth = 1;
  ctx.strokeRect(plot.x + 0.5, plot.y + 0.5, plot.w - 1, plot.h - 1);

  // row axis
  ctx.font = MONO;
  ctx.fillStyle = theme.muted;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const rowTicks = ticks(vp.y0, vp.y1, plot.h, ROW_PX, spec.rowLabel);
  const maxRowW = plot.x - 12;
  for (const t of rowTicks) {
    const y = screenY(spec, plot, t.i + 0.5);
    ctx.strokeStyle = theme.grid;
    ctx.beginPath();
    ctx.moveTo(plot.x - 4, y);
    ctx.lineTo(plot.x, y);
    ctx.stroke();
    ctx.fillText(fit(ctx, t.label, maxRowW), plot.x - 7, y);
  }
  // column axis
  const rotate = spec.colTextual; // word labels are always rotated; the tick step thins them out
  const colTicks = ticks(vp.x0, vp.x1, plot.w, rotate ? COL_PX_ROT : (spec.colTickPx ?? COL_PX_NUM), spec.colLabel);
  for (const t of colTicks) {
    const x = screenX(spec, plot, t.i + 0.5);
    ctx.strokeStyle = theme.grid;
    ctx.beginPath();
    ctx.moveTo(x, plot.y + plot.h);
    ctx.lineTo(x, plot.y + plot.h + 4);
    ctx.stroke();
    if (rotate) {
      ctx.save();
      ctx.translate(x, plot.y + plot.h + 8);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "right";
      ctx.textBaseline = "middle";
      ctx.fillText(fit(ctx, t.label, height - plot.y - plot.h - 30), 0, 0);
      ctx.restore();
    } else {
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(t.label, x, plot.y + plot.h + 7);
    }
  }
  // axis titles
  ctx.font = SANS;
  ctx.fillStyle = theme.fg;
  ctx.textAlign = "center";
  ctx.textBaseline = "alphabetic";
  ctx.fillText(fit(ctx, spec.colTitle, plot.w), plot.x + plot.w / 2, height - 6);
  ctx.save();
  ctx.translate(13, plot.y + plot.h / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = "middle";
  ctx.fillText(fit(ctx, spec.rowTitle, plot.h), 0, 0);
  ctx.restore();

  // colour bar
  const { lut } = spec.color;
  for (let j = 0; j < bar.h; j++) {
    const t = 1 - j / (bar.h - 1);
    const k = Math.round(t * 255) * 4;
    ctx.fillStyle = `rgb(${lut[k]},${lut[k + 1]},${lut[k + 2]})`;
    ctx.fillRect(bar.x, bar.y + j, bar.w, 1.5);
  }
  ctx.strokeStyle = theme.grid;
  ctx.strokeRect(bar.x + 0.5, bar.y + 0.5, bar.w - 1, bar.h - 1);
  ctx.font = MONO;
  ctx.fillStyle = theme.muted;
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";
  const nT = 5;
  for (let k = 0; k < nT; k++) {
    const t = k / (nT - 1);
    const y = bar.y + bar.h - t * bar.h;
    const v = tToValue(t, spec.color);
    ctx.fillText(formatNumber(Math.abs(v) < Math.abs(spec.color.vmax) * 1e-9 ? 0 : v), bar.x + bar.w + 5, Math.min(bar.y + bar.h - 5, Math.max(bar.y + 5, y)));
  }
  ctx.font = SANS;
  ctx.fillStyle = theme.fg;
  ctx.textAlign = "left";
  ctx.textBaseline = "alphabetic";
  if (spec.colorTitle) ctx.fillText(fit(ctx, spec.colorTitle, BAR_W - 6), bar.x - 2, bar.y - 6);
  return L;
}

/** Paint a matrix into a 1px-per-cell canvas. `flipY` writes rows bottom-up. */
export function paintCells(
  values: Float32Array,
  rows: number,
  cols: number,
  color: ColorSpec,
  flipY: boolean,
  target?: HTMLCanvasElement,
): HTMLCanvasElement {
  const canvas = target ?? document.createElement("canvas");
  canvas.width = cols;
  canvas.height = rows;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(cols, rows);
  const { lut } = color;
  for (let r = 0; r < rows; r++) {
    const dstRow = flipY ? rows - 1 - r : r;
    for (let c = 0; c < cols; c++) {
      const t = valueToT(values[r * cols + c], color);
      const o = (dstRow * cols + c) * 4;
      if (Number.isNaN(t)) {
        img.data[o + 3] = 0;
        continue;
      }
      const k = Math.round(t * 255) * 4;
      img.data[o] = lut[k];
      img.data[o + 1] = lut[k + 1];
      img.data[o + 2] = lut[k + 2];
      img.data[o + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}
