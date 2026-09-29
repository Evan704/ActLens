import type { AttnMeta } from "../api";
import { LIGHT } from "../figure";
import type { AttnStatInfo } from "../meta";
import type { ExportOptions } from "../Heatmap";

/** Render the head thumbnail grid (in display order) onto a fresh light-theme canvas for PNG/PDF export. */
export function renderHeadGrid(
  { scale, title, meta }: ExportOptions,
  order: number[],
  thumbs: { head: number; canvas: HTMLCanvasElement }[],
  metrics: AttnMeta["metrics"] | undefined,
  statInfos: AttnStatInfo[],
): { canvas: HTMLCanvasElement; width: number; height: number } {
  const cols = 4;
  const cell = 180;
  const gap = 10;
  const header = 22 + (meta?.length ?? 0) * 15 + 8;
  const rowsN = Math.ceil(order.length / cols);
  const width = cols * (cell + gap) + gap;
  const height = header + rowsN * (cell + 20 + gap) + gap;
  const cv = document.createElement("canvas");
  cv.width = Math.round(width * scale);
  cv.height = Math.round(height * scale);
  const ctx = cv.getContext("2d")!;
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  ctx.fillStyle = LIGHT.bg;
  ctx.fillRect(0, 0, width, height);
  ctx.fillStyle = LIGHT.fg;
  ctx.font = "600 13px -apple-system, 'PingFang SC', sans-serif";
  ctx.fillText(title ?? "", gap, 16);
  ctx.font = "12px -apple-system, 'PingFang SC', sans-serif";
  ctx.fillStyle = LIGHT.muted;
  meta?.forEach((m, i) => ctx.fillText(m, gap, 32 + i * 15));
  ctx.imageSmoothingEnabled = false;
  const byHead = new Map(thumbs.map((t) => [t.head, t.canvas]));
  order.forEach((h, i) => {
    const x = gap + (i % cols) * (cell + gap);
    const y = header + Math.floor(i / cols) * (cell + 20 + gap);
    ctx.fillStyle = LIGHT.fg;
    ctx.font = "11px ui-monospace, Menlo, monospace";
    ctx.fillText(`H${h}` + (metrics ? statInfos.filter((s) => s.badge && metrics[s.id]).map((s) => `  ${s.tag} ${metrics[s.id][h].toFixed(2)}`).join("") : ""), x, y + 12);
    const c = byHead.get(h);
    if (c) ctx.drawImage(c, 0, 0, c.width, c.height, x, y + 18, cell, cell);
    ctx.strokeStyle = LIGHT.grid;
    ctx.strokeRect(x + 0.5, y + 18.5, cell - 1, cell - 1);
  });
  return { canvas: cv, width, height };
}
