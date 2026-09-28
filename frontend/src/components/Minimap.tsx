import { useEffect, useLayoutEffect, useRef, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { getFrame, runPath, type OrderKind, type ActId, type ProfileMeta } from "../api";
import { DARK, LIGHT, groupBoundaries } from "../figure";
import { useIsDark } from "../Heatmap";

/** Per-dimension |max| over all tokens, with the current window drawn on top. Drag to move the window. */
export function DimMinimap({
  runId, act, layer, order, x0, x1, nDims, headDim, onMove,
}: {
  runId: string; act: ActId; layer: number; order: OrderKind; x0: number; x1: number; nDims: number;
  /** Draw head boundaries (natural order only). */
  headDim?: number | null;
  onMove: (newX0: number) => void;
}) {
  const wrap = useRef<HTMLDivElement>(null);
  const cv = useRef<HTMLCanvasElement>(null);
  const [w, setW] = useState(600);
  const dark = useIsDark();
  const H = 44;

  useEffect(() => {
    const el = wrap.current!;
    const ro = new ResizeObserver(() => setW(Math.max(100, el.clientWidth)));
    ro.observe(el);
    setW(Math.max(100, el.clientWidth));
    return () => ro.disconnect();
  }, []);

  const q = useQuery({
    queryKey: ["profile", runId, act, layer, order, w],
    queryFn: ({ signal }) => getFrame<ProfileMeta>(runPath(runId, "profile"), { act, layer, order, bins: w }, signal),
    placeholderData: keepPreviousData,
  });

  useLayoutEffect(() => {
    const c = cv.current!;
    const t = dark ? DARK : LIGHT;
    const dpr = window.devicePixelRatio || 1;
    c.width = Math.round(w * dpr);
    c.height = Math.round(H * dpr);
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = t.plotBg;
    ctx.fillRect(0, 0, w, H);
    const d = q.data;
    if (d) {
      const v = d.data;
      const mx = Math.max(...v, 1e-9);
      const bw = w / v.length;
      ctx.fillStyle = t.muted;
      for (let i = 0; i < v.length; i++) {
        const hgt = (Math.log1p((v[i] / mx) * 100) / Math.log1p(100)) * (H - 4);
        ctx.fillRect(i * bw, H - hgt, Math.max(1, bw - 0.2), hgt);
      }
    }
    if (headDim && order === "natural") {
      ctx.strokeStyle = t.fg;
      ctx.globalAlpha = 0.3;
      ctx.lineWidth = 1;
      for (const b of groupBoundaries(0, nDims, headDim, w, 4)) {
        const x = Math.round((b / nDims) * w) + 0.5;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }
    const xa = (x0 / nDims) * w;
    const xb = (x1 / nDims) * w;
    ctx.fillStyle = t.accent;
    ctx.globalAlpha = 0.22;
    ctx.fillRect(xa, 0, Math.max(2, xb - xa), H);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = t.accent;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(xa + 0.75, 0.75, Math.max(2, xb - xa) - 1.5, H - 1.5);
  }, [q.data, w, x0, x1, nDims, dark, headDim, order]);

  const dragging = useRef(false);
  const move = (e: React.PointerEvent) => {
    const r = cv.current!.getBoundingClientRect();
    const center = ((e.clientX - r.left) / r.width) * nDims;
    onMove(center - (x1 - x0) / 2);
  };

  return (
    <div className="minimap" ref={wrap} title="Per-channel |max| over all tokens (log). Drag to move the window.">
      <canvas
        ref={cv}
        style={{ width: w, height: H }}
        onPointerDown={(e) => { dragging.current = true; (e.target as Element).setPointerCapture(e.pointerId); move(e); }}
        onPointerMove={(e) => dragging.current && move(e)}
        onPointerUp={() => (dragging.current = false)}
      />
    </div>
  );
}
