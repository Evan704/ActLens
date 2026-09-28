import { forwardRef, useCallback, useEffect, useImperativeHandle, useLayoutEffect, useMemo, useRef, useState } from "react";
import { makeColorSpec, type Display } from "./colormaps";
import type { ValueRange } from "./api";
import {
  axisAt,
  drawFigure,
  LIGHT,
  paintCells,
  type Cell,
  type CellImage,
  type FigureSpec,
  type Layout,
  type Overlay,
  type Region,
  type Theme,
} from "./figure";
import { DARK } from "./figure";

export type { Display };
import { clampViewport, pan, sameViewport, zoomAt, type Extent, type Viewport } from "./viewport";

export interface Matrix {
  values: Float32Array;
  rows: number;
  cols: number;
  row0: number;
  col0: number;
  bh: number;
  bw: number;
}

export interface ExportOptions {
  scale: number;
  title?: string;
  meta?: string[];
}

export interface HeatmapHandle {
  /** Render the current view onto a fresh canvas using the light theme, without interaction overlays. */
  exportCanvas(opts: ExportOptions): { canvas: HTMLCanvasElement; width: number; height: number };
}

interface Props {
  matrix: Matrix | null;
  loading?: boolean;
  error?: string | null;
  display: Display;
  /** Reference value range for the "robust"/"full" modes. */
  range: ValueRange | null;
  viewport: Viewport;
  home: Viewport;
  extent: Extent;
  minSpan?: number;
  onViewport: (vp: Viewport) => void;
  flipY?: boolean;
  rowLabel: (i: number) => string | null;
  colLabel: (i: number) => string | null;
  rowTextual?: boolean;
  colTextual?: boolean;
  rowTitle: string;
  colTitle: string;
  colorTitle: string;
  /** See FigureSpec.colGroupSize / colTickPx. */
  colGroupSize?: number | null;
  colTickPx?: number;
  title?: string;
  describe: (cell: Cell) => string[] | null;
  brush?: Region | null;
  onBrush?: (r: Region | null) => void;
  cursor?: Cell | null;
  onCursor?: (c: Cell) => void;
  hoverRow?: number | null;
  onHoverRow?: (row: number | null) => void;
}

const COMMIT_MS = 120;

export function useIsDark(): boolean {
  const [dark, setDark] = useState(() => window.matchMedia("(prefers-color-scheme: dark)").matches);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const h = (e: MediaQueryListEvent) => setDark(e.matches);
    mq.addEventListener("change", h);
    return () => mq.removeEventListener("change", h);
  }, []);
  return dark;
}

export const Heatmap = forwardRef<HeatmapHandle, Props>(function Heatmap(props, ref) {
  const { matrix, display, range, extent, minSpan = 2, flipY = false } = props;
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const layoutRef = useRef<Layout | null>(null);
  const [size, setSize] = useState({ w: 600, h: 400 });
  const [live, setLive] = useState(props.viewport);
  const liveRef = useRef(live);
  liveRef.current = live;
  const lastEmitted = useRef(props.viewport);
  const timer = useRef<number | undefined>(undefined);
  const [hover, setHover] = useState<Cell | null>(null);
  const [tip, setTip] = useState<{ x: number; y: number; lines: string[] } | null>(null);
  const [dragBrush, setDragBrush] = useState<Region | null>(null);
  const dragBrushRef = useRef<Region | null>(null);
  const dark = useIsDark();
  const theme: Theme = dark ? DARK : LIGHT;

  // controlled viewport: adopt external changes (numeric inputs, minimap, reset), ignore our own echoes
  useEffect(() => {
    if (!sameViewport(props.viewport, lastEmitted.current)) {
      lastEmitted.current = props.viewport;
      setLive(props.viewport);
    }
  }, [props.viewport]);

  const emit = useCallback(
    (vp: Viewport, immediate = false) => {
      setLive(vp);
      window.clearTimeout(timer.current);
      const fire = () => {
        lastEmitted.current = vp;
        props.onViewport(vp);
      };
      if (immediate) fire();
      else timer.current = window.setTimeout(fire, COMMIT_MS);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [props.onViewport],
  );
  useEffect(() => () => window.clearTimeout(timer.current), []);

  useLayoutEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setSize({ w: Math.max(200, el.clientWidth), h: Math.max(160, el.clientHeight) }));
    ro.observe(el);
    setSize({ w: Math.max(200, el.clientWidth), h: Math.max(160, el.clientHeight) });
    return () => ro.disconnect();
  }, []);

  // colours -> image
  const color = useMemo(() => makeColorSpec(display, range, matrix?.values ?? null), [display, range, matrix]);

  const image: CellImage | null = useMemo(() => {
    if (!matrix) return null;
    const cv = paintCells(matrix.values, matrix.rows, matrix.cols, color, flipY);
    return { canvas: cv, rows: matrix.rows, cols: matrix.cols, row0: matrix.row0, col0: matrix.col0, bh: matrix.bh, bw: matrix.bw };
  }, [matrix, color, flipY]);

  const spec: FigureSpec = useMemo(
    () => ({
      image,
      viewport: live,
      flipY,
      color,
      rowLabel: props.rowLabel,
      colLabel: props.colLabel,
      rowTextual: !!props.rowTextual,
      colTextual: !!props.colTextual,
      rowTitle: props.rowTitle,
      colTitle: props.colTitle,
      colorTitle: props.colorTitle,
      colGroupSize: props.colGroupSize,
      colTickPx: props.colTickPx,
      title: props.title,
    }),
    [image, live, flipY, color, props.rowLabel, props.colLabel, props.rowTextual, props.colTextual, props.rowTitle, props.colTitle, props.colorTitle, props.colGroupSize, props.colTickPx, props.title],
  );
  const overlay: Overlay = { hover, hoverRow: props.hoverRow, brush: dragBrush ?? props.brush, cursor: props.cursor };

  useLayoutEffect(() => {
    const cv = canvas.current;
    if (!cv) return;
    const dpr = window.devicePixelRatio || 1;
    const pw = Math.round(size.w * dpr);
    const ph = Math.round(size.h * dpr);
    if (cv.width !== pw) cv.width = pw;
    if (cv.height !== ph) cv.height = ph;
    layoutRef.current = drawFigure(cv.getContext("2d")!, spec, overlay, theme, size.w, size.h, dpr);
  });

  useImperativeHandle(
    ref,
    () => ({
      exportCanvas: ({ scale, title, meta }) => {
        const width = Math.max(size.w, 900);
        const height = Math.max(size.h, 520) + (meta?.length ?? 0) * 15 + (title ? 22 : 0);
        const cv = document.createElement("canvas");
        cv.width = Math.round(width * scale);
        cv.height = Math.round(height * scale);
        drawFigure(cv.getContext("2d")!, { ...spec, title: title ?? spec.title, meta }, {}, LIGHT, width, height, scale);
        return { canvas: cv, width, height };
      },
    }),
    [spec, size],
  );

  // ----- interaction -----
  const localPoint = (e: { clientX: number; clientY: number }) => {
    const r = canvas.current!.getBoundingClientRect();
    return { px: e.clientX - r.left, py: e.clientY - r.top };
  };
  const cellAt = (px: number, py: number, clampToExtent: boolean): Cell | null => {
    const L = layoutRef.current;
    if (!L) return null;
    const { plot } = L;
    if (!clampToExtent && (px < plot.x || px > plot.x + plot.w || py < plot.y || py > plot.y + plot.h)) return null;
    const a = axisAt({ viewport: liveRef.current, flipY }, plot, px, py);
    const col = Math.min(extent.nx - 1, Math.max(0, Math.floor(a.x)));
    const row = Math.min(extent.ny - 1, Math.max(0, Math.floor(a.y)));
    return { row, col };
  };

  const drag = useRef<{ x: number; y: number; vp: Viewport; brush: boolean; moved: boolean } | null>(null);

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    (e.target as Element).setPointerCapture(e.pointerId);
    const { px, py } = localPoint(e);
    drag.current = { x: px, y: py, vp: liveRef.current, brush: e.shiftKey, moved: false };
    setTip(null);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const { px, py } = localPoint(e);
    const L = layoutRef.current;
    const d = drag.current;
    if (d && L) {
      if (Math.hypot(px - d.x, py - d.y) > 3) d.moved = true;
      if (!d.moved) return;
      if (d.brush) {
        const a = axisAt({ viewport: liveRef.current, flipY }, L.plot, d.x, d.y);
        const b = axisAt({ viewport: liveRef.current, flipY }, L.plot, px, py);
        const region = {
          x0: Math.max(0, Math.min(a.x, b.x)),
          x1: Math.min(extent.nx, Math.max(a.x, b.x)),
          y0: Math.max(0, Math.min(a.y, b.y)),
          y1: Math.min(extent.ny, Math.max(a.y, b.y)),
        };
        dragBrushRef.current = region;
        setDragBrush(region);
      } else {
        const spanX = d.vp.x1 - d.vp.x0;
        const spanY = d.vp.y1 - d.vp.y0;
        const dx = -((px - d.x) / L.plot.w) * spanX;
        const dyRaw = ((py - d.y) / L.plot.h) * spanY;
        emit(pan(d.vp, extent, dx, flipY ? dyRaw : -dyRaw));
      }
      return;
    }
    const c = cellAt(px, py, false);
    setHover((h) => (h && c && h.row === c.row && h.col === c.col ? h : c));
    props.onHoverRow?.(c ? c.row : null);
    const lines = c ? props.describe(c) : null;
    setTip(lines ? { x: px, y: py, lines } : null);
  };

  const onPointerUp = (e: React.PointerEvent) => {
    const d = drag.current;
    drag.current = null;
    if (!d) return;
    const { px, py } = localPoint(e);
    if (d.brush && d.moved) {
      const b = dragBrushRef.current;
      dragBrushRef.current = null;
      setDragBrush(null);
      if (b) props.onBrush?.({ x0: Math.floor(b.x0), x1: Math.max(Math.ceil(b.x1), Math.floor(b.x0) + 1), y0: Math.floor(b.y0), y1: Math.max(Math.ceil(b.y1), Math.floor(b.y0) + 1) });
    } else if (!d.moved) {
      const c = cellAt(px, py, false);
      if (c) props.onCursor?.(c);
    }
  };

  const onPointerLeave = () => {
    if (drag.current) return;
    setHover(null);
    setTip(null);
    props.onHoverRow?.(null);
  };

  // wheel: pinch / ctrl / cmd = zoom around the cursor, plain scroll = pan. Needs a non-passive listener.
  useEffect(() => {
    const cv = canvas.current;
    if (!cv) return;
    const onWheel = (e: WheelEvent) => {
      const L = layoutRef.current;
      if (!L) return;
      e.preventDefault();
      const r = cv.getBoundingClientRect();
      const px = e.clientX - r.left;
      const py = e.clientY - r.top;
      const vp = liveRef.current;
      if (e.ctrlKey || e.metaKey) {
        const k = e.ctrlKey ? 0.015 : 0.004;
        const factor = Math.exp(Math.max(-100, Math.min(100, e.deltaY)) * k);
        const fx = Math.min(1, Math.max(0, (px - L.plot.x) / L.plot.w));
        const fyRaw = Math.min(1, Math.max(0, (py - L.plot.y) / L.plot.h));
        emit(zoomAt(vp, extent, fx, flipY ? 1 - fyRaw : fyRaw, factor, "both", minSpan));
      } else {
        const dx = (e.deltaX / L.plot.w) * (vp.x1 - vp.x0);
        const dy = (e.deltaY / L.plot.h) * (vp.y1 - vp.y0);
        emit(pan(vp, extent, dx, flipY ? -dy : dy));
      }
      setTip(null);
    };
    cv.addEventListener("wheel", onWheel, { passive: false });
    return () => cv.removeEventListener("wheel", onWheel);
  }, [emit, extent, flipY, minSpan]);

  const zoomBy = (factor: number) => emit(zoomAt(liveRef.current, extent, 0.5, 0.5, factor, "both", minSpan), true);
  const reset = () => emit(clampViewport(props.home, extent, minSpan), true);

  return (
    <div className="heatmap" ref={wrap}>
      <canvas
        ref={canvas}
        style={{ width: "100%", height: "100%", cursor: drag.current?.brush ? "crosshair" : "grab", touchAction: "none" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerLeave={onPointerLeave}
        onDoubleClick={reset}
      />
      <div className="heatmap-zoom">
        <button title="Zoom in" onClick={() => zoomBy(0.6)}>+</button>
        <button title="Zoom out" onClick={() => zoomBy(1 / 0.6)}>−</button>
        <button title="Reset view (double-click)" onClick={reset}>⟲</button>
      </div>
      {props.loading && <div className="heatmap-badge">loading…</div>}
      {props.error && <div className="heatmap-error">{props.error}</div>}
      {tip && (
        <div className="tooltip" style={{ left: Math.min(tip.x + 14, size.w - 210), top: Math.min(tip.y + 14, size.h - 20 - tip.lines.length * 16) }}>
          {tip.lines.map((l, i) => (
            <div key={i}>{l}</div>
          ))}
        </div>
      )}
    </div>
  );
});
