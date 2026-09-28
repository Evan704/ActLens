import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { Stats } from "../api";
import { formatNumber } from "../colormaps";
import { DARK, LIGHT, type Theme } from "../figure";
import { savePng, slug } from "../export";
import { useIsDark } from "../Heatmap";
import { barHeight, barX, binAtX, histMargins, outsideNote, plotHeight, plotWidth, valueToX, yTicks } from "./histogramGeometry";

const MONO = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
const SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Noto Sans CJK SC', sans-serif";

export type HistData = Stats["hist"];
export interface HistMarker {
  value: number;
  kind: "zero" | "mean" | "selected";
}

interface DrawOpts {
  hist: HistData;
  logY: boolean;
  theme: Theme;
  w: number;
  h: number;
  scale: number;
  xLabel?: string;
  /** export only: bold title + muted lines above the plot */
  title?: string;
  meta?: string[];
  hover?: number | null;
  markers?: HistMarker[];
}

function draw(ctx: CanvasRenderingContext2D, o: DrawOpts) {
  const { hist, logY, theme, w, h, hover } = o;
  ctx.setTransform(o.scale, 0, 0, o.scale, 0, 0);
  ctx.fillStyle = theme.bg;
  ctx.fillRect(0, 0, w, h);
  const headerLines = o.title ? 1 + (o.meta?.length ?? 0) : 0;
  const m = histMargins({ headerLines, xLabel: !!o.xLabel });
  const pw = plotWidth(w, m);
  const ph = plotHeight(h, m);
  const { counts, edges } = hist;
  const maxCount = Math.max(1, ...counts);
  const lo = edges[0];
  const hi = edges[edges.length - 1];

  ctx.textBaseline = "alphabetic";
  ctx.textAlign = "left";
  if (o.title) {
    ctx.font = `600 13px ${SANS}`;
    ctx.fillStyle = theme.fg;
    ctx.fillText(o.title, 8, 18);
    ctx.font = `12px ${SANS}`;
    ctx.fillStyle = theme.muted;
    o.meta?.forEach((line, k) => ctx.fillText(line, 8, 34 + k * 15));
  }

  // y grid + labels
  ctx.font = MONO;
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  let lastLabelY = Infinity;
  for (const t of yTicks(maxCount, logY)) {
    const yy = m.t + ph - t.frac * ph;
    ctx.strokeStyle = theme.grid;
    ctx.globalAlpha = 0.5;
    ctx.beginPath();
    ctx.moveTo(m.l, yy);
    ctx.lineTo(m.l + pw, yy);
    ctx.stroke();
    ctx.globalAlpha = 1;
    // skip labels that would collide with the previous one (log axis: 0 and 1 sit close together)
    const ly = Math.min(m.t + ph - 5, Math.max(m.t + 5, yy));
    if (lastLabelY - ly < 12 && lastLabelY !== Infinity) continue;
    lastLabelY = ly;
    ctx.fillStyle = theme.muted;
    ctx.fillText(formatNumber(t.count), m.l - 5, ly);
  }

  // bars
  counts.forEach((c, i) => {
    const { x, w: bw } = barX(i, edges, lo, hi, m.l, pw);
    const bh = barHeight(c, maxCount, logY, ph);
    ctx.fillStyle = i === hover ? theme.fg : theme.accent;
    ctx.globalAlpha = i === hover ? 0.9 : 0.85;
    ctx.fillRect(x, m.t + ph - bh, bw, bh);
    ctx.globalAlpha = 1;
  });

  // markers (zero line, mean)
  ctx.save();
  ctx.beginPath();
  ctx.rect(m.l, m.t, pw, ph);
  ctx.clip();
  for (const mk of o.markers ?? []) {
    if (!(mk.value >= lo && mk.value <= hi)) continue;
    const x = Math.round(valueToX(mk.value, lo, hi, m.l, pw)) + 0.5;
    ctx.strokeStyle = mk.kind === "selected" ? (theme === DARK ? "#74c0fc" : "#1971c2") : mk.kind === "mean" ? theme.fg : theme.muted;
    ctx.lineWidth = mk.kind === "zero" ? 1 : mk.kind === "mean" ? 1.5 : 2;
    ctx.setLineDash(mk.kind === "zero" ? [3, 3] : []);
    ctx.beginPath();
    ctx.moveTo(x, m.t);
    ctx.lineTo(x, m.t + ph);
    ctx.stroke();
  }
  ctx.restore();
  ctx.setLineDash([]);
  ctx.lineWidth = 1;

  ctx.strokeStyle = theme.grid;
  ctx.strokeRect(m.l + 0.5, m.t + 0.5, pw - 1, ph - 1);

  // x axis labels
  ctx.font = MONO;
  ctx.fillStyle = theme.muted;
  ctx.textBaseline = "top";
  for (let k = 0; k <= 4; k++) {
    const v = lo + (k / 4) * (hi - lo);
    ctx.textAlign = k === 0 ? "left" : k === 4 ? "right" : "center";
    ctx.fillText(formatNumber(v), valueToX(v, lo, hi, m.l, pw), m.t + ph + 6);
  }
  if (o.xLabel) {
    ctx.font = `12px ${SANS}`;
    ctx.fillStyle = theme.fg;
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    ctx.fillText(o.xLabel, m.l + pw / 2, h - 5);
  }
  return m;
}

export interface HistogramProps {
  hist: HistData;
  /** file-name stem for the PNG export */
  name: string;
  /** title of the exported PNG */
  title: string;
  /** extra lines under the title in the exported PNG */
  meta?: string[];
  /** x axis title (shown in the panel and in the export) */
  xLabel?: string;
  /** total number of values summarised (for the hover percentage); defaults to the sum of the counts */
  total?: number;
  markers?: HistMarker[];
  /** legend text of the `selected` marker */
  selectedLabel?: string;
  /** log-count state; uncontrolled (default on) unless `logY` and `onLogY` are given */
  logY?: boolean;
  onLogY?: (v: boolean) => void;
  /** clip toggle (adds a checkbox and an "outside" note when given) */
  clip?: boolean;
  onClip?: (v: boolean) => void;
  height?: number;
}

export function Histogram(p: HistogramProps) {
  const { hist, markers } = p;
  const ref = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const [logLocal, setLogLocal] = useState(true);
  const logY = p.logY ?? logLocal;
  const setLogY = p.onLogY ?? setLogLocal;
  const height = p.height ?? (p.xLabel ? 176 : 160);
  const [size, setSize] = useState({ w: 320, h: height });
  const [hover, setHover] = useState<number | null>(null);
  const dark = useIsDark();
  const total = p.total ?? hist.counts.reduce((a, b) => a + b, 0) + hist.n_clipped;

  useEffect(() => {
    const el = wrap.current!;
    const upd = () => setSize({ w: Math.max(160, el.clientWidth), h: height });
    const ro = new ResizeObserver(upd);
    ro.observe(el);
    upd();
    return () => ro.disconnect();
  }, [height]);

  useLayoutEffect(() => {
    const cv = ref.current!;
    const dpr = window.devicePixelRatio || 1;
    cv.width = Math.round(size.w * dpr);
    cv.height = Math.round(size.h * dpr);
    draw(cv.getContext("2d")!, { hist, logY, theme: dark ? DARK : LIGHT, w: size.w, h: size.h, scale: dpr, xLabel: p.xLabel, hover, markers });
  }, [hist, logY, size, dark, hover, p.xLabel, markers]);

  // a new histogram may have a different number of bins
  useEffect(() => setHover((h) => (h !== null && h >= hist.counts.length ? null : h)), [hist]);

  const onMove = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    const m = histMargins({ xLabel: !!p.xLabel });
    setHover(binAtX(e.clientX - r.left, size.w, m, hist.counts.length));
  };

  const save = async () => {
    const cv = document.createElement("canvas");
    const s = 3;
    const w = 720;
    const h = 380;
    const meta = [...(p.meta ?? []), `${logY ? "log" : "linear"} count${p.clip ? " · clipped to p0.1–p99.9" : ""}`];
    cv.width = w * s;
    cv.height = h * s;
    draw(cv.getContext("2d")!, { hist, logY, theme: LIGHT, w, h, scale: s, xLabel: p.xLabel ?? "value", title: p.title, meta, markers });
    await savePng(cv, `${slug(p.name)}_hist.png`);
  };

  const hasClip = p.onClip !== undefined;
  return (
    <div className="hist" ref={wrap}>
      <canvas ref={ref} style={{ width: size.w, height: size.h }} onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
      {hover !== null && hist.counts[hover] !== undefined && (
        <div className="hist-tip">
          [{formatNumber(hist.edges[hover])}, {formatNumber(hist.edges[hover + 1])}) · {hist.counts[hover].toLocaleString()}
          {total > 0 && <span className="muted"> · {((hist.counts[hover] / total) * 100).toFixed(hist.counts[hover] / total < 0.001 ? 3 : 1)}%</span>}
        </div>
      )}
      <div className="hist-tools">
        <label className="check" title="Logarithmic count axis, so rare outlier bins stay visible">
          <input type="checkbox" checked={logY} onChange={(e) => setLogY(e.target.checked)} /> log count
        </label>
        {hasClip && (
          <label className="check" title="Restrict the histogram range to the 0.1–99.9 percentile so outliers don't flatten it">
            <input type="checkbox" checked={!!p.clip} onChange={(e) => p.onClip!(e.target.checked)} /> clip p0.1–p99.9
          </label>
        )}
        {markers && markers.length > 0 && (
          <span className="hist-legend muted">
            {markers.some((m) => m.kind === "zero") && (
              <>
                <i className="m-zero" /> 0
              </>
            )}
            {markers.some((m) => m.kind === "mean") && (
              <>
                <i className="m-mean" /> mean
              </>
            )}
            {markers.some((m) => m.kind === "selected") && (
              <>
                <i className="m-selected" /> {p.selectedLabel ?? "selected"}
              </>
            )}
          </span>
        )}
        <button onClick={save} title="Save histogram as PNG (light theme, high resolution)">
          PNG
        </button>
      </div>
      {hasClip && p.clip && (
        <div className="muted small hist-note">
          range [{formatNumber(hist.edges[0])}, {formatNumber(hist.edges[hist.edges.length - 1])}] · {outsideNote(hist.n_clipped, total)}
        </div>
      )}
    </div>
  );
}
