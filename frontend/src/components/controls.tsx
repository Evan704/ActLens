import { useState } from "react";
import { COLORMAPS, colormapById, type RangeMode, type ScaleKind } from "../colormaps";
import type { Display } from "../colormaps";
import type { HeatmapHandle } from "../Heatmap";
import { savePdf, savePng, slug } from "../export";

export function DisplayControls({ display, onChange }: { display: Display; onChange: (d: Display) => void }) {
  const cm = colormapById(display.cmap);
  const set = (p: Partial<Display>) => onChange({ ...display, ...p });
  return (
    <div className="controls">
      <label>
        Colormap
        <select value={display.cmap} onChange={(e) => set({ cmap: e.target.value })}>
          {COLORMAPS.map((c) => (
            <option key={c.id} value={c.id}>{c.label}</option>
          ))}
        </select>
      </label>
      <label title="robust: global 1–99 percentile · window: 1–99 percentile of what is on screen · full: min–max · manual: your own limits">
        Range
        <select value={display.range} onChange={(e) => set({ range: e.target.value as RangeMode })}>
          <option value="robust">robust (p1–p99)</option>
          <option value="window">window (p1–p99)</option>
          <option value="full">full (min–max)</option>
          <option value="manual">manual</option>
        </select>
      </label>
      {display.range === "manual" && (
        <span className="manual">
          <NumberBox value={display.manual[0]} onCommit={(v) => set({ manual: [v, display.manual[1]] })} />
          <span>–</span>
          <NumberBox value={display.manual[1]} onCommit={(v) => set({ manual: [display.manual[0], v] })} />
        </span>
      )}
      <label title="symlog compresses outliers so structure stays visible next to massive activations">
        Scale
        <select value={display.scale} onChange={(e) => set({ scale: e.target.value as ScaleKind })}>
          <option value="linear">linear</option>
          <option value="sqrt">sqrt</option>
          <option value="symlog">symlog</option>
        </select>
      </label>
      <label className="check" title={cm.diverging ? "Centre the colormap on 0" : "Only for diverging colormaps"}>
        <input type="checkbox" checked={display.symmetric && cm.diverging} disabled={!cm.diverging} onChange={(e) => set({ symmetric: e.target.checked })} />
        symmetric
      </label>
    </div>
  );
}

export function NumberBox({ value, onCommit, min, max, width = 64, step }: { value: number; onCommit: (v: number) => void; min?: number; max?: number; width?: number; step?: number }) {
  const [text, setText] = useState<string | null>(null);
  const commit = () => {
    if (text === null) return;
    const v = Number(text);
    if (text.trim() !== "" && Number.isFinite(v)) onCommit(Math.min(max ?? Infinity, Math.max(min ?? -Infinity, v)));
    setText(null);
  };
  return (
    <input
      className="num"
      style={{ width }}
      type="number"
      step={step}
      value={text ?? String(Number(value.toPrecision(6)))}
      onChange={(e) => setText(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") (e.target as HTMLInputElement).blur();
        if (e.key === "Escape") setText(null);
      }}
    />
  );
}

export interface ExportMeta {
  name: string;
  title: string;
  meta: string[];
}

export function ExportButtons({ handle, describe }: { handle: React.RefObject<HeatmapHandle | null>; describe: () => ExportMeta }) {
  const [scale, setScale] = useState(3);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const run = async (kind: "png" | "pdf") => {
    if (!handle.current) return;
    setBusy(kind);
    setErr(null);
    try {
      const d = describe();
      const { canvas, width, height } = handle.current.exportCanvas({ scale, title: d.title, meta: d.meta });
      const base = slug(d.name);
      if (kind === "png") await savePng(canvas, `${base}.png`);
      else await savePdf(canvas, width, height, `${base}.pdf`, d.title);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };
  return (
    <span className="export">
      <button onClick={() => run("png")} disabled={!!busy} title="Save the current view as PNG">PNG</button>
      <button onClick={() => run("pdf")} disabled={!!busy} title="Save the current view as a one-page PDF">PDF</button>
      <select value={scale} onChange={(e) => setScale(Number(e.target.value))} title="Export resolution">
        {[1, 2, 3, 4].map((s) => (
          <option key={s} value={s}>{s}×</option>
        ))}
      </select>
      {err && <span className="err" title={err}>export failed</span>}
    </span>
  );
}

export function LayerSlider({ labels, value, onChange, name = "Layer" }: { labels: string[]; value: number; onChange: (i: number) => void; name?: string }) {
  const last = labels.length - 1;
  return (
    <div className="layer-slider">
      <span className="lbl">{name}</span>
      <button onClick={() => onChange(Math.max(0, value - 1))} disabled={value <= 0} title="Previous layer">◀</button>
      <input type="range" min={0} max={last} value={value} onChange={(e) => onChange(Number(e.target.value))} />
      <button onClick={() => onChange(Math.min(last, value + 1))} disabled={value >= last} title="Next layer">▶</button>
      <b className="layer-name">{labels[value] ?? value}</b>
      <span className="muted">/ {labels[last]}</span>
    </div>
  );
}
