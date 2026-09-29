import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { getFrame, getJSON, runPath, type ActivationInfo, type AttnMeta, type AttnOverviewMeta, type AttnStat, type Stats } from "../api";
import { makeColorSpec, type Display } from "../colormaps";
import { paintCells, type Cell, type Region } from "../figure";
import { Heatmap, type HeatmapHandle, type Matrix } from "../Heatmap";
import { DisplayControls, ExportButtons } from "../components/controls";
import { StatsCard } from "../components/StatsCard";
import { useMeta } from "../meta";
import { useStore } from "../store";
import { fmtToken } from "../tokens";
import { fetchWindow, type Viewport } from "../viewport";
import { renderHeadGrid } from "./attnGridExport";

type Scope = "window" | "selection" | "token" | "dim";

const MAP_DISPLAY: Display = { cmap: "viridis", range: "full", scale: "linear", symmetric: false, manual: [0, 1] };
const THUMB = 96;

/** "Layer" mode for attn_pattern: head grid of the selected layer -> single head detail, plus the layer x head map. */
export function AttentionPanel({ info }: { info: ActivationInfo }) {
  const attnStats = useMeta().attn_stats;
  const badged = attnStats.filter((s) => s.badge);
  const run = useStore((s) => s.run)!;
  const layer = useStore((s) => s.layer);
  const setLayer = useStore((s) => s.setLayer);
  const a = useStore((s) => s.attn);
  const patchAttn = useStore((s) => s.patchAttn);
  const hoverRow = useStore((s) => s.hoverRow);
  const setHoverRow = useStore((s) => s.setHoverRow);
  const detailHandle = useRef<HeatmapHandle>(null);
  const mapHandle = useRef<HeatmapHandle>(null);
  const [scope, setScope] = useState<Scope>("window");
  const [clip, setClip] = useState(false);

  const T = run.tokens.length;
  const L = info.n_layers;
  const H = info.n_heads ?? 1;

  // --- L x H overview map ---
  const ov = useQuery({
    queryKey: ["attn_overview", run.run_id, a.stat],
    queryFn: ({ signal }) => getFrame<AttnOverviewMeta>(runPath(run.run_id, "attn_overview"), { stat: a.stat }, signal),
    placeholderData: keepPreviousData,
  });
  const ovMatrix: Matrix | null = ov.data ? { values: ov.data.data, rows: L, cols: H, row0: 0, col0: 0, bh: 1, bw: 1 } : null;
  const ovExt = useMemo(() => ({ nx: H, ny: L }), [H, L]);
  const ovHome = useMemo(() => ({ x0: 0, x1: H, y0: 0, y1: L }), [H, L]);
  const [ovVp, setOvVp] = useState<Viewport>(ovHome);
  useEffect(() => setOvVp(ovHome), [ovHome]);

  // --- thumbnails for the selected layer ---
  const grid = useQuery({
    queryKey: ["attn_grid", run.run_id, layer],
    queryFn: ({ signal }) => getFrame<AttnMeta>(runPath(run.run_id, "attn"), { layer, head: -1, max_q: THUMB, max_k: THUMB, agg: "max" }, signal),
    placeholderData: keepPreviousData,
  });
  const thumbs = useMemo(() => {
    const d = grid.data;
    if (!d) return [];
    const { rows, cols } = d.meta;
    return d.meta.heads.map((h, i) => {
      const vals = d.data.subarray(i * rows * cols, (i + 1) * rows * cols);
      return { head: h, canvas: paintCells(vals, rows, cols, makeColorSpec(a.display, { min: 0, max: 1, p01: 0, p99: 1 }, vals), false) };
    });
  }, [grid.data, a.display]);
  const order = useMemo(() => {
    const idx = Array.from({ length: H }, (_, i) => i);
    const m = grid.data?.meta.metrics;
    if (a.sortBy !== "index" && m) {
      const key = m[a.sortBy];
      const up = attnStats.find((s) => s.id === a.sortBy)?.ascending;
      if (key) idx.sort((x, y) => (up ? key[x] - key[y] : key[y] - key[x]));
    }
    return idx;
  }, [H, grid.data, a.sortBy]);

  // --- detail (one head) ---
  const ext = useMemo(() => ({ nx: T, ny: T }), [T]);
  const home = useMemo(() => ({ x0: 0, x1: T, y0: 0, y1: T }), [T]);
  const win = useMemo(() => fetchWindow(a.vp, ext, 0.2), [a.vp, ext]);
  const head = a.head;
  const detail = useQuery({
    queryKey: ["attn_detail", run.run_id, layer, head, win],
    enabled: head !== null,
    queryFn: ({ signal }) =>
      getFrame<AttnMeta>(runPath(run.run_id, "attn"), { layer, head: head!, q0: win.t0, q1: win.t1, k0: win.d0, k1: win.d1, max_q: 512, max_k: 512, agg: "max" }, signal),
    placeholderData: keepPreviousData,
  });
  const matrix: Matrix | null = useMemo(() => {
    const d = detail.data;
    if (!d || head === null) return null;
    const m = d.meta;
    return { values: d.data, rows: m.rows, cols: m.cols, row0: m.q0, col0: m.k0, bh: m.bq, bw: m.bk };
  }, [detail.data, head]);

  const tokLabel = useCallback((i: number) => (i < T ? `${i} ${fmtToken(run.tokens[i], 14)}` : null), [run.tokens, T]);
  const describe = useCallback(
    (c: Cell) => {
      const d = detail.data;
      if (!d) return null;
      const m = d.meta;
      const r = Math.floor((c.row - m.q0) / m.bq);
      const k = Math.floor((c.col - m.k0) / m.bk);
      if (r < 0 || k < 0 || r >= m.rows || k >= m.cols) return null;
      const lines = [`query #${c.row} ${JSON.stringify(run.tokens[c.row])}`, `key   #${c.col} ${JSON.stringify(run.tokens[c.col])}`, `attention ${d.data[r * m.cols + k].toPrecision(4)}`];
      if (c.col > c.row) lines.push("(masked: key is in the future)");
      if (m.bq > 1 || m.bk > 1) lines.push(`${m.bq}×${m.bk} cells pooled (max)`);
      return lines;
    },
    [detail.data, run.tokens],
  );

  // --- statistics ---
  const region = useMemo(() => {
    const w = { q0: Math.floor(a.vp.y0), q1: Math.ceil(a.vp.y1), k0: Math.floor(a.vp.x0), k1: Math.ceil(a.vp.x1) };
    if (scope === "selection" && a.brush) return { q0: a.brush.y0, q1: a.brush.y1, k0: a.brush.x0, k1: a.brush.x1 };
    if (scope === "token" && a.cursor) return { q0: a.cursor.row, q1: a.cursor.row + 1, k0: 0, k1: T };
    if (scope === "dim" && a.cursor) return { q0: 0, q1: T, k0: a.cursor.col, k1: a.cursor.col + 1 };
    return w;
  }, [scope, a.brush, a.cursor, a.vp, T]);
  const stats = useQuery({
    queryKey: ["attn_stats", run.run_id, layer, head, region, clip],
    enabled: head !== null,
    queryFn: ({ signal }) => getJSON<Stats>(runPath(run.run_id, "attn_stats"), { layer, head: head!, ...region, clip }, signal),
    placeholderData: keepPreviousData,
    retry: false,
  });

  // --- exporting the thumbnail grid ---
  const gridHandle = useRef<HeatmapHandle>({ exportCanvas: () => { throw new Error("not ready"); } });
  gridHandle.current = { exportCanvas: (opts) => renderHeadGrid(opts, order, thumbs, grid.data?.meta.metrics, attnStats) };

  const layerLabel = info.layer_labels[layer] ?? String(layer);
  const metaLines = () => [`prompt: “${run.tokens.join("").replace(/\s+/g, " ").slice(0, 110)}”`];
  const selectedMetrics = head !== null && grid.data ? (() => {
    const i = grid.data.meta.heads.indexOf(head);
    const m = grid.data.meta.metrics;
    return i >= 0 ? Object.fromEntries(attnStats.map((s) => [s.id, m[s.id]?.[i]])) as Record<AttnStat, number | undefined> : null;
  })() : null;

  return (
    <div className="view">
      <div className="main">
        <div className="toolbar">
          {head !== null ? (
            <>
              <label>
                Head
                <select value={head} onChange={(e) => patchAttn({ head: Number(e.target.value), brush: null, cursor: null })}>
                  {Array.from({ length: H }, (_, i) => <option key={i} value={i}>{i}</option>)}
                </select>
              </label>
              <button onClick={() => patchAttn({ head: null, brush: null, cursor: null })}>← all heads</button>
            </>
          ) : (
            <label>
              Sort heads by
              <select value={a.sortBy} onChange={(e) => patchAttn({ sortBy: e.target.value as typeof a.sortBy })}>
                <option value="index">head index</option>
                {attnStats.map((s) => <option key={s.id} value={s.id}>{s.sort_label}</option>)}
              </select>
            </label>
          )}
          <DisplayControls display={a.display} onChange={(display) => patchAttn({ display })} />
          <ExportButtons
            handle={head === null ? gridHandle : detailHandle}
            describe={() =>
              head === null
                ? { name: `actlens_${run.model_id}_attn_L${layerLabel}_heads`, title: `${run.model_id} · attention patterns · layer ${layerLabel}`, meta: [...metaLines(), `${H} heads · query rows × key columns · ${a.display.cmap}/${a.display.range}/${a.display.scale} · pooled to ${THUMB}px (max)`] }
                : {
                    name: `actlens_${run.model_id}_attn_L${layerLabel}_H${head}`,
                    title: `${run.model_id} · attention · layer ${layerLabel} head ${head}`,
                    meta: [...metaLines(), `queries ${Math.floor(a.vp.y0)}–${Math.ceil(a.vp.y1) - 1} · keys ${Math.floor(a.vp.x0)}–${Math.ceil(a.vp.x1) - 1} · ${a.display.cmap}/${a.display.range}/${a.display.scale}`],
                  }
            }
          />
        </div>
        {head === null ? (
          <div className="grid-wrap">
            {grid.error && <div className="err">{(grid.error as Error).message}</div>}
            <div className="thumbs">
              {order.map((h) => {
                const t = thumbs.find((x) => x.head === h);
                const m = grid.data?.meta.metrics;
                return (
                  <button key={h} className="thumb" onClick={() => patchAttn({ head: h, vp: home })} title="Open this head">
                    <div className="thumb-label">
                      <b>H{h}</b>
                      {m && <span title={badged.map((s) => s.detail_label).join(" · ")}>{badged.filter((s) => m[s.id]).map((s) => `${s.badge}${m[s.id][h].toFixed(2)}`).join(" · ")}</span>}
                    </div>
                    <ThumbCanvas source={t?.canvas ?? null} />
                  </button>
                );
              })}
            </div>
            <div className="hint">Rows are query tokens, columns are key tokens. Click a head to inspect it at full resolution.</div>
          </div>
        ) : (
          <>
            <div className="figure">
              <Heatmap
                ref={detailHandle}
                matrix={matrix}
                loading={detail.isFetching}
                error={detail.error ? (detail.error as Error).message : null}
                display={a.display}
                range={{ min: 0, max: 1, p01: 0, p99: 1 }}
                viewport={a.vp}
                home={home}
                extent={ext}
                onViewport={(vp) => patchAttn({ vp })}
                rowLabel={tokLabel}
                colLabel={tokLabel}
                rowTextual
                colTextual
                rowTitle="query token"
                colTitle="key token"
                colorTitle="attention"
                describe={describe}
                brush={a.brush}
                onBrush={(brush: Region | null) => { patchAttn({ brush }); if (brush) setScope("selection"); }}
                cursor={a.cursor}
                onCursor={(cursor) => patchAttn({ cursor })}
                hoverRow={hoverRow}
                onHoverRow={setHoverRow}
              />
            </div>
            <div className="hint">drag: pan · pinch or ⌘/Ctrl + scroll: zoom · shift + drag: select · click: cursor · double-click: reset. The upper triangle is masked (causal).</div>
          </>
        )}
      </div>
      <aside className="side">
        <div className="card">
          <div className="card-title">Layer × head map</div>
          <select value={a.stat} onChange={(e) => patchAttn({ stat: e.target.value })}>
            {attnStats.map((s) => <option key={s.id} value={s.id}>{s.label}</option>)}
          </select>
          <div className="mini-map">
            <Heatmap
              ref={mapHandle}
              matrix={ovMatrix}
              loading={ov.isFetching}
              display={MAP_DISPLAY}
              range={ov.data ? { min: ov.data.meta.range.min, max: ov.data.meta.range.max, p01: ov.data.meta.range.min, p99: ov.data.meta.range.max } : null}
              viewport={ovVp}
              home={ovHome}
              extent={ovExt}
              onViewport={setOvVp}
              flipY
              rowLabel={(i) => (i < L ? info.layer_labels[i] ?? String(i) : null)}
              colLabel={(i) => (i < H ? String(i) : null)}
              rowTitle="layer"
              colTitle="head"
              colorTitle=""
              describe={(c) => {
                const v = ovMatrix && c.row < L && c.col < H ? ovMatrix.values[c.row * H + c.col] : null;
                return v === null ? null : [`layer ${c.row} · head ${c.col}`, `${a.stat} = ${v.toPrecision(4)}`];
              }}
              cursor={{ row: layer, col: head ?? -1 }}
              onCursor={(c) => { setLayer(c.row); patchAttn({ head: c.col, vp: home, brush: null, cursor: null }); }}
            />
          </div>
          <div className="muted small">Click a cell to open that head.</div>
        </div>
        {head !== null ? (
          <>
            <div className="seg wide">
              {(["window", "selection", "token", "dim"] as Scope[]).map((s) => (
                <button key={s} className={scope === s ? "on" : ""} disabled={(s === "selection" && !a.brush) || ((s === "token" || s === "dim") && !a.cursor)} onClick={() => setScope(s)}>
                  {s === "token" ? "query row" : s === "dim" ? "key column" : s}
                </button>
              ))}
            </div>
            {(a.brush || a.cursor) && <button className="small" onClick={() => { patchAttn({ brush: null, cursor: null }); setScope("window"); }}>clear selection</button>}
            {selectedMetrics && (
              <div className="card small">
                <div className="kv">
                  {attnStats.filter((s) => selectedMetrics[s.id] !== undefined).map((s) => (
                    <div key={s.id}><span>{s.detail_label}</span><b>{selectedMetrics[s.id]!.toFixed(s.digits)}</b></div>
                  ))}
                </div>
              </div>
            )}
            <StatsCard
              stats={stats.data}
              title={`Attention L${layerLabel} H${head}`}
              subtitle={`queries ${region.q0}–${region.q1 - 1} · keys ${region.k0}–${region.k1 - 1} (causal entries only)`}
              exportName={`actlens_${run.model_id}_attn_L${layerLabel}_H${head}_stats`}
              clip={clip}
              onClip={setClip}
              loading={stats.isFetching}
              error={stats.error ? (stats.error as Error).message : null}
            />
          </>
        ) : (
          <div className="card muted small">Pick a head to see its attention statistics. Entropy near 0 means a head that looks at a single token; a high sink mass means most attention lands on the first token.</div>
        )}
      </aside>
    </div>
  );
}

function ThumbCanvas({ source }: { source: HTMLCanvasElement | null }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const ctx = cv.getContext("2d")!;
    if (!source) {
      ctx.clearRect(0, 0, cv.width, cv.height);
      return;
    }
    cv.width = source.width;
    cv.height = source.height;
    ctx.drawImage(source, 0, 0);
  }, [source]);
  return <canvas ref={ref} className="thumb-canvas" />;
}
