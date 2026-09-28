import { useCallback, useMemo, useRef } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { getFrame, runPath, type ActivationInfo, type AggKind, type OrderKind, type SliceMeta } from "../api";
import { channelLabel, channelTooltip, headLayout } from "../channels";
import { Heatmap, type HeatmapHandle, type Matrix } from "../Heatmap";
import type { Cell, Region } from "../figure";
import { DisplayControls, ExportButtons } from "../components/controls";
import { DistributionPanel, type Region4 } from "../components/DistributionPanel";
import { DimMinimap } from "../components/Minimap";
import { WindowBar } from "../components/WindowBar";
import { useStore } from "../store";
import { fmtToken } from "../tokens";
import { defaultWindow } from "../viewState";
import { clampViewport, fetchWindow, windowViewport, type Viewport } from "../viewport";

const ORDER_LABEL: Record<OrderKind, string> = {
  natural: "natural index",
  absmax: "|max| over tokens ↓",
  std: "std over tokens ↓",
  mean_abs: "|mean| over tokens ↓",
};

/** "Layer" mode for token activations: token x channel heatmap of one layer, plus the distribution panel. */
export function TokenChannelView({ info }: { info: ActivationInfo }) {
  const run = useStore((s) => s.run)!;
  const layer = useStore((s) => s.layer);
  const v = useStore((s) => s.tok);
  const patchTok = useStore((s) => s.patchTok);
  const jumpToHead = useStore((s) => s.jumpToHead);
  const pickToken = useStore((s) => s.pickToken);
  const pickChannel = useStore((s) => s.pickChannel);
  const clearSelection = useStore((s) => s.clearSelection);
  const hoverRow = useStore((s) => s.hoverRow);
  const setHoverRow = useStore((s) => s.setHoverRow);
  const handle = useRef<HeatmapHandle>(null);

  const T = run.tokens.length;
  const C = info.dim ?? 0;
  const heads = headLayout(info);
  const headDim = heads?.headDim ?? null;
  const ext = useMemo(() => ({ nx: C, ny: T }), [C, T]);
  const home = useMemo(() => defaultWindow(info, T), [info, T]);
  const layerLabel = info.layer_labels[layer] ?? String(layer);
  const shortLabel = info.label.split(" — ")[0];

  const setVp = useCallback((vp: Viewport) => patchTok({ vp }), [patchTok]);
  const win = useMemo(() => fetchWindow(v.vp, ext, 0.2), [v.vp, ext]);
  const slice = useQuery({
    queryKey: ["slice", run.run_id, info.id, layer, win, v.order, v.agg],
    queryFn: ({ signal }) =>
      getFrame<SliceMeta>(runPath(run.run_id, "slice"), { act: info.id, layer, ...win, max_h: 512, max_w: 1024, agg: v.agg, order: v.order }, signal),
    placeholderData: keepPreviousData,
  });

  const matrix: Matrix | null = useMemo(() => {
    const d = slice.data;
    if (!d) return null;
    const m = d.meta;
    return { values: d.data, rows: m.rows, cols: m.cols, row0: m.t0, col0: m.d0, bh: m.bh, bw: m.bw };
  }, [slice.data]);

  /** Original channel id at a rank column, when it is inside the loaded slice (natural order: the column itself). */
  const idOf = useCallback(
    (col: number): number | null => {
      const m = slice.data?.meta;
      const j = m ? col - m.d0 : -1;
      if (m && j >= 0 && j < m.dims.length) return m.dims[j];
      return v.order === "natural" ? col : null;
    },
    [slice.data, v.order],
  );

  const rowLabel = useCallback((i: number) => (i < T ? `${i} ${fmtToken(run.tokens[i], 16)}` : null), [run.tokens, T]);
  const colLabel = useCallback((i: number) => channelLabel(idOf(i) ?? i, headDim), [idOf, headDim]);

  const describe = useCallback(
    (c: Cell) => {
      const d = slice.data;
      if (!d) return null;
      const m = d.meta;
      const r = Math.floor((c.row - m.t0) / m.bh);
      const k = Math.floor((c.col - m.d0) / m.bw);
      if (r < 0 || k < 0 || r >= m.rows || k >= m.cols) return null;
      const id = m.dims[c.col - m.d0] ?? c.col;
      const lines = [`token #${c.row}  ${JSON.stringify(run.tokens[c.row])}`];
      lines.push(channelTooltip(id, headDim, v.order === "natural" ? undefined : c.col));
      lines.push(`value ${d.data[r * m.cols + k].toPrecision(5)}`);
      if (m.bh > 1 || m.bw > 1) lines.push(`${m.bh}×${m.bw} cells pooled (${m.agg})`);
      return lines;
    },
    [slice.data, run.tokens, v.order, headDim],
  );

  const region: Region4 = useMemo(
    () => ({ t0: Math.floor(v.vp.y0), t1: Math.ceil(v.vp.y1), d0: Math.floor(v.vp.x0), d1: Math.ceil(v.vp.x1) }),
    [v.vp],
  );
  const selection: Region4 | null = useMemo(() => (v.brush ? { t0: v.brush.y0, t1: v.brush.y1, d0: v.brush.x0, d1: v.brush.x1 } : null), [v.brush]);
  const cursor = useMemo(() => (v.cursor ? { token: v.cursor.row, channel: v.cursor.col } : null), [v.cursor]);
  const cursorChannelId = v.cursor ? (v.cursor.id ?? idOf(v.cursor.col)) : null;

  const exportDescribe = () => ({
    name: `actlens_${run.model_id}_${info.id}_L${layerLabel}_t${region.t0}-${region.t1}_d${region.d0}-${region.d1}`,
    title: `${run.model_id} · ${info.label} · layer ${layerLabel}`,
    meta: [
      `prompt: “${run.tokens.join("").replace(/\s+/g, " ").slice(0, 110)}”`,
      `tokens ${region.t0}–${region.t1 - 1} · channels ${region.d0}–${region.d1 - 1} of ${C}${heads && v.order === "natural" ? ` (heads of ${heads.headDim})` : ""} · order: ${v.order} · pooling: ${v.agg} · ${v.display.cmap}/${v.display.range}/${v.display.scale}`,
    ],
  });

  const setWindow = (p: Partial<Viewport>) => patchTok({ vp: clampViewport({ ...v.vp, ...p }, ext, 1) });
  const setSpan = (axis: "x" | "y", n: number) => {
    const full = axis === "x" ? ext.nx : ext.ny;
    const span = n === 0 ? full : Math.min(n, full);
    patchTok({ vp: axis === "x" ? windowViewport(v.vp.x0, span, v.vp.y0, v.vp.y1 - v.vp.y0, ext) : windowViewport(v.vp.x0, v.vp.x1 - v.vp.x0, v.vp.y0, span, ext) });
  };
  const grouped = heads && v.order === "natural" ? heads.headDim : null;

  return (
    <div className="view">
      <div className="main">
        <div className="toolbar">
          <label>
            Order
            <select value={v.order} onChange={(e) => patchTok({ order: e.target.value as OrderKind, brush: null, cursor: null })}>
              {(Object.keys(ORDER_LABEL) as OrderKind[]).map((o) => (
                <option key={o} value={o}>{ORDER_LABEL[o]}</option>
              ))}
            </select>
          </label>
          <label title="How cells are combined when the window has more entries than pixels">
            Pooling
            <select value={v.agg} onChange={(e) => patchTok({ agg: e.target.value as AggKind })}>
              <option value="absmax">abs-max (keeps outliers)</option>
              <option value="mean">mean</option>
              <option value="max">max</option>
              <option value="min">min</option>
            </select>
          </label>
          <DisplayControls display={v.display} onChange={(display) => patchTok({ display })} />
          <ExportButtons handle={handle} describe={exportDescribe} />
        </div>
        <WindowBar vp={v.vp} nTokens={T} nChannels={C} heads={heads} onWindow={setWindow} onSpan={setSpan} onHead={jumpToHead} />
        <DimMinimap
          runId={run.run_id} act={info.id} layer={layer} order={v.order} x0={v.vp.x0} x1={v.vp.x1} nDims={C} headDim={headDim}
          onMove={(x0) => patchTok({ vp: windowViewport(x0, v.vp.x1 - v.vp.x0, v.vp.y0, v.vp.y1 - v.vp.y0, ext) })}
        />
        <div className="figure">
          <Heatmap
            ref={handle}
            matrix={matrix}
            loading={slice.isFetching}
            error={slice.error ? (slice.error as Error).message : null}
            display={v.display}
            range={slice.data?.meta.range ?? null}
            viewport={v.vp}
            home={home}
            extent={ext}
            onViewport={setVp}
            rowLabel={rowLabel}
            colLabel={colLabel}
            rowTextual
            colGroupSize={grouped}
            colTickPx={headDim ? 54 : undefined}
            rowTitle="token"
            colTitle={
              (headDim ? "channel (head·dim)" : "channel") + (v.order === "natural" ? "" : `, ranked by ${ORDER_LABEL[v.order]}`)
            }
            colorTitle="value"
            describe={describe}
            brush={v.brush}
            onBrush={(brush: Region | null) => patchTok({ brush })}
            cursor={v.cursor}
            onCursor={(c) => patchTok({ cursor: { ...c, id: idOf(c.col) ?? undefined } })}
            hoverRow={hoverRow}
            onHoverRow={setHoverRow}
          />
        </div>
        <div className="hint">drag: pan · pinch or ⌘/Ctrl + scroll: zoom · scroll: pan · shift + drag: select region · click: place cursor · double-click: reset</div>
      </div>
      <aside className="side">
        {(v.brush || v.cursor) && (
          <div className="row small selsummary">
            {v.cursor && (
              <span className="muted">
                cursor: token #{v.cursor.row} {fmtToken(run.tokens[v.cursor.row] ?? "", 12)},{" "}
                {cursorChannelId !== null ? channelTooltip(cursorChannelId, headDim) : `channel rank ${v.cursor.col}`}
              </span>
            )}
            <button onClick={clearSelection}>clear</button>
          </div>
        )}
        <DistributionPanel
          runId={run.run_id}
          act={info.id}
          actLabel={shortLabel}
          layer={layer}
          layerLabel={layerLabel}
          nTokens={T}
          nChannels={C}
          order={v.order}
          tokens={run.tokens}
          headDim={headDim}
          window={region}
          selection={selection}
          cursor={cursor}
          cursorChannelId={cursorChannelId}
          exportName={`actlens_${run.model_id}_${info.id}_L${layerLabel}`}
          onPickToken={pickToken}
          onPickChannel={pickChannel}
        />
        {v.cursor && slice.data && (() => {
          const d = describe(v.cursor);
          return d ? <div className="card small"><div className="card-title sub">Cursor</div>{d.map((l, i) => <div key={i}>{l}</div>)}</div> : null;
        })()}
      </aside>
    </div>
  );
}
