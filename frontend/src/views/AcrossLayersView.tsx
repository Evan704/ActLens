import { useCallback, useMemo, useRef } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { getFrame, runPath, type ActivationInfo, type OverviewMeta, type OverviewStat } from "../api";
import { channelLabel, headLayout } from "../channels";
import { formatNumber } from "../colormaps";
import { Heatmap, type HeatmapHandle, type Matrix } from "../Heatmap";
import type { Cell } from "../figure";
import { DisplayControls, ExportButtons, NumberBox } from "../components/controls";
import { Trajectory } from "../components/Trajectory";
import { useStore } from "../store";
import { fmtToken } from "../tokens";
import { ACT_DISPLAY, SEQ_DISPLAY } from "../viewState";
import type { Viewport } from "../viewport";

export const STAT_LABEL: Record<OverviewStat, string> = {
  norm: "L2 norm",
  absmax: "|max|",
  mean: "mean",
  std: "std",
  kurtosis: "excess kurtosis",
  dim: "single channel",
};
const SIGNED: OverviewStat[] = ["mean", "dim"];

/** "Across layers" mode: token x layer map of a per-token statistic of the selected activation. */
export function AcrossLayersView({ info }: { info: ActivationInfo }) {
  const run = useStore((s) => s.run)!;
  const layer = useStore((s) => s.layer);
  const o = useStore((s) => s.across);
  const cursorRow = useStore((s) => s.tok.cursor?.row ?? null);
  const patchAcross = useStore((s) => s.patchAcross);
  const setLayer = useStore((s) => s.setLayer);
  const openCell = useStore((s) => s.openCell);
  const hoverRow = useStore((s) => s.hoverRow);
  const setHoverRow = useStore((s) => s.setHoverRow);
  const handle = useRef<HeatmapHandle>(null);

  const T = run.tokens.length;
  const C = info.dim ?? 0;
  const headDim = headLayout(info)?.headDim ?? null;
  const ext = useMemo(() => ({ nx: info.n_layers, ny: T }), [info.n_layers, T]);
  const home: Viewport = useMemo(() => ({ x0: 0, x1: ext.nx, y0: 0, y1: ext.ny }), [ext]);

  const q = useQuery({
    queryKey: ["overview", run.run_id, info.id, o.stat, o.channel],
    queryFn: ({ signal }) => getFrame<OverviewMeta>(runPath(run.run_id, "overview"), { act: info.id, stat: o.stat, dim: o.channel }, signal),
    placeholderData: keepPreviousData,
  });

  // backend returns [layers, tokens]; we draw tokens as rows so labels sit on the left, like the layer view
  const matrix: Matrix | null = useMemo(() => {
    const d = q.data;
    if (!d) return null;
    const { rows: L, cols: n } = d.meta;
    const out = new Float32Array(L * n);
    for (let l = 0; l < L; l++) for (let t = 0; t < n; t++) out[t * L + l] = d.data[l * n + t];
    return { values: out, rows: n, cols: L, row0: 0, col0: 0, bh: 1, bw: 1 };
  }, [q.data]);

  const rowLabel = useCallback((i: number) => (i < T ? `${i} ${fmtToken(run.tokens[i], 16)}` : null), [run.tokens, T]);
  const colLabel = useCallback((i: number) => info.layer_labels[i] ?? null, [info.layer_labels]);
  const statText = o.stat === "dim" ? `channel ${channelLabel(o.channel, headDim)}` : STAT_LABEL[o.stat];

  const valueAt = useCallback((c: Cell) => (matrix && c.row < matrix.rows && c.col < matrix.cols ? matrix.values[c.row * matrix.cols + c.col] : null), [matrix]);
  const describe = useCallback(
    (c: Cell) => {
      const v = valueAt(c);
      if (v === null) return null;
      return [`token #${c.row}  ${JSON.stringify(run.tokens[c.row])}`, `layer ${info.layer_labels[c.col]}`, `${statText} = ${v.toPrecision(5)}`, "click: open this layer"];
    },
    [valueAt, run.tokens, info.layer_labels, statText],
  );

  const setStat = (stat: OverviewStat) => patchAcross({ stat, display: SIGNED.includes(stat) ? ACT_DISPLAY : SEQ_DISPLAY });
  const tokenRow = hoverRow ?? cursorRow;
  const trajectory = useMemo(() => {
    if (!matrix || tokenRow === null || tokenRow >= matrix.rows) return null;
    return Array.from(matrix.values.subarray(tokenRow * matrix.cols, (tokenRow + 1) * matrix.cols));
  }, [matrix, tokenRow]);
  const active = Math.min(layer, info.n_layers - 1);

  return (
    <div className="view">
      <div className="main">
        <div className="toolbar">
          <label>
            Statistic per token
            <select value={o.stat} onChange={(e) => setStat(e.target.value as OverviewStat)}>
              {(Object.keys(STAT_LABEL) as OverviewStat[]).map((s) => (
                <option key={s} value={s}>{STAT_LABEL[s]}</option>
              ))}
            </select>
          </label>
          {o.stat === "dim" && (
            <label>
              channel <NumberBox value={o.channel} min={0} max={C - 1} width={64} onCommit={(channel) => patchAcross({ channel: Math.round(channel) })} />
              {headDim && <span className="muted small">{channelLabel(o.channel, headDim)}</span>}
            </label>
          )}
          <DisplayControls display={o.display} onChange={(display) => patchAcross({ display })} />
          <ExportButtons
            handle={handle}
            describe={() => ({
              name: `actlens_${run.model_id}_across_${info.id}_${o.stat}${o.stat === "dim" ? o.channel : ""}`,
              title: `${run.model_id} · ${info.label} · ${statText} per token and layer`,
              meta: [`prompt: “${run.tokens.join("").replace(/\s+/g, " ").slice(0, 110)}”`, `${T} tokens × ${info.n_layers} layers · ${o.display.cmap}/${o.display.range}/${o.display.scale}`],
            })}
          />
        </div>
        <div className="figure">
          <Heatmap
            ref={handle}
            matrix={matrix}
            loading={q.isFetching}
            error={q.error ? (q.error as Error).message : null}
            display={o.display}
            range={q.data?.meta.range ?? null}
            viewport={o.vp}
            home={home}
            extent={ext}
            onViewport={(vp) => patchAcross({ vp })}
            rowLabel={rowLabel}
            colLabel={colLabel}
            rowTextual
            rowTitle="token"
            colTitle="layer"
            colorTitle={o.stat === "dim" ? "value" : STAT_LABEL[o.stat]}
            describe={describe}
            cursor={cursorRow !== null ? { row: cursorRow, col: active } : null}
            onCursor={(c) => openCell(c.col, c.row)}
            hoverRow={hoverRow}
            onHoverRow={setHoverRow}
          />
        </div>
        <div className="hint">Each cell is one token at one layer, summarised over the channels. Click a cell to open that layer with the token cursor placed.</div>
      </div>
      <aside className="side">
        {trajectory && tokenRow !== null ? (
          <div className="card">
            <div className="card-title">Token #{tokenRow} {fmtToken(run.tokens[tokenRow], 20)}</div>
            <div className="muted small">{statText} across layers</div>
            <Trajectory values={trajectory} labels={info.layer_labels} active={active} onPick={setLayer} />
            <div className="kv">
              <div><span>layer</span><b>{info.layer_labels[active]}</b></div>
              <div><span>value</span><b>{formatNumber(trajectory[active])}</b></div>
            </div>
            <div className="muted small">Click the chart to change the selected layer; click a heatmap cell to open it.</div>
          </div>
        ) : (
          <div className="card muted">Hover a token row to see it across layers. Click a cell to open that layer.</div>
        )}
      </aside>
    </div>
  );
}
