/**
 * Distribution panel for token activations.
 *
 * Three modes: raw *values* (`/stats`), a per-*channel* statistic or a per-*token* statistic (`/axis_stats`).
 * The per-channel / per-token modes also show the values *inside* the cursor's channel / token (`/stats` again).
 */
import { useEffect, useMemo, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import type { OrderKind, Stats } from "../api";
import { statOfId, useMeta } from "../meta";
import {
  REGION_SCOPES,
  VALUE_SCOPES,
  axisStatLabel,
  axisStatTitle,
  axisStatsKey,
  channelName,
  clippedShare,
  effectiveScope,
  fetchAxisStats,
  fetchValueStats,
  fmtNum,
  insideRegion,
  keepSameAxis,
  regionText,
  resolveAxisRegion,
  resolveValueRegion,
  scopeAvailable,
  scopeDisabledReason,
  scopeText,
  statOf,
  tokenLabel,
  topLabel,
  valueStatsKey,
  type AxisKind,
  type AxisStat,
  type AxisStatsResponse,
  type RegionScope,
  type ScopeContext,
  type ValueScope,
} from "../axisStats";
import { Histogram, type HistMarker } from "./Histogram";
import { PercentileTable, SummaryTable } from "./StatsCard";
import "./distribution.css";

export interface Region4 {
  t0: number;
  t1: number;
  d0: number;
  d1: number;
}

export interface DistributionPanelProps {
  runId: string;
  act: string;
  actLabel: string;
  layer: number;
  layerLabel: string;
  nTokens: number;
  nChannels: number;
  order: OrderKind;
  tokens: string[];
  headDim: number | null;
  window: Region4;
  selection: Region4 | null;
  cursor: { token: number; channel: number } | null;
  cursorChannelId: number | null;
  exportName: string;
  onPickToken: (token: number) => void;
  onPickChannel: (rank: number) => void;
}

type Mode = "values" | AxisKind;
const MODES: { id: Mode; label: string; title: string }[] = [
  { id: "values", label: "Values", title: "Histogram of the raw activation values" },
  { id: "channel", label: "Per channel", title: "One number per channel (computed over tokens), then its distribution across channels" },
  { id: "token", label: "Per token", title: "One number per token (computed over channels), then its distribution across tokens" },
];

const TOP_N = 10;

/** `title` attribute text for a stat/scope chip. */
function chipTitle(title: string, disabledReason: string | null): string {
  return disabledReason ? `${title} — ${disabledReason}` : title;
}

function meanMarkers(s: Pick<Stats, "mean">): HistMarker[] {
  return [{ value: 0, kind: "zero" }, ...(Number.isFinite(s.mean) ? [{ value: s.mean, kind: "mean" as const }] : [])];
}

export function DistributionPanel(props: DistributionPanelProps) {
  const { runId, act, actLabel, layer, layerLabel, nTokens, nChannels, order, tokens, headDim, cursor, cursorChannelId } = props;
  const meta = useMeta();
  const [mode, setMode] = useState<Mode>("values");
  const [valueScope, setValueScope] = useState<ValueScope>("window");
  const [axisScope, setAxisScope] = useState<RegionScope>("window");
  const [stat, setStat] = useState<AxisStat>("absmax");
  const [clip, setClip] = useState(false);
  const [logY, setLogY] = useState(true);

  const ctx: ScopeContext = { nTokens, nChannels, window: props.window, selection: props.selection, cursor };

  // a scope whose prerequisite vanished (selection cleared, cursor removed) resets to the window
  useEffect(() => {
    if (!scopeAvailable(valueScope, ctx)) setValueScope("window");
    if (!scopeAvailable(axisScope, ctx)) setAxisScope("window");
  });
  const vScope = effectiveScope(valueScope, ctx);
  const aScope = effectiveScope(axisScope, ctx);

  const valuesRegion = resolveValueRegion(vScope, ctx)!;
  const axisRegion = resolveAxisRegion(aScope, ctx)!;
  const axis: AxisKind = mode === "values" ? "channel" : mode;

  const valuesQ = useQuery({
    queryKey: valueStatsKey({ runId, act, layer, region: valuesRegion, order, clip }),
    queryFn: ({ signal }) => fetchValueStats({ runId, act, layer, region: valuesRegion, order, clip }, signal),
    enabled: mode === "values",
    placeholderData: keepPreviousData,
  });
  const axisQuery = { runId, act, layer, axis, stat, region: axisRegion, order, clip, top: TOP_N };
  const axisQ = useQuery<AxisStatsResponse, Error, AxisStatsResponse, readonly unknown[]>({
    queryKey: axisStatsKey(axisQuery),
    queryFn: ({ signal }) => fetchAxisStats(axisQuery, signal),
    enabled: mode !== "values",
    placeholderData: keepSameAxis<AxisStatsResponse>(axis),
  });

  const q = mode === "values" ? valuesQ : axisQ;
  const stats = q.data as (Stats | AxisStatsResponse) | undefined;
  const fetching = q.isFetching;
  const errMsg = q.error ? (q.error as Error).message : null;

  const title = `${actLabel} · layer ${layerLabel}`;
  const scopeLine = useMemo(() => {
    const o = { cursor, cursorChannelId, tokens, headDim };
    return mode === "values" ? scopeText(vScope, o) : scopeText(aScope, o);
  }, [mode, vScope, aScope, cursor, cursorChannelId, tokens, headDim]);

  return (
    <div className={`card dist ${fetching && stats ? "is-stale" : ""}`} data-testid="distribution-panel" data-mode={mode}>
      <div className="dist-head">
        <div className="card-title">Distribution</div>
        <span className="dist-busy muted small">{fetching ? (stats ? "updating…" : "computing…") : ""}</span>
      </div>

      <div className="dist-seg" role="tablist" aria-label="Distribution mode">
        {MODES.map((m) => (
          <button key={m.id} role="tab" aria-selected={mode === m.id} className={mode === m.id ? "on" : ""} title={m.title} onClick={() => setMode(m.id)}>
            {m.label}
          </button>
        ))}
      </div>

      {mode === "values" ? (
        <div className="dist-chips" role="group" aria-label="Scope">
          <span className="dist-chips-label">scope</span>
          {VALUE_SCOPES.map((s) => {
            const ok = scopeAvailable(s.id, ctx);
            return (
              <button key={s.id} className={vScope === s.id ? "on" : ""} disabled={!ok} title={chipTitle(s.title, ok ? null : scopeDisabledReason(s.id))} onClick={() => setValueScope(s.id)}>
                {s.label}
              </button>
            );
          })}
        </div>
      ) : (
        <>
          <div className="dist-chips" role="group" aria-label="Statistic">
            <span className="dist-chips-label">stat</span>
            {meta.stats.map((s) => (
              <button key={s.id} className={stat === s.id ? "on" : ""} title={s.title} onClick={() => setStat(s.id)}>
                {s.label}
              </button>
            ))}
          </div>
          <div className="dist-chips" role="group" aria-label="Region">
            <span className="dist-chips-label">over</span>
            {REGION_SCOPES.map((s) => {
              const ok = scopeAvailable(s.id, ctx);
              return (
                <button key={s.id} className={aScope === s.id ? "on" : ""} disabled={!ok} title={chipTitle(s.title, ok ? null : scopeDisabledReason(s.id))} onClick={() => setAxisScope(s.id)}>
                  {s.label}
                </button>
              );
            })}
          </div>
        </>
      )}

      {errMsg ? (
        <div className="dist-error">
          <div className="err">{errMsg}</div>
          <button onClick={() => q.refetch()}>retry</button>
        </div>
      ) : !stats ? (
        <div className="muted dist-loading">{fetching ? "computing…" : "no data"}</div>
      ) : mode === "values" ? (
        <ValuesBody
          stats={stats as Stats}
          title={title}
          scopeLine={scopeLine}
          region={valuesRegion}
          {...props}
          clip={clip}
          onClip={setClip}
          logY={logY}
          onLogY={setLogY}
          scope={vScope}
        />
      ) : (
        <AxisBody
          stats={stats as AxisStatsResponse}
          title={title}
          scopeLine={scopeLine}
          {...props}
          clip={clip}
          onClip={setClip}
          logY={logY}
          onLogY={setLogY}
          scope={aScope}
        />
      )}
    </div>
  );
}

type BodyBase = DistributionPanelProps & {
  title: string;
  scopeLine: string;
  clip: boolean;
  onClip: (v: boolean) => void;
  logY: boolean;
  onLogY: (v: boolean) => void;
};

function ValuesBody(p: BodyBase & { stats: Stats; region: Region4; scope: ValueScope }) {
  const { stats, region } = p;
  const rt = regionText(stats.region ? (stats.region as unknown as Region4) : region, p.order, p.headDim);
  const single = stats.n <= 1;
  const clipShare = clippedShare(stats.hist.n_clipped, stats.n);
  return (
    <>
      <div className="muted small dist-sub" title={p.title}>
        {p.scopeLine} · {rt} · n = {stats.n.toLocaleString()}
        {p.scope === "channel" && p.cursorChannelId !== null && <> · {channelName(p.cursorChannelId, p.headDim)}</>}
      </div>
      <SummaryTable stats={stats} />
      <PercentileTable stats={stats} />
      {single ? (
        <div className="muted small">a single value ({fmtNum(stats.mean)}); no distribution to show</div>
      ) : (
        <Histogram
          hist={stats.hist}
          name={`${p.exportName}_values_${p.scope}`}
          title={`${p.title} — values`}
          meta={[`scope: ${p.scopeLine}`, `${rt} · n = ${stats.n.toLocaleString()}`, `mean ${fmtNum(stats.mean)} · std ${fmtNum(stats.std)} · kurtosis ${fmtNum(stats.kurtosis)}`]}
          xLabel="activation value"
          total={stats.n}
          markers={meanMarkers(stats)}
          logY={p.logY}
          onLogY={p.onLogY}
          clip={p.clip}
          onClip={p.onClip}
        />
      )}
      {p.clip && clipShare > 0.05 && <div className="muted small">Note: more than 5% of the values lie outside the clipped range.</div>}
      {stats.top && stats.top.length > 0 && (
        <>
          <div className="dist-h">Largest |value|</div>
          <div className="dist-top">
            {stats.top.map((t, i) => (
              <button key={i} onClick={() => p.onPickToken(t.token)} title="Move the cursor to this token">
                <span className="dist-top-main">
                  {tokenLabel(t.token, p.tokens, 10)}
                  <span className="muted"> · ch {t.dim}</span>
                </span>
                <b>{fmtNum(t.value)}</b>
              </button>
            ))}
          </div>
        </>
      )}
    </>
  );
}

function AxisBody(p: BodyBase & { stats: AxisStatsResponse; scope: RegionScope }) {
  const { stats } = p;
  const axis = stats.axis;
  const what = axis === "channel" ? "channels" : "tokens";
  const rt = regionText(stats.region, p.order, p.headDim);
  const statInfos = useMeta().stats;
  const statName = axisStatLabel(stats.stat, statInfos);
  const st = axisStatTitle(axis, stats.stat, statInfos);
  const maxAbs = Math.max(1e-30, ...stats.top.map((t) => Math.abs(t.value)));
  const cursorIndex = p.cursor ? (axis === "channel" ? p.cursor.channel : p.cursor.token) : null;
  const tooFew = stats.n <= 1;
  const reduced = axis === "channel" ? stats.region.t1 - stats.region.t0 : stats.region.d1 - stats.region.d0;

  // the cursor's own channel / token: its values (the ones the statistic summarises) as a second histogram
  const inner = insideRegion(axis, stats.region, p.cursor);
  const innerQuery = inner && { runId: p.runId, act: p.act, layer: p.layer, region: inner, order: p.order, clip: p.clip };
  const innerQ = useQuery({
    queryKey: innerQuery ? valueStatsKey(innerQuery) : ["stats", "inside-off"],
    queryFn: ({ signal }) => fetchValueStats(innerQuery!, signal),
    enabled: innerQuery !== null,
    placeholderData: keepPreviousData,
  });
  const innerStats = inner ? innerQ.data : undefined;
  const marker = innerStats ? statOf(stats.stat, innerStats) : null;
  const overallMarkers: HistMarker[] | undefined = (() => {
    const base = stats.min < 0 ? meanMarkers(stats) : [];
    const all = marker !== null && Number.isFinite(marker) ? [...base, { value: marker, kind: "selected" as const }] : base;
    return all.length > 0 ? all : undefined;
  })();
  const innerName = axis === "channel" ? (p.cursorChannelId !== null ? channelName(p.cursorChannelId, p.headDim) : `channel rank ${p.cursor?.channel}`) : p.cursor ? tokenLabel(p.cursor.token, p.tokens, 16) : "";
  return (
    <>
      <div className="muted small dist-sub" title={p.title}>
        {stats.n.toLocaleString()} {what} · {statName} over {axis === "channel" ? "tokens" : "channels"} · {p.scopeLine} · {rt}
      </div>
      {reduced <= 1 && (
        <div className="muted small">
          Each {axis} is summarised from a single {axis === "channel" ? "token" : "channel"}; widen the region for a meaningful {statName}.
        </div>
      )}
      {tooFew ? (
        <div className="muted small">only one {axis} in this region; no distribution to show</div>
      ) : (
        <Histogram
          hist={stats.hist}
          name={`${p.exportName}_${axis}_${stats.stat}_${p.scope}`}
          title={`${p.title} — ${st}`}
          meta={[`region: ${p.scopeLine}`, `${rt} · n = ${stats.n.toLocaleString()} ${what}`, `${stats.stat}: mean ${fmtNum(stats.mean)} · std ${fmtNum(stats.std)} · max ${fmtNum(stats.max)}`]}
          xLabel={`${statName} per ${axis}`}
          total={stats.n}
          markers={overallMarkers}
          selectedLabel={`selected ${axis}`}
          logY={p.logY}
          onLogY={p.onLogY}
          clip={p.clip}
          onClip={p.onClip}
        />
      )}
      <div className="dist-inside" data-testid="inside-panel">
        <div className="dist-h">
          Inside the selected {axis}
          {inner && <span className="muted"> · {innerName}</span>}
        </div>
        {!p.cursor ? (
          <div className="muted small">Click a cell on the heatmap (or an entry of the list below) to see the distribution of the values inside that {axis}.</div>
        ) : !inner ? (
          <div className="muted small">
            The cursor's {axis} lies outside this region; choose the “layer” region, or click a {axis} inside it.
          </div>
        ) : innerQ.error ? (
          <div className="dist-error">
            <div className="err">{(innerQ.error as Error).message}</div>
            <button onClick={() => innerQ.refetch()}>retry</button>
          </div>
        ) : !innerStats ? (
          <div className="muted small">computing…</div>
        ) : (
          <>
            <div className="muted small dist-sub">
              n = {innerStats.n.toLocaleString()} {axis === "channel" ? "tokens" : "channels"} · {regionText(inner, p.order, p.headDim)} {marker !== null && ` · ${statName} = ${fmtNum(marker)}`}
            </div>
            {innerStats.n <= 1 ? (
              <div className="muted small">a single value ({fmtNum(innerStats.mean)}); no distribution to show</div>
            ) : (
              <>
                <Histogram
                  hist={innerStats.hist}
                  name={`${p.exportName}_${axis}_inside_${p.cursor[axis]}`}
                  title={`${p.title} — inside ${innerName}`}
                  meta={[`${innerName}`, `${regionText(inner, p.order, p.headDim)} · n = ${innerStats.n.toLocaleString()}`, `mean ${fmtNum(innerStats.mean)} · std ${fmtNum(innerStats.std)} · kurtosis ${fmtNum(innerStats.kurtosis)}`]}
                  xLabel="activation value"
                  total={innerStats.n}
                  markers={meanMarkers(innerStats)}
                  logY={p.logY}
                  onLogY={p.onLogY}
                  clip={p.clip}
                  onClip={p.onClip}
                />
                <details className="dist-details">
                  <summary>Summary of the values inside this {axis}</summary>
                  <SummaryTable stats={innerStats} />
                  <PercentileTable stats={innerStats} />
                </details>
              </>
            )}
          </>
        )}
      </div>
      <div className="dist-h">
        Top {what} by {statOfId(statInfos, stats.stat)?.signed ? `|${stats.stat}|` : statName}
        <span className="muted"> · click to {axis === "channel" ? "jump to the channel" : "move the cursor"}</span>
      </div>
      <div className="dist-top" data-testid="top-list">
        {stats.top.map((t) => {
          const l = topLabel(axis, t, p.tokens, p.headDim);
          const tip = axis === "channel" ? `${channelName(t.id, p.headDim)}${l.sub ? ` · ${l.sub}` : ""}` : `token ${t.index}: ${JSON.stringify(p.tokens[t.index] ?? "")}`;
          return (
            <button
              key={`${t.index}`}
              className={cursorIndex === t.index ? "on" : ""}
              title={tip}
              onClick={() => (axis === "channel" ? p.onPickChannel(t.index) : p.onPickToken(t.index))}
            >
              <span className="dist-top-bar" style={{ width: `${(Math.abs(t.value) / maxAbs) * 100}%` }} />
              <span className="dist-top-main">
                {l.main}
                {l.sub && <span className="muted"> · {l.sub}</span>}
              </span>
              <b>{fmtNum(t.value)}</b>
            </button>
          );
        })}
      </div>
      <details className="dist-details">
        <summary>Summary of the {statName} values across {what}</summary>
        <SummaryTable stats={stats} />
        <PercentileTable stats={stats} />
      </details>
    </>
  );
}
