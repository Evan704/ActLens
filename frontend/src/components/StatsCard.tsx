import type { Stats } from "../api";
import { fmtNum, fmtPct } from "../axisStats";
import { Histogram } from "./Histogram";

const PCT_ORDER = ["0.1", "1", "5", "25", "50", "75", "95", "99", "99.9"];

/** mean / std / min / max / |max| / L2 / kurtosis / skew / frac>0 / frac>3σ. */
export function SummaryTable({ stats }: { stats: Pick<Stats, "mean" | "std" | "min" | "max" | "absmax" | "l2" | "kurtosis" | "skew" | "frac_pos" | "frac_3sigma"> }) {
  const cells: [string, string, string?][] = [
    ["mean", fmtNum(stats.mean)],
    ["std", fmtNum(stats.std)],
    ["min", fmtNum(stats.min)],
    ["max", fmtNum(stats.max)],
    ["|max|", fmtNum(stats.absmax)],
    ["L2 norm", fmtNum(stats.l2)],
    ["kurtosis", fmtNum(stats.kurtosis), "excess kurtosis; 0 = Gaussian, large = heavy tails / outliers"],
    ["skew", fmtNum(stats.skew)],
    ["> 0", fmtPct(stats.frac_pos, 1), "fraction of values above zero"],
    ["> 3σ", fmtPct(stats.frac_3sigma, 2), "fraction of values more than 3 standard deviations from the mean"],
  ];
  return (
    <div className="kv">
      {cells.map(([k, v, tip]) => (
        <div key={k} title={tip}>
          <span>{k}</span>
          <b>{v}</b>
        </div>
      ))}
    </div>
  );
}

export function PercentileTable({ stats }: { stats: Pick<Stats, "percentiles"> }) {
  return (
    <div className="pcts">
      {PCT_ORDER.map((p) => (
        <div key={p}>
          <span>p{p}</span>
          <b>{fmtNum(stats.percentiles[p])}</b>
        </div>
      ))}
    </div>
  );
}

/** Stats card used by the attention view (and as a building block): summary, percentiles, histogram, optional top list. */
export function StatsCard({
  stats,
  title,
  subtitle,
  exportName,
  clip,
  onClip,
  onPickTop,
  topLabel,
  loading,
  error,
}: {
  stats: Stats | undefined;
  title: string;
  subtitle: string;
  exportName: string;
  clip: boolean;
  onClip: (v: boolean) => void;
  onPickTop?: (t: { token: number; dim: number; value: number }) => void;
  topLabel?: (t: { token: number; dim: number; value: number }) => string;
  loading?: boolean;
  error?: string | null;
}) {
  if (error) return <div className="card"><div className="err">{error}</div></div>;
  if (!stats) return <div className="card muted">{loading ? "computing…" : "no data"}</div>;
  return (
    <div className={`card ${loading ? "stale" : ""}`}>
      <div className="card-title">{title}</div>
      <div className="muted small">{subtitle} · n = {stats.n.toLocaleString()}</div>
      <SummaryTable stats={stats} />
      <PercentileTable stats={stats} />
      <Histogram
        hist={stats.hist}
        name={exportName}
        title={title}
        meta={[`${subtitle} · n = ${stats.n.toLocaleString()}`]}
        total={stats.n}
        clip={clip}
        onClip={onClip}
      />
      {stats.top && stats.top.length > 0 && (
        <>
          <div className="card-title sub">Largest |value|</div>
          <div className="top">
            {stats.top.map((t, i) => (
              <button key={i} onClick={() => onPickTop?.(t)} title="Move the cursor here">
                <span>{topLabel ? topLabel(t) : `t${t.token}, d${t.dim}`}</span>
                <b>{fmtNum(t.value)}</b>
              </button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
