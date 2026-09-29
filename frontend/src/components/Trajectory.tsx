import { formatNumber } from "../colormaps";

/** Small line chart of one token's statistic across layers; click to pick a layer. `caption` replaces the min/max line. */
export function Trajectory({ values, labels, active, onPick, caption }: { values: number[]; labels: string[]; active: number; onPick: (i: number) => void; caption?: (lo: number, hi: number) => string }) {
  const W = 300;
  const H = 110;
  const m = { l: 6, r: 6, t: 8, b: 16 };
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const x = (i: number) => m.l + (values.length === 1 ? 0.5 : i / (values.length - 1)) * (W - m.l - m.r);
  const y = (v: number) => m.t + (1 - (v - lo) / (hi - lo || 1)) * (H - m.t - m.b);
  return (
    <svg
      className="traj"
      viewBox={`0 0 ${W} ${H}`}
      width="100%"
      onClick={(e) => {
        const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
        const f = ((e.clientX - r.left) / r.width) * W;
        onPick(Math.max(0, Math.min(values.length - 1, Math.round(((f - m.l) / (W - m.l - m.r)) * (values.length - 1)))));
      }}
    >
      <polyline points={values.map((v, i) => `${x(i)},${y(v)}`).join(" ")} fill="none" stroke="var(--accent)" strokeWidth="1.6" />
      <line x1={x(active)} x2={x(active)} y1={m.t} y2={H - m.b} stroke="var(--fg)" strokeDasharray="3 3" opacity="0.6" />
      <circle cx={x(active)} cy={y(values[active])} r="3.5" fill="var(--accent)" />
      <text x={m.l} y={H - 3} fontSize="10" fill="var(--muted)">{labels[0]}</text>
      <text x={W - m.r} y={H - 3} fontSize="10" fill="var(--muted)" textAnchor="end">{labels[labels.length - 1]}</text>
      <text x={m.l} y={m.t + 2} fontSize="10" fill="var(--muted)">{caption ? caption(lo, hi) : `min ${formatNumber(lo)} · max ${formatNumber(hi)}`}</text>
    </svg>
  );
}
