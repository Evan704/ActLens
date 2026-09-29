import { useMemo, useState } from "react";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { getJSON, runPath, type LogitLens } from "../api";
import { formatNumber } from "../colormaps";
import { NumberBox } from "../components/controls";
import { Trajectory } from "../components/Trajectory";
import { useStore } from "../store";
import { fmtToken } from "../tokens";

export const LENS_ACTS = ["resid_pre", "resid_mid", "resid_post"];
type TargetMode = "next" | "none" | "custom";

/** Percentage that stays short for vanishing probabilities, so the tracked column keeps a bounded width. */
const fmtPct = (p: number) => (p >= 1e-4 ? `${(p * 100).toPrecision(2)}%` : "<0.01%");

/** Logit lens: the residual stream of one token position, unembedded after every layer (top-k table + target rank). */
export function LensView() {
  const run = useStore((s) => s.run)!;
  const layer = useStore((s) => s.layer);
  const setLayer = useStore((s) => s.setLayer);
  const cursorRow = useStore((s) => s.tok.cursor?.row ?? null);
  const T = run.tokens.length;
  const acts = run.activations.filter((a) => LENS_ACTS.includes(a.id));
  const [act, setAct] = useState(acts.find((a) => a.id === "resid_post")?.id ?? acts[0].id);
  const [pos, setPos] = useState(Math.min(cursorRow ?? T - 1, T - 1));
  const [k, setK] = useState(8);
  const [tmode, setTmode] = useState<TargetMode>("next");
  const [custom, setCustom] = useState(0);

  const target = tmode === "custom" ? custom : tmode === "next" && pos + 1 < T ? run.token_ids[pos + 1] : undefined;
  const q = useQuery({
    queryKey: ["lens", run.run_id, act, pos, k, target],
    queryFn: ({ signal }) => getJSON<LogitLens>(runPath(run.run_id, "logit_lens"), { act, pos, k, target }, signal),
    placeholderData: keepPreviousData,
  });
  const d = q.data;
  const L = d?.ids.length ?? 0;
  const active = Math.min(layer, Math.max(0, L - 1));
  const labels = d?.layer_labels ?? [];
  const targetText = target === undefined ? null : (tmode === "next" ? fmtToken(run.tokens[pos + 1], 16) : `id ${target}`);
  const ranks = d?.target?.rank;
  const rankTraj = useMemo(() => ranks?.map((r) => -Math.log10(r)), [ranks]); // log scale, rank 1 at the top
  const firstTop1 = ranks?.findIndex((r) => r === 1) ?? -1;

  return (
    <div className="view">
      <div className="main">
        <div className="toolbar">
          <label>
            Stream
            <select value={act} onChange={(e) => setAct(e.target.value)}>
              {acts.map((a) => <option key={a.id} value={a.id}>{a.id}</option>)}
            </select>
          </label>
          <label>
            top-k
            <select value={k} onChange={(e) => setK(Number(e.target.value))}>
              {[3, 5, 8, 12, 20].map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          </label>
          <label title="Track a token's probability and rank at every layer">
            Track
            <select value={tmode} onChange={(e) => setTmode(e.target.value as TargetMode)}>
              <option value="next">next prompt token</option>
              <option value="custom">token id…</option>
              <option value="none">nothing</option>
            </select>
          </label>
          {tmode === "custom" && <NumberBox value={custom} min={0} width={80} onCommit={(v) => setCustom(Math.round(v))} />}
        </div>
        <div className="lens-toks" role="group" aria-label="Token position">
          {run.tokens.map((t, i) => (
            <button key={i} className={i === pos ? "on" : ""} title={`token #${i}`} onClick={() => setPos(i)}>{fmtToken(t, 10)}</button>
          ))}
        </div>
        <div className="lens-table">
          {q.error ? (
            <div className="heatmap-error">{(q.error as Error).message}</div>
          ) : !d ? (
            <div className="muted" style={{ padding: 12 }}>Computing…</div>
          ) : (
            <table>
              <thead>
                <tr>
                  <th style={{ textAlign: "right" }}>layer</th>
                  {d.target && <th className="tgt">{targetText}</th>}
                  {Array.from({ length: k }, (_, r) => <th key={r}>#{r + 1}</th>)}
                </tr>
              </thead>
              <tbody>
                {d.ids.map((ids, l) => (
                  <tr key={l} className={l === active ? "active" : ""}>
                    <td className="layer" onClick={() => setLayer(l)}>{labels[l]}</td>
                    {d.target && (
                      <td className="lens-tgt" title={`rank ${d.target.rank[l]} · p = ${Math.exp(d.target.logprob[l]).toPrecision(3)}`}>
                        rank <b>{d.target.rank[l]}</b> · {fmtPct(Math.exp(d.target.logprob[l]))}
                      </td>
                    )}
                    {ids.map((id, r) => {
                      const p = Math.exp(d.logprobs[l][r]);
                      return (
                        <td key={r}>
                          <div
                            className={"lens-cell" + (id === d.target?.id ? " target" : "")}
                            style={{ "--p": (p * 0.8).toFixed(3) } as React.CSSProperties}
                            title={`${JSON.stringify(d.tokens[l][r])}  id ${id}  p = ${p.toPrecision(3)}`}
                            onClick={() => setLayer(l)}
                          >
                            <span>{fmtToken(d.tokens[l][r], 12)}</span><small>{(p * 100).toFixed(p < 0.1 ? 1 : 0)}%</small>
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
        <div className="hint">
          Each row unembeds the {act} of token #{pos} {fmtToken(run.tokens[pos], 14)} after that layer with the model's final norm and output head. Early layers are often noise: the residual stream is not yet in the space the head reads. Outlined cells are the tracked token.
        </div>
      </div>
      <aside className="side">
        {d && (
          <div className="card">
            <div className="card-title">Token #{pos} {fmtToken(run.tokens[pos], 20)}</div>
            {rankTraj && d.target ? (
              <>
                <div className="muted small">Rank of {targetText} across layers (top = 1)</div>
                <Trajectory
                  values={rankTraj}
                  labels={labels}
                  active={active}
                  onPick={setLayer}
                  caption={(lo, hi) => `rank ${Math.round(10 ** -hi)} (best) · ${Math.round(10 ** -lo)} (worst), log scale`}
                />
                <div className="kv">
                  <div><span>layer</span><b>{labels[active]}</b></div>
                  <div><span>rank</span><b>{d.target.rank[active]}</b></div>
                  <div><span>p</span><b>{formatNumber(Math.exp(d.target.logprob[active]))}</b></div>
                  <div><span>first top-1</span><b>{firstTop1 < 0 ? "never" : labels[firstTop1]}</b></div>
                </div>
              </>
            ) : (
              <div className="muted small">{tmode === "next" && pos + 1 >= T ? "The last token has no next prompt token to track." : "No token tracked."}</div>
            )}
            <div className="card-title sub">Entropy across layers</div>
            <div className="muted small">Uncertainty of the next-token distribution (nats)</div>
            <Trajectory values={d.entropy} labels={labels} active={active} onPick={setLayer} />
            <div className="kv">
              <div><span>entropy</span><b>{formatNumber(d.entropy[active])}</b></div>
              <div><span>top-1</span><b>{fmtToken(d.tokens[active][0], 12)}</b></div>
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}
