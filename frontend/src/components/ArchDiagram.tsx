import { useMemo } from "react";
import { BOX_H, buildDiagram, hasFlow } from "../archGraph";
import { useStore } from "../store";

/** Decoder-block diagram of the loaded model. Clicking a node opens that activation in the Layer view (the layer is kept). */
export function ArchDiagram() {
  const run = useStore((s) => s.run)!;
  const act = useStore((s) => s.act);
  const layer = useStore((s) => s.layer);
  const setAct = useStore((s) => s.setAct);
  const setMode = useStore((s) => s.setMode);
  const g = useMemo(() => buildDiagram(run), [run.model_id, run.activations]);
  const nLayers = run.model.n_layers;
  if (!hasFlow(run)) return <aside className="arch"><p className="arch-scroll">This backend does not report the block dataflow.</p></aside>;

  return (
    <aside className="arch">
      <div className="arch-scroll">
        <svg width={g.width} height={g.height} viewBox={`0 0 ${g.width} ${g.height}`} role="img" aria-label="Decoder block diagram">
          <defs>
            <marker id="arch-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto">
              <path d="M0 0L8 4L0 8z" className="arch-arrowhead" />
            </marker>
          </defs>
          <rect className="arch-block" x={4} y={g.block.y0} width={g.width - 8} height={g.block.y1 - g.block.y0} rx={10} />
          <text className="arch-blocklbl" x={g.width - 12} y={g.block.y0 + 14} textAnchor="end">
            decoder block × {nLayers} · layer {layer}
          </text>
          {g.edges.map((e, i) => (
            <path key={i} d={e.d} className="arch-edge" markerEnd={e.arrow ? "url(#arch-arrow)" : undefined} />
          ))}
          {g.nodes.map((n) => {
            const clickable = n.actId !== null;
            const cls = `arch-node${n.stream ? " stream" : ""}${clickable ? " act" : ""}${n.actId === act ? " sel" : ""}`;
            const activate = () => {
              if (!n.actId) return;
              setAct(n.actId);
              setMode("layer");
            };
            return (
              <g
                key={n.key}
                className={cls}
                transform={`translate(${n.x - n.w / 2} ${n.y - BOX_H / 2})`}
                onClick={clickable ? activate : undefined}
                onKeyDown={clickable ? (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), activate()) : undefined}
                tabIndex={clickable ? 0 : undefined}
                role={clickable ? "button" : undefined}
                aria-label={clickable ? n.title.split("\n")[0] : undefined}
                aria-pressed={clickable ? n.actId === act : undefined}
              >
                <title>{n.title}</title>
                <rect width={n.w} height={BOX_H} rx={n.stream ? BOX_H / 2 : 7} />
                <text x={n.w / 2} y={n.sub ? 14 : BOX_H / 2 + 4} textAnchor="middle" className="arch-name">{n.label}</text>
                {n.sub && <text x={n.w / 2} y={27} textAnchor="middle" className="arch-sub">{n.sub}</text>}
              </g>
            );
          })}
        </svg>
      </div>
    </aside>
  );
}
