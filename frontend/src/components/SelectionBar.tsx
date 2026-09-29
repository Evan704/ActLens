import { actOptionLabel, groupActivations } from "../channels";
import { useStore, type Mode } from "../store";
import { findAct } from "../viewState";
import { LENS_ACTS } from "../views/LensView";
import { LayerSlider } from "./controls";

/** Activation picker, layer slider and Layer | Across layers toggle: the selection shared by every view. */
export function SelectionBar() {
  const run = useStore((s) => s.run)!;
  const act = useStore((s) => s.act);
  const layer = useStore((s) => s.layer);
  const mode = useStore((s) => s.mode);
  const setAct = useStore((s) => s.setAct);
  const setLayer = useStore((s) => s.setLayer);
  const setMode = useStore((s) => s.setMode);
  const info = findAct(run, act);
  if (!info) return null;
  const isAttn = info.kind === "attn";
  const hasLens = run.activations.some((a) => LENS_ACTS.includes(a.id));
  const modes: { id: Mode; label: string; title: string }[] = [
    { id: "layer", label: "Layer", title: isAttn ? "Attention heads of the selected layer" : "Token × channel heatmap of the selected layer" },
    {
      id: "across",
      label: "Across layers",
      title: isAttn ? "Not available for attention patterns: the layer × head map on the right shows the same idea" : "Token × layer map of a per-token statistic",
    },
    { id: "lens", label: "Logit lens", title: "Unembed the residual stream after every layer: what the model would predict at each depth" },
    { id: "arch", label: "Architecture", title: "Decoder-block diagram: click a node to open its activation" },
  ];
  return (
    <div className="selbar">
      <label className="actpick" title={info.description}>
        <span className="lbl">Activation</span>
        <select value={act} onChange={(e) => setAct(e.target.value)} aria-label="Activation">
          {groupActivations(run.activations).map((g) => (
            <optgroup key={g.group} label={g.group}>
              {g.items.map((a) => (
                <option key={a.id} value={a.id}>{actOptionLabel(a)}</option>
              ))}
            </optgroup>
          ))}
        </select>
      </label>
      <LayerSlider labels={info.layer_labels} value={layer} onChange={setLayer} />
      <div className="seg" role="group" aria-label="Mode">
        {modes.map((m) => (
          <button
            key={m.id}
            className={(isAttn && mode === "across" ? "layer" : mode) === m.id ? "on" : ""}
            disabled={(isAttn && m.id === "across") || (m.id === "lens" && !hasLens)}
            title={m.title}
            onClick={() => setMode(m.id)}
          >
            {m.label}
          </button>
        ))}
      </div>
      <span className="actdesc muted small" title={info.description}>{info.description}</span>
    </div>
  );
}
