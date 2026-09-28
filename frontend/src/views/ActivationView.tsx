import { SelectionBar } from "../components/SelectionBar";
import { useStore } from "../store";
import { findAct } from "../viewState";
import { AcrossLayersView } from "./AcrossLayersView";
import { AttentionPanel } from "./AttentionPanel";
import { TokenChannelView } from "./TokenChannelView";

/** The single view of the app: selection bar on top, then the view that matches the activation kind and mode. */
export function ActivationView() {
  const run = useStore((s) => s.run)!;
  const act = useStore((s) => s.act);
  const mode = useStore((s) => s.mode);
  const info = findAct(run, act);
  if (!info) return <div className="empty">Unknown activation “{act}”.</div>;
  return (
    <div className="activation">
      <SelectionBar />
      {info.kind === "attn" ? (
        <AttentionPanel key={info.id} info={info} />
      ) : mode === "across" ? (
        <AcrossLayersView key={`across-${info.id}`} info={info} />
      ) : (
        <TokenChannelView key={`layer-${info.id}`} info={info} />
      )}
    </div>
  );
}
