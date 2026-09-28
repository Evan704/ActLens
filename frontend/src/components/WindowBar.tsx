import { clampHead, windowHead, type HeadLayout } from "../channels";
import type { Viewport } from "../viewport";
import { NumberBox } from "./controls";

const CHANNEL_SIZES = [32, 64, 128, 256, 512, 1024, 0];
const TOKEN_SIZES = [16, 32, 64, 128, 0];

/** Numeric window controls (token and channel ranges, size presets) plus the head jump for head-structured activations. */
export function WindowBar({
  vp, nTokens, nChannels, heads, onWindow, onSpan, onHead,
}: {
  vp: Viewport;
  nTokens: number;
  nChannels: number;
  heads: HeadLayout | null;
  onWindow: (p: Partial<Viewport>) => void;
  onSpan: (axis: "x" | "y", n: number) => void;
  onHead: (head: number) => void;
}) {
  const head = heads ? windowHead(vp, heads) : 0;
  const exactHead = heads && vp.x0 === head * heads.headDim && vp.x1 === (head + 1) * heads.headDim;
  return (
    <div className="toolbar windowbar">
      <span className="lbl">Window</span>
      <span>tokens</span>
      <NumberBox value={Math.floor(vp.y0)} min={0} max={nTokens - 1} width={56} onCommit={(y0) => onWindow({ y0 })} />
      <span>–</span>
      <NumberBox value={Math.ceil(vp.y1)} min={1} max={nTokens} width={56} onCommit={(y1) => onWindow({ y1 })} />
      <span className="presets">{TOKEN_SIZES.map((n) => <button key={n} onClick={() => onSpan("y", n)}>{n || "all"}</button>)}</span>
      <span className="sep" />
      <span>channels</span>
      <NumberBox value={Math.floor(vp.x0)} min={0} max={nChannels - 1} width={62} onCommit={(x0) => onWindow({ x0 })} />
      <span>–</span>
      <NumberBox value={Math.ceil(vp.x1)} min={1} max={nChannels} width={62} onCommit={(x1) => onWindow({ x1 })} />
      <span className="presets">{CHANNEL_SIZES.filter((n) => n === 0 || n < nChannels).map((n) => <button key={n} onClick={() => onSpan("x", n)}>{n || "all"}</button>)}</span>
      <span className="muted small">of {nTokens} × {nChannels}</span>
      {heads && (
        <span className="headjump" title={`Set the window to exactly one head's ${heads.headDim} channels (switches to natural channel order)`}>
          <span className="sep" />
          <span className="lbl">Head</span>
          <button aria-label="Previous head" onClick={() => onHead(clampHead(head - 1, heads.nHeads))} disabled={!!exactHead && head <= 0}>◀</button>
          <NumberBox value={head} min={0} max={heads.nHeads - 1} width={52} onCommit={(h) => onHead(clampHead(h, heads.nHeads))} />
          <button aria-label="Next head" onClick={() => onHead(clampHead(head + 1, heads.nHeads))} disabled={!!exactHead && head >= heads.nHeads - 1}>▶</button>
          <span className="muted small">of {heads.nHeads} × {heads.headDim}</span>
        </span>
      )}
    </div>
  );
}
