/**
 * Layout of the architecture diagram: one decoder block, drawn bottom-up like the LLM Gallery figures. The graph is
 * derived from the activations the loaded model actually exposes (QK-norm, RoPE, a gate, ... appear only when the
 * adapter provides them), so it needs no per-architecture code. Pure, so it can be unit-tested.
 */
import type { ActivationInfo, RunInfo } from "./api";

export const BOX_W = 84;
export const BOX_H = 34;
export const STREAM_W = 78;
export const PITCH = 50;
export const STREAM_X = 45;
export const COL_X = [128, 226, 324];
export const WIDTH = 372;

export interface GNode {
  key: string;
  /** Activation this node opens; null for the static embedding / head nodes. */
  actId: string | null;
  label: string;
  sub?: string;
  x: number;
  y: number;
  w: number;
  stream: boolean;
  title: string;
}

export interface Diagram {
  nodes: GNode[];
  /** SVG path data; `arrow` edges end with an arrowhead. */
  edges: { d: string; arrow: boolean }[];
  block: { y0: number; y1: number };
  height: number;
}

const has = (run: RunInfo, id: string) => run.activations.some((a) => a.id === id);
const info = (run: RunInfo, id: string) => run.activations.find((a) => a.id === id);

function sub(a: ActivationInfo | undefined): string | undefined {
  if (!a) return undefined;
  if (a.kind === "attn") return `${a.n_heads ?? "H"}×T×T`;
  return `T×${a.dim}`;
}

export function buildDiagram(run: RunInfo): Diagram {
  // rows are counted upwards from 0 (the embedding); converted to y at the end
  const pos = new Map<string, { col: number; row: number; stream: boolean }>();
  const put = (id: string, col: number, row: number, stream = false) => {
    if (id === "embed" || id === "final" || id === "head" || has(run, id)) pos.set(id, { col, row, stream });
  };
  const edges: [string, string, "src" | "dst"][] = [];
  const link = (a: string, b: string, turn: "src" | "dst" = "dst") => {
    if (pos.has(a) && pos.has(b)) edges.push([a, b, turn]);
  };

  put("embed", -1, 0, true);
  put("resid_pre", -1, 1, true);
  let row = 2;

  // attention branch
  put("attn_norm", 1, row);
  const qc = ["q", "q_norm", "q_rope"].filter((i) => has(run, i));
  const kc = ["k", "k_norm", "k_rope"].filter((i) => has(run, i));
  const L = Math.max(qc.length, kc.length, has(run, "v") ? 1 : 0);
  qc.forEach((id, i) => put(id, 0, row + 1 + i));
  kc.forEach((id, i) => put(id, 1, row + 1 + i));
  put("v", 2, row + 1);
  let top = row + L;
  if (has(run, "attn_pattern")) put("attn_pattern", 0.5, ++top);
  put("attn_ctx", 1, ++top);
  put("o", 1, ++top);
  const qLast = qc[qc.length - 1], kLast = kc[kc.length - 1];
  for (const id of ["q", "k", "v"]) link("attn_norm", id, "src");
  for (const c of [qc, kc]) for (let i = 1; i < c.length; i++) link(c[i - 1], c[i]);
  if (pos.has("attn_pattern")) {
    if (qLast) link(qLast, "attn_pattern");
    if (kLast) link(kLast, "attn_pattern");
    link("attn_pattern", "attn_ctx");
  } else if (qLast) link(qLast, "attn_ctx");
  link("v", "attn_ctx");
  link("attn_ctx", "o");
  const attnIn = pos.has("attn_norm") ? "attn_norm" : null;
  const attnOut = pos.has("o") ? "o" : null;
  row = top + 1;
  put("resid_mid", -1, row, true);

  // MLP branch
  row++;
  put("mlp_norm", 1, row);
  const gated = has(run, "gate");
  if (gated) {
    put("gate", 0, row + 1);
    put("up", 2, row + 1);
    let t = row + 1;
    if (has(run, "silu")) put("silu", 0, ++t);
    put("swiglu", 1, ++t);
    put("down", 1, ++t);
    link("mlp_norm", "gate", "src");
    link("mlp_norm", "up", "src");
    link("gate", "silu");
    link(pos.has("silu") ? "silu" : "gate", "swiglu");
    link("up", "swiglu");
    link("swiglu", "down");
    top = t;
  } else {
    const seq = ["up", "mlp_act", "down"].filter((i) => has(run, i));
    seq.forEach((id, i) => put(id, 1, row + 1 + i));
    [ "mlp_norm", ...seq].forEach((id, i, a) => i && link(a[i - 1], id));
    top = row + seq.length;
  }
  const mlpIn = pos.has("mlp_norm") ? "mlp_norm" : null;
  const mlpOut = pos.has("down") ? "down" : null;
  row = top + 1;
  put("resid_post", -1, row, true);
  const blockTop = row;
  put("final", -1, row + 1.4, true);
  put("head", -1, row + 2.4, true);
  const maxRow = row + 2.4;

  const height = (maxRow + 1) * PITCH + 8;
  const yOf = (r: number) => height - 4 - (r + 0.5) * PITCH;
  const nodes: GNode[] = [];
  const at = new Map<string, GNode>();
  const fixed: Record<string, [string, string]> = {
    embed: ["embedding", "tokens → T×D"],
    final: ["final norm", ""],
    head: ["lm_head", "→ logits"],
  };
  for (const [key, p] of pos) {
    const a = info(run, key);
    const f = fixed[key];
    const n: GNode = {
      key, actId: f ? null : key, label: f ? f[0] : key, sub: f ? f[1] || undefined : sub(a),
      x: p.col < 0 ? STREAM_X : p.col === 0.5 ? (COL_X[0] + COL_X[1]) / 2 : COL_X[p.col],
      y: yOf(p.row), w: p.stream ? STREAM_W : BOX_W, stream: p.stream,
      title: a ? `${a.label}\n${a.description}` : f ? f[0] : key,
    };
    nodes.push(n);
    at.set(key, n);
  }

  const paths: { d: string; arrow: boolean }[] = [];
  const top_ = (n: GNode) => n.y - BOX_H / 2, bot = (n: GNode) => n.y + BOX_H / 2;
  for (const [a, b, turn] of edges) {
    const s = at.get(a)!, t = at.get(b)!;
    const my = turn === "src" ? top_(s) - 8 : bot(t) + 8;
    paths.push({ d: `M${s.x} ${top_(s)}V${my}H${t.x}V${bot(t)}`, arrow: true });
  }
  // residual stream: embedding → resid_pre → resid_mid → resid_post → final norm → lm_head
  const stream = ["embed", "resid_pre", "resid_mid", "resid_post", "final", "head"].filter((k) => at.has(k));
  for (let i = 1; i < stream.length; i++) {
    const s = at.get(stream[i - 1])!, t = at.get(stream[i])!;
    paths.push({ d: `M${s.x} ${top_(s)}V${bot(t)}`, arrow: true });
  }
  // branches read from the stream just before their norm and add back into the next stream node
  const branch = (inKey: string | null, outKey: string | null, fromKey: string, toKey: string) => {
    if (inKey && at.has(fromKey)) {
      const n = at.get(inKey)!;
      paths.push({ d: `M${STREAM_X} ${bot(n) + 8}H${n.x}V${bot(n)}`, arrow: true });
    }
    if (outKey && at.has(toKey)) {
      const o = at.get(outKey)!, t = at.get(toKey)!;
      paths.push({ d: `M${o.x} ${top_(o)}V${t.y}H${STREAM_X + STREAM_W / 2}`, arrow: true });
    }
  };
  branch(attnIn, attnOut, "resid_pre", "resid_mid");
  branch(mlpIn, mlpOut, "resid_mid", "resid_post");

  return { nodes, edges: paths, block: { y0: yOf(blockTop) - PITCH / 2, y1: yOf(1) + PITCH / 2 }, height };
}
