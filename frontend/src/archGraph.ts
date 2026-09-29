/**
 * Layout of the architecture diagram: one decoder block, drawn bottom-up like the LLM Gallery figures.
 *
 * The graph comes entirely from the API: every activation lists the activations it is computed from (`inputs`) and
 * whether it sits on the residual stream (`stream`). Nothing here knows an activation id or an architecture, so a
 * model with a post-norm, a sandwich norm, a parallel residual, no RoPE, ... is drawn correctly as soon as its
 * adapter exposes the right activations. Pure, so it can be unit-tested.
 *
 * Rows: a node sits one row above its highest input, so independent branches share rows. Columns: a node is centred
 * under its inputs; branches that occupy the same rows (a parallel residual) are placed side by side.
 */
import type { ActivationInfo, RunInfo } from "./api";

export const BOX_W = 84;
export const BOX_H = 34;
export const STREAM_W = 78;
export const PITCH = 50;
export const STREAM_X = 45;
const COL_X0 = 128;
const COL_W = 98;
const MARGIN = 10;

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
  width: number;
  height: number;
}

function push<K, V>(m: Map<K, V[]>, key: K, value: V) {
  const list = m.get(key);
  if (list) list.push(value);
  else m.set(key, [value]);
}

const mean = (xs: number[]) => xs.reduce((s, x) => s + x, 0) / xs.length;

/** Row of each activation: 1 + the highest row among its inputs (the block input, which has none, is row 1). */
function rowsOf(acts: ActivationInfo[]): Map<string, number> {
  const byId = new Map(acts.map((a) => [a.id, a]));
  const rows = new Map<string, number>();
  const visiting = new Set<string>();
  const row = (id: string): number => {
    const known = rows.get(id);
    if (known !== undefined) return known;
    if (visiting.has(id)) return 0; // a cycle cannot be drawn upwards; ignore the back edge
    visiting.add(id);
    const ins = (byId.get(id)?.inputs ?? []).filter((i) => byId.has(i));
    const r = ins.length ? 1 + Math.max(...ins.map(row)) : 1;
    visiting.delete(id);
    rows.set(id, r);
    return r;
  };
  acts.forEach((a) => row(a.id));
  return rows;
}

/** The branches: connected groups of non-stream activations, each in API order. */
function branchesOf(acts: ActivationInfo[]): ActivationInfo[][] {
  const inBranch = new Set(acts.filter((a) => !a.stream).map((a) => a.id));
  const root = new Map([...inBranch].map((id) => [id, id]));
  const find = (id: string): string => (root.get(id) === id ? id : find(root.get(id)!));
  for (const a of acts) for (const i of a.inputs) if (inBranch.has(a.id) && inBranch.has(i)) root.set(find(a.id), find(i));
  const groups = new Map<string, ActivationInfo[]>();
  for (const a of acts) if (inBranch.has(a.id)) push(groups, find(a.id), a);
  return [...groups.values()];
}

/** Lane (a float column index) of each activation of one branch: centred under its inputs, a lane apart in a row. */
function lanesOf(branch: ActivationInfo[], rows: Map<string, number>): Map<string, number> {
  const lane = new Map<string, number>();
  const byRow = new Map<number, ActivationInfo[]>();
  for (const a of branch) push(byRow, rows.get(a.id)!, a);
  for (const r of [...byRow.keys()].sort((p, q) => p - q)) {
    const items = byRow.get(r)!.map((a) => {
      const xs = a.inputs.filter((i) => lane.has(i)).map((i) => lane.get(i)!);
      return { id: a.id, want: xs.length ? mean(xs) : null };
    });
    items.sort((p, q) => (p.want ?? -Infinity) - (q.want ?? -Infinity)); // stable: ties keep API order
    const placed: number[] = [];
    items.forEach((it, i) => placed.push(Math.max(it.want ?? 0, i ? placed[i - 1] + 1 : -Infinity)));
    const wants = items.flatMap((it, i) => (it.want === null ? [] : [placed[i] - it.want]));
    const shift = wants.length ? mean(wants) : 0; // re-centre the row where its members wanted to be
    items.forEach((it, i) => lane.set(it.id, placed[i] - shift));
  }
  return lane;
}

/** Whether the backend reported a block dataflow (an older backend does not, and there is nothing to draw). */
export const hasFlow = (run: RunInfo) => run.activations.some((a) => (a.inputs ?? []).length > 0);

export function buildDiagram(run: RunInfo): Diagram {
  const acts = run.activations.map((a) => ({ ...a, inputs: a.inputs ?? [], stream: a.stream ?? false }));
  const rows = rowsOf(acts);
  const streamActs = acts.filter((a) => a.stream);
  const topRow = Math.max(1, ...streamActs.map((a) => rows.get(a.id)!));

  // columns: each branch gets its own lanes; branches that share rows are put next to each other
  const x = new Map<string, number>();
  const placed: { lo: number; hi: number; right: number }[] = [];
  for (const br of branchesOf(acts)) {
    const lane = lanesOf(br, rows);
    const ls = [...lane.values()];
    const r = br.map((a) => rows.get(a.id)!);
    const lo = Math.min(...r), hi = Math.max(...r), minLane = Math.min(...ls);
    const left = Math.max(0, ...placed.filter((p) => p.lo <= hi && lo <= p.hi).map((p) => p.right + 1));
    for (const [id, l] of lane) x.set(id, COL_X0 + (left + l - minLane) * COL_W);
    placed.push({ lo, hi, right: left + Math.max(...ls) - minLane });
  }
  const width = Math.max(COL_X0, ...x.values()) + BOX_W / 2 + MARGIN;

  const height = (topRow + 2.4 + 1) * PITCH + 8;
  const yOf = (row: number) => height - 4 - (row + 0.5) * PITCH;

  const nodes: GNode[] = [];
  const at = new Map<string, GNode>();
  const add = (n: GNode) => (nodes.push(n), at.set(n.key, n));
  const fixed = (key: string, label: string, sub: string | undefined, row: number) =>
    add({ key, actId: null, label, sub, x: STREAM_X, y: yOf(row), w: STREAM_W, stream: true, title: label });
  fixed("embed", "embedding", "tokens → T×D", 0);
  for (const a of acts) {
    add({
      key: a.id, actId: a.id, label: a.id, sub: a.kind === "attn" ? `${a.n_heads ?? "H"}×T×T` : `T×${a.dim}`,
      x: a.stream ? STREAM_X : x.get(a.id)!, y: yOf(rows.get(a.id)!), w: a.stream ? STREAM_W : BOX_W,
      stream: a.stream, title: `${a.label}\n${a.description}`,
    });
  }
  fixed("final", "final norm", undefined, topRow + 1.4);
  fixed("head", "lm_head", "→ logits", topRow + 2.4);

  const top = (n: GNode) => n.y - BOX_H / 2, bot = (n: GNode) => n.y + BOX_H / 2;
  const edges: Diagram["edges"] = [];
  const arrow = (d: string) => edges.push({ d, arrow: true });
  const fanOut = new Map<string, number>();
  for (const a of acts) for (const i of a.inputs) fanOut.set(i, (fanOut.get(i) ?? 0) + 1);

  // the residual stream is a spine through the stream nodes in display order; branches tap it and add back into it
  const spine = ["embed", ...streamActs.map((a) => a.id), "final", "head"];
  for (let i = 1; i < spine.length; i++) {
    const s = at.get(spine[i - 1])!, t = at.get(spine[i])!;
    arrow(`M${s.x} ${top(s)}V${bot(t)}`);
  }
  for (const a of acts) {
    const t = at.get(a.id)!;
    for (const s of a.inputs.map((i) => at.get(i))) {
      if (!s || (s.stream && t.stream)) continue; // stream-to-stream is the spine
      if (s.stream) arrow(`M${STREAM_X} ${bot(t) + 8}H${t.x}V${bot(t)}`);
      else if (t.stream) arrow(`M${s.x} ${top(s)}V${t.y}H${STREAM_X + STREAM_W / 2}`);
      else if (s.x === t.x) arrow(`M${s.x} ${top(s)}V${bot(t)}`);
      else {
        const my = (fanOut.get(s.key) ?? 0) > 1 ? top(s) - 8 : bot(t) + 8; // bend near the source when it fans out
        arrow(`M${s.x} ${top(s)}V${my}H${t.x}V${bot(t)}`);
      }
    }
  }

  const rowOf = (a: ActivationInfo | undefined, fallback: number) => (a ? rows.get(a.id)! : fallback);
  return {
    nodes, edges, width, height,
    block: { y0: yOf(rowOf(streamActs.at(-1), topRow)) - PITCH / 2, y1: yOf(rowOf(streamActs[0], 1)) + PITCH / 2 },
  };
}
