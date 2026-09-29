import { describe, expect, it } from "vitest";
import type { ActivationInfo, RunInfo } from "./api";
import { BOX_H, BOX_W, STREAM_X, buildDiagram, hasFlow } from "./archGraph";
import flows from "./fixtures/flows.json";

// Real activation lists (with the backend's dataflow) of the tiny test model of each architecture family.
const FLOWS = flows as unknown as Record<string, Partial<ActivationInfo>[]>;
const run = (name: string): RunInfo => ({
  run_id: "r", model_id: name, tokens: ["a"], token_ids: [1], truncated: false,
  model: { n_layers: 2, hidden_size: 8, intermediate_size: 16, n_heads: 2, n_kv_heads: 2, head_dim: 4 },
  activations: FLOWS[name].map((a) => ({ n_layers: 2, layer_labels: [], label: a.id, group: "x", description: "", head_dim: null, ...a }) as ActivationInfo),
});
const node = (name: string, key: string) => buildDiagram(run(name)).nodes.find((n) => n.key === key)!;

describe("buildDiagram", () => {
  it("has a clickable node per exposed activation and no others, for every architecture", () => {
    for (const name of Object.keys(FLOWS)) {
      const g = buildDiagram(run(name));
      expect(g.nodes.filter((n) => n.actId).map((n) => n.actId).sort(), name).toEqual(FLOWS[name].map((a) => a.id).sort());
    }
  });

  it("stacks bottom-up along the dataflow: every node is above all of its inputs", () => {
    for (const name of Object.keys(FLOWS)) {
      const g = buildDiagram(run(name));
      const y = (k: string) => g.nodes.find((n) => n.key === k)!.y;
      for (const a of FLOWS[name]) for (const i of a.inputs!) expect(y(a.id!), `${name}: ${a.id} above ${i}`).toBeLessThan(y(i));
      expect(y("embed")).toBeGreaterThan(y("resid_pre"));
      expect(y("resid_post")).toBeGreaterThan(y("head"));
      expect(Math.min(...g.nodes.map((n) => n.y))).toBeGreaterThan(0);
    }
  });

  it("never overlaps two boxes and keeps every node inside the drawing", () => {
    for (const name of Object.keys(FLOWS)) {
      const g = buildDiagram(run(name));
      const boxes = g.nodes.map((n) => ({ key: n.key, l: n.x - n.w / 2, r: n.x + n.w / 2, t: n.y - BOX_H / 2, b: n.y + BOX_H / 2 }));
      for (const [i, p] of boxes.entries()) {
        expect(p.l, `${name}: ${p.key}`).toBeGreaterThanOrEqual(0);
        expect(p.r, `${name}: ${p.key}`).toBeLessThanOrEqual(g.width);
        for (const q of boxes.slice(i + 1)) {
          const apart = p.r <= q.l || q.r <= p.l || p.b <= q.t || q.b <= p.t;
          expect(apart, `${name}: ${p.key} overlaps ${q.key}`).toBe(true);
        }
      }
    }
  });

  it("draws an edge into every non-root node, and only along the API's inputs", () => {
    for (const name of Object.keys(FLOWS)) {
      const g = buildDiagram(run(name));
      const inputs = FLOWS[name].reduce((n, a) => n + a.inputs!.length, 0);
      // stream-to-stream edges are the spine, which also covers embed/final/head
      const spine = FLOWS[name].filter((a) => a.stream).length + 1;
      const streamToStream = FLOWS[name].reduce((n, a) => n + (a.stream ? a.inputs!.filter((i) => FLOWS[name].find((b) => b.id === i)!.stream).length : 0), 0);
      expect(g.edges.length, name).toBe(inputs - streamToStream + spine + 1);
    }
  });

  it("draws only what the model has: no RoPE or gate for GPT-2, no pre-norms for OLMo-2", () => {
    const keys = (name: string) => buildDiagram(run(name)).nodes.map((n) => n.key);
    expect(keys("gpt2")).not.toContain("q_rope");
    expect(keys("gpt2")).not.toContain("gate");
    expect(keys("gpt2")).toContain("mlp_act");
    expect(keys("olmo2")).not.toContain("attn_norm");
    expect(keys("olmo2")).toEqual(expect.arrayContaining(["o_norm", "down_norm", "q_norm"]));
  });

  it("sequential blocks stack the MLP above the attention; parallel blocks put them side by side", () => {
    const seq = buildDiagram(run("gpt_neox_seq"));
    const par = buildDiagram(run("gpt_neox"));
    const at = (g: typeof seq, k: string) => g.nodes.find((n) => n.key === k)!;
    expect(at(seq, "mlp_norm").y).toBeLessThan(at(seq, "o").y); // the MLP starts after attention finished
    expect(at(par, "mlp_norm").y).toBe(at(par, "attn_norm").y); // both read the block input, on the same row
    expect(at(par, "mlp_norm").x).not.toBe(at(par, "attn_norm").x);
    expect(par.nodes.some((n) => n.key === "resid_mid")).toBe(false);
    expect(par.width).toBeGreaterThan(seq.width);
  });

  it("reads the stream where the dataflow says: the parallel MLP taps the block input, not a mid stream", () => {
    const g = buildDiagram(run("gpt_neox"));
    const n = node("gpt_neox", "mlp_norm");
    // a read edge starts on the spine just under the node it feeds
    expect(g.edges.some((e) => e.d.startsWith(`M${STREAM_X} ${n.y + BOX_H / 2 + 8}H${n.x}`))).toBe(true);
  });

  it("puts the branch nodes right of the residual stream", () => {
    for (const name of Object.keys(FLOWS)) {
      for (const n of buildDiagram(run(name)).nodes.filter((m) => !m.stream)) expect(n.x - BOX_W / 2, `${name}: ${n.key}`).toBeGreaterThan(STREAM_X + 39);
    }
  });

  it("copes with a backend that reports no dataflow instead of crashing", () => {
    const old = run("llama");
    old.activations = old.activations.map(({ inputs, stream, ...a }) => a as ActivationInfo);
    expect(hasFlow(old)).toBe(false);
    expect(() => buildDiagram(old)).not.toThrow();
    expect(hasFlow(run("llama"))).toBe(true);
  });
});
