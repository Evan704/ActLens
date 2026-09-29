import { describe, expect, it } from "vitest";
import type { ActivationInfo, RunInfo } from "./api";
import { buildDiagram } from "./archGraph";

const mk = (ids: string[]): RunInfo => ({
  run_id: "r", model_id: "m", tokens: ["a"], token_ids: [1], truncated: false,
  model: { n_layers: 2, hidden_size: 8, intermediate_size: 16, n_heads: 2, n_kv_heads: 2, head_dim: 4 },
  activations: ids.map((id): ActivationInfo => ({
    id, label: id, group: "x", kind: id === "attn_pattern" ? "attn" : "token", n_layers: 2,
    dim: id === "attn_pattern" ? null : 8, layer_labels: [], n_heads: null, head_dim: null, description: "",
  })),
});
const LLAMA = ["resid_pre", "resid_mid", "resid_post", "attn_norm", "q", "k", "v", "q_norm", "k_norm", "q_rope", "k_rope",
  "attn_pattern", "attn_ctx", "o", "mlp_norm", "gate", "up", "silu", "swiglu", "down"];
const GPT2 = ["resid_pre", "resid_mid", "resid_post", "attn_norm", "q", "k", "v", "attn_pattern", "attn_ctx", "o", "mlp_norm", "up", "mlp_act", "down"];

describe("buildDiagram", () => {
  it("has a clickable node per exposed activation and no others", () => {
    for (const ids of [LLAMA, GPT2]) {
      const g = buildDiagram(mk(ids));
      expect(g.nodes.filter((n) => n.actId).map((n) => n.actId).sort()).toEqual([...ids].sort());
    }
  });
  it("draws QK-norm/RoPE and the gate only when the model has them", () => {
    const keys = buildDiagram(mk(GPT2)).nodes.map((n) => n.key);
    expect(keys).not.toContain("q_rope");
    expect(keys).not.toContain("gate");
    expect(keys).toContain("mlp_act");
  });
  it("stacks bottom-up: embedding lowest, lm_head highest, nodes never overlap", () => {
    const g = buildDiagram(mk(LLAMA));
    const y = (k: string) => g.nodes.find((n) => n.key === k)!.y;
    expect(y("embed")).toBeGreaterThan(y("resid_pre"));
    expect(y("resid_pre")).toBeGreaterThan(y("q"));
    expect(y("q")).toBeGreaterThan(y("q_rope"));
    expect(y("resid_mid")).toBeGreaterThan(y("mlp_norm") - 1);
    expect(y("resid_post")).toBeGreaterThan(y("head"));
    const boxes = g.nodes.map((n) => `${Math.round(n.x)},${Math.round(n.y)}`);
    expect(new Set(boxes).size).toBe(boxes.length);
    expect(Math.min(...g.nodes.map((n) => n.y))).toBeGreaterThan(0);
  });
});
