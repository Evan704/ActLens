import { describe, expect, it } from "vitest";
import type { ActivationInfo } from "./api";
import { actOptionLabel, channelHead, channelLabel, channelTooltip, clampHead, groupActivations, headLayout, headWindow, windowHead } from "./channels";
import { groupBoundaries } from "./figure";

const act = (p: Partial<ActivationInfo>): ActivationInfo => ({
  id: "x", label: "x", group: "Attention", kind: "token", n_layers: 28, dim: 1024, layer_labels: [], n_heads: null, head_dim: null, description: "", stream: false, inputs: [], ...p,
});
const q = act({ id: "q", label: "q — q_proj", dim: 2048, n_heads: 16, head_dim: 128 });
const ext = { nx: 2048, ny: 40 };

describe("channel naming", () => {
  it("splits a channel id into head and dim", () => {
    expect(channelHead(401, 128)).toEqual({ head: 3, dim: 17 });
    expect(channelHead(0, 128)).toEqual({ head: 0, dim: 0 });
    expect(channelHead(2047, 128)).toEqual({ head: 15, dim: 127 });
  });
  it("labels ticks h{head}·{d} only for head-structured acts", () => {
    expect(channelLabel(401, 128)).toBe("h3·17");
    expect(channelLabel(401, null)).toBe("401");
  });
  it("tooltip reads 'head 3, dim 17 (channel 401)'", () => {
    expect(channelTooltip(401, 128)).toBe("head 3, dim 17 (channel 401)");
    expect(channelTooltip(401, null)).toBe("channel 401");
    expect(channelTooltip(401, 128, 5)).toBe("head 3, dim 17 (channel 401)  (rank 5)");
  });
});

describe("head jump", () => {
  const layout = headLayout(q)!;
  it("headLayout is null for non-head activations and attention", () => {
    expect(layout).toEqual({ nHeads: 16, headDim: 128 });
    expect(headLayout(act({}))).toBeNull();
    expect(headLayout(act({ kind: "attn", dim: null, n_heads: 16 }))).toBeNull();
  });
  it("sets the window to exactly one head and keeps the tokens", () => {
    const vp = headWindow({ x0: 5, x1: 90, y0: 3, y1: 30 }, 3, layout, ext);
    expect(vp).toEqual({ x0: 384, x1: 512, y0: 3, y1: 30 });
  });
  it("clamps out-of-range heads", () => {
    expect(headWindow({ x0: 0, x1: 128, y0: 0, y1: 8 }, 99, layout, ext).x0).toBe(15 * 128);
    expect(headWindow({ x0: 0, x1: 128, y0: 0, y1: 8 }, -4, layout, ext).x0).toBe(0);
    expect(clampHead(2.6, 16)).toBe(3);
  });
  it("uses head_dim of k/v heads (nKV) for their own extent", () => {
    const k = headLayout(act({ id: "k", dim: 1024, n_heads: 8, head_dim: 128 }))!;
    expect(headWindow({ x0: 0, x1: 128, y0: 0, y1: 8 }, 7, k, { nx: 1024, ny: 40 })).toMatchObject({ x0: 896, x1: 1024 });
  });
  it("windowHead finds the head under the window centre", () => {
    expect(windowHead({ x0: 384, x1: 512, y0: 0, y1: 1 }, layout)).toBe(3);
    expect(windowHead({ x0: 0, x1: 128, y0: 0, y1: 1 }, layout)).toBe(0);
    expect(windowHead({ x0: 100, x1: 700, y0: 0, y1: 1 }, layout)).toBe(3);
  });
});

describe("group separators", () => {
  it("draws boundaries strictly inside the view when a group is >= 6px wide", () => {
    // 4 heads of 128 in a 512-wide view on a 512px plot: 128px per head
    expect(groupBoundaries(0, 512, 128, 512)).toEqual([128, 256, 384]);
    expect(groupBoundaries(100, 300, 128, 400)).toEqual([128, 256]);
  });
  it("draws nothing when heads are narrower than 6px", () => {
    // 16 heads of 128 = 2048 channels on 600px -> 37px per head is fine, 100px view of 2048 channels is not
    expect(groupBoundaries(0, 2048, 128, 600).length).toBe(15);
    expect(groupBoundaries(0, 2048, 128, 90)).toEqual([]);
    expect(groupBoundaries(0, 2048, 128, 96).length).toBe(15); // exactly 6px per head
  });
  it("ignores degenerate input", () => {
    expect(groupBoundaries(0, 100, 0, 500)).toEqual([]);
    expect(groupBoundaries(5, 5, 10, 500)).toEqual([]);
  });
});

describe("activation picker", () => {
  it("shows channel counts, head shapes and head counts", () => {
    expect(actOptionLabel(act({ label: "resid_pre — attn_norm input", dim: 1024 }))).toBe("resid_pre — attn_norm input · 1024 ch");
    expect(actOptionLabel(q)).toBe("q — q_proj · 16×128");
    expect(actOptionLabel(act({ label: "attn_pattern", kind: "attn", dim: null, n_heads: 16 }))).toBe("attn_pattern · 16 heads");
  });
  it("groups in the order the backend sent", () => {
    const g = groupActivations([act({ id: "a", group: "Residual" }), act({ id: "b", group: "Attention" }), act({ id: "c", group: "Residual" })]);
    expect(g.map((x) => [x.group, x.items.map((i) => i.id)])).toEqual([["Residual", ["a", "c"]], ["Attention", ["b"]]]);
  });
});
