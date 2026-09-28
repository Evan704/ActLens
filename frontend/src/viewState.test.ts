import { describe, expect, it } from "vitest";
import type { ActivationInfo, RunInfo } from "./api";
import {
  ACT_DISPLAY, ATTN_DISPLAY, defaultOrder, defaultSettings, defaultWindow, initialCore, jumpToHead, openCell, pickChannel, pickToken,
  resetForRun, switchActivation, type Core, type TokenView,
} from "./viewState";

const labels = (n: number) => Array.from({ length: n }, (_, i) => String(i));
const act = (p: Partial<ActivationInfo>): ActivationInfo => ({
  id: "x", label: "x", group: "Residual", kind: "token", n_layers: 28, dim: 1024, layer_labels: labels(28), n_heads: null, head_dim: null, description: "", ...p,
});
const resid = act({ id: "resid_post", label: "resid_post" });
const q = act({ id: "q", group: "Attention", dim: 2048, n_heads: 16, head_dim: 128 });
const k = act({ id: "k", group: "Attention", dim: 1024, n_heads: 8, head_dim: 128 });
const gate = act({ id: "gate", group: "MLP", dim: 3072 });
const down = act({ id: "down", group: "MLP", dim: 1024 });
const pattern = act({ id: "attn_pattern", group: "Attention", kind: "attn", dim: null, n_heads: 16 });

const run = (T = 100, model = "m", acts = [resid, q, k, gate, down, pattern]): RunInfo => ({
  run_id: "r", model_id: model, tokens: Array.from({ length: T }, (_, i) => `t${i}`), token_ids: [], truncated: false,
  model: { n_layers: 28, hidden_size: 1024, intermediate_size: 3072, n_heads: 16, n_kv_heads: 8, head_dim: 128 }, activations: acts,
});
const tokView = (info: ActivationInfo, T = 100): TokenView => ({ ...defaultSettings(info, T), brush: null, cursor: null });

describe("defaults", () => {
  it("window is 64 tokens x min(C,128); short prompts are not padded", () => {
    expect(defaultWindow(resid, 100)).toEqual({ x0: 0, x1: 128, y0: 0, y1: 64 });
    expect(defaultWindow(act({ dim: 96 }), 20)).toEqual({ x0: 0, x1: 96, y0: 0, y1: 20 });
  });
  it("head-structured acts start on exactly head 0", () => {
    expect(defaultWindow(act({ dim: 2048, n_heads: 16, head_dim: 64 }), 100)).toEqual({ x0: 0, x1: 64, y0: 0, y1: 64 });
    expect(defaultWindow(q, 100).x1).toBe(128);
  });
  it("signed acts use RdBu symmetric, attention magma manual [0,1] sqrt", () => {
    expect(defaultSettings(resid, 10).display).toEqual(ACT_DISPLAY);
    expect(defaultSettings(q, 10).display).toEqual(ACT_DISPLAY);
    expect(ATTN_DISPLAY).toMatchObject({ cmap: "magma", range: "manual", scale: "sqrt", manual: [0, 1] });
  });
  it("wide MLP acts are ranked by |max|, everything else is natural", () => {
    expect(defaultOrder(gate)).toBe("absmax");
    expect(defaultOrder(down)).toBe("natural");
    expect(defaultOrder(q)).toBe("natural");
  });
});

describe("switching activation", () => {
  it("keeps the token window, restores the new activation's own settings", () => {
    let tok: TokenView = { ...tokView(resid), vp: { x0: 10, x1: 74, y0: 30, y1: 60 }, order: "std", display: { ...ACT_DISPLAY, cmap: "PuOr" } };
    let memory = {};
    ({ tok, memory } = switchActivation(tok, memory, resid, q, 100));
    // q has no memory yet: default channel window (head 0), but the token window carries over
    expect(tok.vp).toEqual({ x0: 0, x1: 128, y0: 30, y1: 60 });
    expect(tok.order).toBe("natural");
    expect(tok.display.cmap).toBe("RdBu");
    tok = { ...tok, vp: { x0: 256, x1: 384, y0: 40, y1: 70 }, agg: "mean" };
    ({ tok, memory } = switchActivation(tok, memory, q, resid, 100));
    // back to resid: its remembered order/display/channel window, current token window
    expect(tok).toMatchObject({ order: "std", vp: { x0: 10, x1: 74, y0: 40, y1: 70 }, agg: "absmax" });
    expect(tok.display.cmap).toBe("PuOr");
    ({ tok } = switchActivation(tok, memory, resid, q, 100));
    expect(tok).toMatchObject({ vp: { x0: 256, x1: 384, y0: 40, y1: 70 }, agg: "mean" });
  });
  it("clears the selection and clamps the window to the new channel extent", () => {
    const tok: TokenView = { ...tokView(gate), vp: { x0: 2000, x1: 2128, y0: 0, y1: 64 }, brush: { x0: 1, x1: 2, y0: 1, y1: 2 }, cursor: { row: 3, col: 2010 } };
    const out = switchActivation(tok, {}, gate, down, 100).tok;
    expect(out.brush).toBeNull();
    expect(out.cursor).toBeNull();
    expect(out.vp.x1).toBeLessThanOrEqual(1024);
    expect(out.vp.x1 - out.vp.x0).toBe(128);
  });
  it("attention patterns leave the token view untouched but remember the token act", () => {
    const tok: TokenView = { ...tokView(resid), vp: { x0: 5, x1: 69, y0: 8, y1: 40 }, order: "absmax" };
    const sw = switchActivation(tok, {}, resid, pattern, 100);
    expect(sw.tok).toBe(tok);
    expect(sw.memory.resid_post.order).toBe("absmax");
    // and coming back from attention must not store the (stale) live view under the attention id
    const back = switchActivation(sw.tok, sw.memory, pattern, resid, 100);
    expect(back.memory.attn_pattern).toBeUndefined();
    expect(back.tok.vp).toEqual({ x0: 5, x1: 69, y0: 8, y1: 40 });
  });
});

describe("open cell from the across-layers map", () => {
  it("centres the token window on the token and places the cursor", () => {
    const tok = { ...tokView(resid), vp: { x0: 0, x1: 128, y0: 0, y1: 20 } };
    const out = openCell(tok, resid, 100, 50, null);
    expect(out.cursor).toMatchObject({ row: 50, col: 0 });
    expect(out.vp.y1 - out.vp.y0).toBe(20);
    expect(out.vp.y0).toBe(40);
  });
  it("keeps the previous cursor column and clamps at the ends", () => {
    const tok: TokenView = { ...tokView(resid), vp: { x0: 0, x1: 128, y0: 0, y1: 20 }, cursor: { row: 1, col: 77 } };
    expect(openCell(tok, resid, 100, 99, null).cursor).toMatchObject({ row: 99, col: 77 });
    expect(openCell(tok, resid, 100, 99, null).vp.y1).toBe(100);
    expect(openCell(tok, resid, 100, 0, null).vp.y0).toBe(0);
  });
  it("single-channel stat brings that channel into the window", () => {
    const tok = tokView(resid);
    const out = openCell(tok, resid, 100, 10, 700);
    expect(out.cursor).toMatchObject({ row: 10, col: 700, id: 700 });
    expect(out.vp.x0).toBeLessThanOrEqual(700);
    expect(out.vp.x1).toBeGreaterThan(700);
    expect(out.vp.x1 - out.vp.x0).toBe(128);
  });
});

describe("panel picks and head jump", () => {
  it("pickToken re-centres only when the row is outside the window", () => {
    const tok = tokView(resid);
    expect(pickToken(tok, 100, 1024, 10).vp).toEqual(tok.vp);
    const far = pickToken(tok, 100, 1024, 90);
    expect(far.cursor?.row).toBe(90);
    expect(far.vp.y0).toBeGreaterThan(30);
    expect(far.vp.y1).toBe(100);
  });
  it("pickChannel moves the cursor column and re-centres the window", () => {
    const tok = tokView(resid);
    const out = pickChannel(tok, 100, 1024, 900);
    expect(out.cursor?.col).toBe(900);
    expect(out.vp.x0).toBeLessThanOrEqual(900);
    expect(out.vp.x1).toBeGreaterThan(900);
    expect(pickChannel(tok, 100, 1024, 5).vp).toEqual(tok.vp);
  });
  it("jumpToHead sets the window to that head, natural order, selection cleared", () => {
    const tok: TokenView = { ...tokView(q), order: "absmax", vp: { x0: 0, x1: 128, y0: 4, y1: 20 }, cursor: { row: 1, col: 1 } };
    const out = jumpToHead(tok, q, 100, 5);
    expect(out).toMatchObject({ order: "natural", vp: { x0: 640, x1: 768, y0: 4, y1: 20 }, cursor: null });
  });
  it("jumpToHead is a no-op for non-head activations", () => {
    const tok = tokView(resid);
    expect(jumpToHead(tok, resid, 100, 3)).toBe(tok);
  });
});

describe("new run", () => {
  const prev = (over: Partial<Core> = {}, r: RunInfo | null = null): Core & { run: RunInfo | null } => ({ ...initialCore(), ...over, run: r });
  it("first run of a model starts mid-network on resid_post", () => {
    const s = resetForRun(run(), prev());
    expect(s.act).toBe("resid_post");
    expect(s.layer).toBe(14);
    expect(s.tok.vp).toEqual({ x0: 0, x1: 128, y0: 0, y1: 64 });
    expect(s.attn.head).toBeNull();
  });
  it("same model: keeps activation, layer and settings, restarts token windows", () => {
    const r0 = run(100);
    const s0 = resetForRun(r0, prev());
    const cur = { ...s0, act: "q", layer: 5, memory: { resid_post: { ...defaultSettings(resid, 100), order: "std" as const } }, tok: { ...tokView(q), order: "natural" as const, vp: { x0: 256, x1: 384, y0: 50, y1: 90 } } };
    const s1 = resetForRun(run(30), prev(cur, r0));
    expect(s1.act).toBe("q");
    expect(s1.layer).toBe(5);
    expect(s1.tok.vp).toEqual({ x0: 256, x1: 384, y0: 0, y1: 30 });
    expect(s1.memory.resid_post.order).toBe("std");
    expect(s1.memory.resid_post.vp.y1).toBe(30);
  });
  it("new model: memory dropped; a missing activation falls back to resid_post", () => {
    const r0 = run(100, "a");
    const cur = { ...resetForRun(r0, prev()), act: "q", memory: { resid_post: { ...defaultSettings(resid, 100), order: "std" as const } } };
    const s = resetForRun(run(100, "b", [resid, gate]), prev(cur, r0));
    expect(s.act).toBe("resid_post");
    expect(s.memory).toEqual({});
    expect(s.tok.order).toBe("natural");
  });
});
