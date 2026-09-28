import { describe, expect, it } from "vitest";
import {
  axisStatsKey,
  channelLabel,
  channelName,
  clampRegion,
  clippedShare,
  effectiveScope,
  fmtNum,
  fmtPct,
  insideRegion,
  keepSameAxis,
  regionText,
  resolveAxisRegion,
  resolveValueRegion,
  scopeAvailable,
  scopeText,
  statOf,
  tokenLabel,
  topLabel,
  valueStatsKey,
  type ScopeContext,
} from "./axisStats";
import { barHeight, barX, binAtX, histMargins, outsideNote, plotWidth, yTicks, yValue } from "./components/histogramGeometry";

const base: ScopeContext = {
  nTokens: 40,
  nChannels: 512,
  window: { t0: 4, t1: 20, d0: 8, d1: 72 },
  selection: null,
  cursor: null,
};

describe("scope -> region", () => {
  it("window / layer need nothing", () => {
    expect(resolveValueRegion("window", base)).toEqual({ t0: 4, t1: 20, d0: 8, d1: 72 });
    expect(resolveValueRegion("layer", base)).toEqual({ t0: 0, t1: 40, d0: 0, d1: 512 });
  });
  it("selection, channel and token are null without their prerequisite", () => {
    for (const s of ["selection", "channel", "token"] as const) {
      expect(resolveValueRegion(s, base)).toBeNull();
      expect(scopeAvailable(s, base)).toBe(false);
      expect(effectiveScope(s, base)).toBe("window");
    }
  });
  it("selection uses the brushed region", () => {
    const ctx = { ...base, selection: { t0: 2, t1: 6, d0: 100, d1: 110 } };
    expect(resolveValueRegion("selection", ctx)).toEqual({ t0: 2, t1: 6, d0: 100, d1: 110 });
    expect(resolveAxisRegion("selection", ctx)).toEqual({ t0: 2, t1: 6, d0: 100, d1: 110 });
    expect(effectiveScope("selection", ctx)).toBe("selection");
  });
  it("channel = cursor column over all tokens, token = cursor row over all channels", () => {
    const ctx = { ...base, cursor: { token: 7, channel: 33 } };
    expect(resolveValueRegion("channel", ctx)).toEqual({ t0: 0, t1: 40, d0: 33, d1: 34 });
    expect(resolveValueRegion("token", ctx)).toEqual({ t0: 7, t1: 8, d0: 0, d1: 512 });
  });
  it("clamps fractional and out-of-range regions to a non-empty integer region", () => {
    expect(clampRegion({ t0: -3, t1: 500, d0: 10.2, d1: 20.1 }, 40, 512)).toEqual({ t0: 0, t1: 40, d0: 10, d1: 21 });
    expect(clampRegion({ t0: 5, t1: 5, d0: 7, d1: 3 }, 40, 512)).toEqual({ t0: 5, t1: 6, d0: 7, d1: 8 });
    // cursor past the end (e.g. after switching to a smaller activation) stays inside the tensor
    expect(clampRegion({ t0: 99, t1: 100, d0: 0, d1: 1 }, 40, 512)).toEqual({ t0: 39, t1: 40, d0: 0, d1: 1 });
  });
});

describe("query keys", () => {
  const region = { t0: 0, t1: 4, d0: 0, d1: 8 };
  it("change with every parameter that changes the result", () => {
    const k = (over: object = {}) => JSON.stringify(valueStatsKey({ runId: "r", act: "q", layer: 1, region, order: "natural", clip: false, ...over }));
    const ref = k();
    expect(k({ layer: 2 })).not.toBe(ref);
    expect(k({ act: "k" })).not.toBe(ref);
    expect(k({ order: "absmax" })).not.toBe(ref);
    expect(k({ clip: true })).not.toBe(ref);
    expect(k({ region: { ...region, d1: 9 } })).not.toBe(ref);
    const a = (over: object = {}) => JSON.stringify(axisStatsKey({ runId: "r", act: "q", layer: 1, axis: "channel", stat: "std", region, order: "natural", clip: false, ...over }));
    expect(a({ stat: "norm" })).not.toBe(a());
    expect(a({ axis: "token" })).not.toBe(a());
  });
  it("placeholder data is only kept within one axis", () => {
    const keep = keepSameAxis<string>("channel");
    const chan = { queryKey: axisStatsKey({ runId: "r", act: "q", layer: 1, axis: "channel", stat: "std", region, order: "natural", clip: false }) };
    const tok = { queryKey: axisStatsKey({ runId: "r", act: "q", layer: 1, axis: "token", stat: "std", region, order: "natural", clip: false }) };
    expect(keep("prev", chan)).toBe("prev");
    expect(keep("prev", tok)).toBeUndefined();
    expect(keep(undefined, undefined)).toBeUndefined();
  });
});

describe("labels", () => {
  it("channel labels use h{head}·{d} only for head-structured axes", () => {
    expect(channelLabel(401, null)).toBe("401");
    expect(channelLabel(401, 128)).toBe("h3·17");
    expect(channelLabel(0, 64)).toBe("h0·0");
    expect(channelName(401, 128)).toBe("head 3, dim 17 (channel 401)");
    expect(channelName(401, null)).toBe("channel 401");
  });
  it("token labels carry index and a readable token", () => {
    expect(tokenLabel(3, ["a", "b", "c", " capital"])).toBe("#3 ·capital");
    expect(tokenLabel(1, ["a", "\n"])).toBe("#1 ↵");
    expect(tokenLabel(9, ["a"])).toBe("#9");
  });
  it("top labels flag rank != id", () => {
    expect(topLabel("channel", { index: 0, id: 37, value: 5 }, [], null)).toEqual({ main: "ch 37", sub: "rank 0" });
    expect(topLabel("channel", { index: 37, id: 37, value: 5 }, [], null)).toEqual({ main: "ch 37", sub: "" });
    expect(topLabel("channel", { index: 3, id: 401, value: 5 }, [], 128)).toEqual({ main: "401 · h3·17", sub: "rank 3" });
    expect(topLabel("token", { index: 1, id: 1, value: 2 }, ["<s>", " The"], null)).toEqual({ main: "#1 ·The", sub: "" });
  });
  it("scope text names the cursor's channel by original id", () => {
    const o = { cursor: { token: 2, channel: 5 }, cursorChannelId: 401, tokens: ["a", "b", "c"], headDim: 128 };
    expect(scopeText("channel", o)).toBe("channel h3·17, all tokens");
    expect(scopeText("token", o)).toBe("#2 c, all channels");
    expect(scopeText("layer", o)).toBe("whole layer");
  });
  it("region text", () => {
    expect(regionText({ t0: 0, t1: 64, d0: 0, d1: 128 }, "natural")).toBe("tokens 0–63 · channels 0–127");
    expect(regionText({ t0: 3, t1: 4, d0: 0, d1: 128 }, "absmax")).toBe("token 3 · channels 0–127 (ranked)");
    expect(regionText({ t0: 0, t1: 8, d0: 5, d1: 6 }, "absmax")).toBe("tokens 0–7 · channel 5");
    expect(regionText({ t0: 0, t1: 8, d0: 256, d1: 384 }, "natural", 128)).toBe("tokens 0–7 · head 2");
    expect(regionText({ t0: 0, t1: 8, d0: 256, d1: 384 }, "absmax", 128)).toBe("tokens 0–7 · channels 256–383 (ranked)");
  });
});

describe("number formatting", () => {
  it("is tolerant of missing values", () => {
    expect(fmtNum(null)).toBe("–");
    expect(fmtNum(undefined)).toBe("–");
    expect(fmtNum(NaN)).toBe("–");
    expect(fmtNum(0)).toBe("0");
    expect(fmtNum(1234.5)).toBe("1235");
    expect(fmtNum(0.000012)).toBe("1.20e-5");
    expect(fmtNum(-3.14159)).toBe("-3.14");
  });
  it("percentages never round a non-zero fraction to zero", () => {
    expect(fmtPct(0.5)).toBe("50.0%");
    expect(fmtPct(0.00001, 2)).toBe("<0.01%");
    expect(fmtPct(0, 2)).toBe("0.00%");
    expect(fmtPct(null)).toBe("–");
  });
  it("clipped share and the outside note", () => {
    expect(clippedShare(2, 1000)).toBeCloseTo(0.002);
    expect(clippedShare(0, 0)).toBe(0);
    expect(outsideNote(0, 100)).toBe("none outside");
    expect(outsideNote(2, 100000)).toBe("2 outside (<0.1%)");
    expect(outsideNote(50, 1000)).toBe("50 outside (5.0%)");
  });
});

describe("histogram geometry", () => {
  const m = histMargins({});
  const w = 344; // plot width = 344 - 44 - 10 = 290
  it("margins grow with header lines and an x label", () => {
    expect(histMargins({ headerLines: 3 }).t).toBeGreaterThan(m.t);
    expect(histMargins({ xLabel: true }).b).toBeGreaterThan(m.b);
    expect(plotWidth(w, m)).toBe(290);
  });
  it("maps pointer x to a bin, null outside the plot", () => {
    expect(binAtX(m.l - 1, w, m, 29)).toBeNull();
    expect(binAtX(m.l, w, m, 29)).toBe(0);
    expect(binAtX(m.l + 145, w, m, 29)).toBe(14);
    expect(binAtX(m.l + 289.9, w, m, 29)).toBe(28);
    expect(binAtX(m.l + 290, w, m, 29)).toBeNull();
    expect(binAtX(50, w, m, 0)).toBeNull();
  });
  it("bars tile the plot without overlapping", () => {
    const edges = Array.from({ length: 11 }, (_, i) => i);
    const bars = Array.from({ length: 10 }, (_, i) => barX(i, edges, 0, 10, 44, 290));
    for (let i = 1; i < bars.length; i++) expect(bars[i].x).toBeGreaterThanOrEqual(bars[i - 1].x + bars[i - 1].w - 1e-9);
    expect(bars[0].x).toBeGreaterThanOrEqual(44);
    expect(bars[9].x + bars[9].w).toBeLessThanOrEqual(44 + 290 + 1e-9);
  });
  it("bar heights: empty -> 0, tallest -> full, tiny non-empty bins stay visible", () => {
    expect(barHeight(0, 1000, false, 100)).toBe(0);
    expect(barHeight(1000, 1000, false, 100)).toBeCloseTo(100);
    expect(barHeight(1, 1e6, false, 100)).toBeGreaterThanOrEqual(1.5);
    // log scale: a singleton next to a 1e6 bin is 5% of the height, not invisible
    expect(barHeight(1, 1e6, true, 100)).toBeCloseTo((Math.log10(2) / Math.log10(1e6 + 1)) * 100, 6);
    expect(yValue(0, true)).toBe(0);
  });
  it("y ticks", () => {
    const lin = yTicks(1000, false);
    expect(lin[0]).toEqual({ count: 0, frac: 0 });
    expect(lin.every((t) => t.frac >= 0 && t.frac <= 1.0001)).toBe(true);
    expect(lin.map((t) => t.count)).toEqual([0, 500, 1000]);
    const log = yTicks(50000, true);
    expect(log.map((t) => t.count)).toEqual([0, 1, 10, 100, 1000, 10000]);
    expect(log.every((t, i) => i === 0 || t.frac > log[i - 1].frac)).toBe(true);
    expect(log[log.length - 1].frac).toBeLessThanOrEqual(1);
    expect(yTicks(3, true).map((t) => t.count)).toEqual([0, 1, 3]);
    expect(yTicks(0, false).length).toBeGreaterThan(0);
  });
});

describe("inside the selected channel / token", () => {
  const region = { t0: 4, t1: 20, d0: 8, d1: 72 };
  it("needs a cursor", () => {
    expect(insideRegion("channel", region, null)).toBeNull();
    expect(insideRegion("token", region, null)).toBeNull();
  });
  it("channel: the cursor column over the region's tokens", () => {
    expect(insideRegion("channel", region, { token: 99, channel: 33 })).toEqual({ t0: 4, t1: 20, d0: 33, d1: 34 });
  });
  it("token: the cursor row over the region's channels", () => {
    expect(insideRegion("token", region, { token: 7, channel: 999 })).toEqual({ t0: 7, t1: 8, d0: 8, d1: 72 });
  });
  it("is null when the cursor lies outside the region (end exclusive)", () => {
    expect(insideRegion("channel", region, { token: 7, channel: 72 })).toBeNull();
    expect(insideRegion("channel", region, { token: 7, channel: 7 })).toBeNull();
    expect(insideRegion("token", region, { token: 20, channel: 10 })).toBeNull();
  });
  it("statOf recovers each axis statistic from a value summary", () => {
    const s = { absmax: 5, std: 2, mean: -1, l2: 9, kurtosis: 3.5 };
    expect(statOf("absmax", s)).toBe(5);
    expect(statOf("std", s)).toBe(2);
    expect(statOf("mean", s)).toBe(-1);
    expect(statOf("norm", s)).toBe(9);
    expect(statOf("kurtosis", s)).toBe(3.5);
  });
});
