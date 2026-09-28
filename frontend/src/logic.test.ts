import { describe, expect, it } from "vitest";
import { decodeFrame } from "./api";
import { COLORMAPS, buildLut, formatNumber, parseColor, resolveRange, tToValue, valueToT } from "./colormaps";
import { fmtToken } from "./tokens";
import { clampViewport, fetchWindow, niceStep, pan, zoomAt } from "./viewport";

const ext = { nx: 1000, ny: 100 };

describe("viewport", () => {
  it("zoom keeps the anchor value under the cursor", () => {
    const vp = { x0: 100, x1: 300, y0: 10, y1: 50 };
    const z = zoomAt(vp, ext, 0.25, 0.5, 0.5);
    // anchor: x = 100 + 0.25*200 = 150, y = 10 + 0.5*40 = 30
    expect(z.x0 + 0.25 * (z.x1 - z.x0)).toBeCloseTo(150);
    expect(z.y0 + 0.5 * (z.y1 - z.y0)).toBeCloseTo(30);
    expect(z.x1 - z.x0).toBeCloseTo(100);
  });
  it("never zooms out past the extent or in past minSpan", () => {
    const out = zoomAt({ x0: 0, x1: 900, y0: 0, y1: 90 }, ext, 0.5, 0.5, 100);
    expect(out).toEqual({ x0: 0, x1: 1000, y0: 0, y1: 100 });
    const inn = zoomAt({ x0: 0, x1: 10, y0: 0, y1: 10 }, ext, 0.5, 0.5, 1e-6, "both", 3);
    expect(inn.x1 - inn.x0).toBeCloseTo(3);
  });
  it("zoom near an edge stays inside the extent", () => {
    const z = zoomAt({ x0: 0, x1: 100, y0: 0, y1: 10 }, ext, 0, 0, 3);
    expect(z.x0).toBeGreaterThanOrEqual(0);
    expect(z.x1).toBeLessThanOrEqual(1000);
  });
  it("pan clamps and preserves span", () => {
    const p = pan({ x0: 900, x1: 1000, y0: 0, y1: 10 }, ext, 500, -50);
    expect(p).toEqual({ x0: 900, x1: 1000, y0: 0, y1: 10 });
    const q = pan({ x0: 100, x1: 200, y0: 5, y1: 15 }, ext, 30, 20);
    expect(q.x1 - q.x0).toBeCloseTo(100);
    expect(q.x0).toBeCloseTo(130);
  });
  it("axis-restricted zoom leaves the other axis alone", () => {
    const vp = { x0: 100, x1: 300, y0: 10, y1: 50 };
    const z = zoomAt(vp, ext, 0.5, 0.5, 0.5, "x");
    expect(z.y0).toBe(10);
    expect(z.y1).toBe(50);
    expect(z.x1 - z.x0).toBeCloseTo(100);
  });
  it("fetchWindow pads, rounds outward and clamps", () => {
    const w = fetchWindow({ x0: 10.4, x1: 20.6, y0: 0.2, y1: 9.9 }, ext, 0.1);
    expect(w.d0).toBe(9);
    expect(w.d1).toBe(22);
    expect(w.t0).toBe(0);
    expect(w.t1).toBe(11);
    const edge = fetchWindow({ x0: 990, x1: 1000, y0: 90, y1: 100 }, ext, 0.5);
    expect(edge.d1).toBe(1000);
    expect(edge.t1).toBe(100);
  });
  it("clampViewport handles oversize and inverted input", () => {
    expect(clampViewport({ x0: -50, x1: 5000, y0: 200, y1: 100 }, ext, 2)).toEqual({ x0: 0, x1: 1000, y0: 98, y1: 100 });
  });
  it("niceStep returns 1/2/5 multiples", () => {
    expect([0.3, 1, 1.1, 2.4, 6, 11, 49].map(niceStep)).toEqual([1, 1, 2, 5, 10, 20, 50]);
  });
});

describe("colour scales", () => {
  const cases = [
    { vmin: 0, vmax: 10, symmetric: false },
    { vmin: -5, vmax: 5, symmetric: true },
  ];
  for (const scale of ["linear", "sqrt", "symlog"] as const) {
    for (const c of cases) {
      it(`${scale}${c.symmetric ? " symmetric" : ""}: valueToT and tToValue are inverses`, () => {
        for (const t of [0, 0.1, 0.35, 0.5, 0.8, 1]) {
          const spec = { ...c, scale };
          expect(valueToT(tToValue(t, spec), spec)).toBeCloseTo(t, 6);
        }
      });
    }
  }
  it("symmetric maps 0 to the midpoint and clamps outliers", () => {
    const spec = { vmin: -5, vmax: 5, symmetric: true, scale: "linear" as const };
    expect(valueToT(0, spec)).toBe(0.5);
    expect(valueToT(1e9, spec)).toBe(1);
    expect(valueToT(-1e9, spec)).toBe(0);
    expect(Number.isNaN(valueToT(NaN, spec))).toBe(true);
  });
  it("symlog keeps small values visible next to a huge outlier", () => {
    const spec = { vmin: -6000, vmax: 6000, symmetric: true, scale: "symlog" as const };
    const lin = { ...spec, scale: "linear" as const };
    expect(valueToT(60, spec) - 0.5).toBeGreaterThan(5 * (valueToT(60, lin) - 0.5));
  });
  it("resolveRange: symmetric, manual and window modes", () => {
    const g = { min: -10, max: 4, p01: -8, p99: 3 };
    expect(resolveRange("robust", true, g, null, [0, 1])).toEqual([-8, 8]);
    expect(resolveRange("full", false, g, null, [0, 1])).toEqual([-10, 4]);
    expect(resolveRange("manual", false, g, null, [2, 7])).toEqual([2, 7]);
    const vals = Float32Array.from({ length: 1001 }, (_, i) => i);
    const [lo, hi] = resolveRange("window", false, null, vals, [0, 1]);
    expect(lo).toBeCloseTo(10, 0);
    expect(hi).toBeCloseTo(990, 0);
    const [a, b] = resolveRange("manual", false, null, null, [3, 3]);
    expect(b).toBeGreaterThan(a);
  });
  it("every colormap builds a full 256-entry opaque LUT", () => {
    for (const cm of COLORMAPS) {
      const lut = buildLut(cm);
      expect(lut.length).toBe(1024);
      expect(lut[3]).toBe(255);
      expect(lut[1023]).toBe(255);
      expect(Array.from(lut.slice(0, 3))).not.toEqual(Array.from(lut.slice(1020, 1023)));
    }
  });
  it("parseColor handles rgb() and hex", () => {
    expect(parseColor("rgb(12, 34, 56)")).toEqual([12, 34, 56]);
    expect(parseColor("#0a141e")).toEqual([10, 20, 30]);
    expect(parseColor("#fff")).toEqual([255, 255, 255]);
  });
  it("formatNumber is compact", () => {
    expect(formatNumber(0)).toBe("0");
    expect(formatNumber(6565.3)).toBe("6565");
    expect(formatNumber(0.12345)).toBe("0.123");
    expect(formatNumber(1.5e-5)).toBe("1.50e-5");
    expect(formatNumber(-42.37)).toBe("-42.4");
  });
});

describe("wire format", () => {
  it("decodes what backend/actlens/wire.py produces", () => {
    const meta = { rows: 2, cols: 3, note: "héllo" }; // multi-byte char makes the byte length differ from the string length
    const json = new TextEncoder().encode(JSON.stringify(meta));
    const pad = (4 - ((4 + json.length) % 4)) % 4;
    const values = [1.5, -2, 3, 4, 5.25, 6];
    const buf = new ArrayBuffer(4 + json.length + pad + values.length * 4);
    new DataView(buf).setUint32(0, json.length, true);
    new Uint8Array(buf, 4, json.length).set(json);
    new Float32Array(buf, 4 + json.length + pad, values.length).set(values);
    const f = decodeFrame<typeof meta>(buf);
    expect(f.meta).toEqual(meta);
    expect(Array.from(f.data)).toEqual(values);
  });
});

describe("fmtToken", () => {
  it("makes whitespace visible and bounds length", () => {
    expect(fmtToken(" capital")).toBe("·capital");
    expect(fmtToken("\n\n")).toBe("↵↵");
    expect(fmtToken("")).toBe("∅");
    expect(Array.from(fmtToken("x".repeat(50), 10)).length).toBe(10);
    expect(fmtToken("人工智能")).toBe("人工智能");
    expect(fmtToken("a\u0001b")).toBe("a�b");
  });
});
