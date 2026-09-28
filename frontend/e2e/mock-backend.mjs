// Synthetic stand-in for GET /api/run/{id}/stats and /axis_stats.
// Heavy-tailed values, a massive-activation channel (37, huge at token 0), a second outlier channel (101),
// kurtosis in the hundreds. Only order=natural|absmax are implemented (enough to test rank != id).
export const T = 48;
export const C = 256;
export const OUTLIER_CH = 37;

function rng(seed) {
  let s = seed >>> 0;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}
function gauss(r) {
  return Math.sqrt(-2 * Math.log(r() + 1e-12)) * Math.cos(2 * Math.PI * r());
}

const cache = new Map();
export function matrix(layer) {
  if (cache.has(layer)) return cache.get(layer);
  const r = rng(1234 + layer);
  const x = new Float64Array(T * C);
  const chScale = Array.from({ length: C }, () => Math.exp(0.6 * gauss(r)));
  for (let t = 0; t < T; t++)
    for (let c = 0; c < C; c++) {
      const heavy = gauss(r) / Math.sqrt(Math.max(0.05, (gauss(r) ** 2 + gauss(r) ** 2 + gauss(r) ** 2) / 3)); // student-t(3)-like
      x[t * C + c] = 0.4 * chScale[c] * heavy * (t === 0 ? 2.5 : 1);
    }
  for (let t = 0; t < T; t++) x[t * C + OUTLIER_CH] = (t === 0 ? 320 : 18) + gauss(r) * 2;
  for (let t = 0; t < T; t++) x[t * C + 101] += 6 * gauss(r);
  cache.set(layer, x);
  return x;
}

function order(layer, kind) {
  const idx = Array.from({ length: C }, (_, i) => i);
  if (kind === "absmax") {
    const x = matrix(layer);
    const am = idx.map((c) => Math.max(...Array.from({ length: T }, (_, t) => Math.abs(x[t * C + c]))));
    idx.sort((a, b) => am[b] - am[a]);
  }
  return idx;
}

const PCTS = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9];
function percentile(sorted, p) {
  const k = ((sorted.length - 1) * p) / 100;
  const f = Math.floor(k);
  const c = Math.min(f + 1, sorted.length - 1);
  return sorted[f] + (sorted[c] - sorted[f]) * (k - f);
}
export function summarize(vals, bins = 64, clip = false) {
  const n = vals.length;
  const v = Float64Array.from(vals);
  const mean = v.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
  const m4 = v.reduce((a, b) => a + (b - mean) ** 4, 0) / n;
  const m3 = v.reduce((a, b) => a + (b - mean) ** 3, 0) / n;
  const sorted = Float64Array.from(v).sort();
  const pct = Object.fromEntries(PCTS.map((p) => [String(p), percentile(sorted, p)]));
  let lo = clip ? pct["0.1"] : sorted[0];
  let hi = clip ? pct["99.9"] : sorted[n - 1];
  if (hi <= lo) hi = lo + 1e-9;
  const counts = new Array(bins).fill(0);
  const edges = Array.from({ length: bins + 1 }, (_, i) => lo + ((hi - lo) * i) / bins);
  let inside = 0;
  for (const x of v) {
    if (x < lo || x > hi) continue;
    counts[Math.min(bins - 1, Math.floor(((x - lo) / (hi - lo)) * bins))]++;
    inside++;
  }
  return {
    n, mean, std, min: sorted[0], max: sorted[n - 1], absmax: Math.max(Math.abs(sorted[0]), Math.abs(sorted[n - 1])),
    l2: Math.sqrt(v.reduce((a, b) => a + b * b, 0)),
    skew: std > 0 ? m3 / std ** 3 : 0, kurtosis: std > 0 ? m4 / std ** 4 - 3 : 0,
    frac_pos: v.filter((a) => a > 0).length / n,
    frac_3sigma: std > 0 ? v.filter((a) => Math.abs(a - mean) > 3 * std).length / n : 0,
    percentiles: pct, hist: { counts, edges, n_clipped: n - inside },
  };
}

function statOf(v, stat) {
  const n = v.length;
  const mean = v.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / n);
  switch (stat) {
    case "absmax": return Math.max(...v.map(Math.abs));
    case "mean": return mean;
    case "std": return std;
    case "norm": return Math.sqrt(v.reduce((a, b) => a + b * b, 0));
    case "kurtosis": return std > 0 ? v.reduce((a, b) => a + (b - mean) ** 4, 0) / n / std ** 4 - 3 : 0;
  }
  throw new Error("bad stat");
}

const clampWin = (a, b, n) => { a = Math.max(0, Math.min(a, n - 1)); b = Math.max(a + 1, Math.min(b, n)); return [a, b]; };

/** @returns {{status:number, body:any}} */
export function handle(pathname, q) {
  const layer = Number(q.get("layer") ?? 0);
  const ord = q.get("order") ?? "natural";
  const clip = q.get("clip") === "true";
  if (q.has("site")) return { status: 400, body: { detail: "unknown parameter 'site' (use 'act')" } };
  if (!q.get("act")) return { status: 400, body: { detail: "missing act" } };
  if (q.get("act") === "boom") return { status: 500, body: { detail: "capture failed: CUDA out of memory (mock)" } };
  let [t0, t1] = clampWin(Number(q.get("t0") ?? 0), Number(q.get("t1") ?? 64), T);
  let [d0, d1] = clampWin(Number(q.get("d0") ?? 0), Number(q.get("d1") ?? 128), C);
  const x = matrix(layer);
  const perm = order(layer, ord);
  const cols = perm.slice(d0, d1);
  const region = { t0, t1, d0, d1 };
  if (pathname.endsWith("/stats")) {
    const vals = [], where = [];
    for (let t = t0; t < t1; t++) for (const c of cols) { vals.push(x[t * C + c]); where.push([t, c]); }
    const out = summarize(vals, 64, clip);
    const idx = vals.map((_, i) => i).sort((a, b) => Math.abs(vals[b]) - Math.abs(vals[a])).slice(0, 8);
    out.top = idx.map((i) => ({ token: where[i][0], dim: where[i][1], value: vals[i] }));
    out.region = region;
    return { status: 200, body: out };
  }
  if (pathname.endsWith("/axis_stats")) {
    const axis = q.get("axis"), stat = q.get("stat") ?? "norm";
    const top = Number(q.get("top") ?? 10), bins = Number(q.get("bins") ?? 64);
    if (!["channel", "token"].includes(axis)) return { status: 400, body: { detail: "bad axis" } };
    const items = [];
    if (axis === "channel") cols.forEach((c, k) => items.push({ index: d0 + k, id: c, value: statOf(Array.from({ length: t1 - t0 }, (_, i) => x[(t0 + i) * C + c]), stat) }));
    else for (let t = t0; t < t1; t++) items.push({ index: t, id: t, value: statOf(cols.map((c) => x[t * C + c]), stat) });
    const out = summarize(items.map((i) => i.value), bins, clip);
    out.axis = axis; out.stat = stat; out.region = region;
    out.top = [...items].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, top);
    return { status: 200, body: out };
  }
  return { status: 404, body: { detail: "not found" } };
}
