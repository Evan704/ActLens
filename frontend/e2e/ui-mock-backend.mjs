/**
 * Mock ActLens v2 backend for UI tests. NOT part of the app build.
 *
 * `installMock(page)` intercepts /api/* with Playwright `page.route` and fulfils the requests with synthetic data in
 * the exact wire format ([u32 json_len][json][pad to 4][float32 payload], see backend/actlens/wire.py). The fake model
 * looks like Qwen3-0.6B: 28 layers, D=1024, I=3072, nH=16, nKV=8, Dh=128, with q/k/v/q_norm/k_norm/q_rope/k_rope.
 * Values are deterministic functions of (act, layer, token, channel) with per-head amplitudes and a few outlier
 * channels, so head separators, ranking and the across-layers map have visible structure.
 */

export const MODEL = { n_layers: 28, hidden_size: 1024, intermediate_size: 3072, n_heads: 16, n_kv_heads: 8, head_dim: 128 };
const L = MODEL.n_layers;
const NH = MODEL.n_heads;
const DH = MODEL.head_dim;

const layerLabels = Array.from({ length: L }, (_, i) => String(i));
const A = (id, label, group, dim, nHeads = null, description = "") => ({
  id, label, group, kind: "token", n_layers: L, dim, layer_labels: layerLabels,
  n_heads: nHeads, head_dim: nHeads ? DH : null, description: description || label,
});
export const ACTIVATIONS = [
  A("resid_pre", "resid_pre — attn_norm input", "Residual", 1024),
  A("resid_mid", "resid_mid — mlp_norm input", "Residual", 1024),
  A("resid_post", "resid_post — block output", "Residual", 1024),
  A("attn_norm", "attn_norm — input_layernorm", "Attention", 1024),
  A("q", "q — q_proj", "Attention", NH * DH, NH),
  A("k", "k — k_proj", "Attention", 8 * DH, 8),
  A("v", "v — v_proj", "Attention", 8 * DH, 8),
  A("q_norm", "q_norm — q after RMSNorm", "Attention", NH * DH, NH),
  A("k_norm", "k_norm — k after RMSNorm", "Attention", 8 * DH, 8),
  A("q_rope", "q_rope — q after RoPE", "Attention", NH * DH, NH),
  A("k_rope", "k_rope — k after RoPE", "Attention", 8 * DH, 8),
  { id: "attn_pattern", label: "attn_pattern — softmax(QK)", group: "Attention", kind: "attn", n_layers: L, dim: null, layer_labels: layerLabels, n_heads: NH, head_dim: null, description: "attention probabilities" },
  A("attn_ctx", "attn_ctx — o_proj input", "Attention", NH * DH, NH),
  A("o", "o — o_proj output", "Attention", 1024),
  A("mlp_norm", "mlp_norm — post_attention_layernorm", "MLP", 1024),
  A("gate", "gate — gate_proj", "MLP", 3072),
  A("up", "up — up_proj", "MLP", 3072),
  A("silu", "silu — act_fn(gate)", "MLP", 3072),
  A("swiglu", "swiglu — silu(gate)·up", "MLP", 3072),
  A("down", "down — down_proj output", "MLP", 1024),
];
const byId = Object.fromEntries(ACTIVATIONS.map((a) => [a.id, a]));

const PROMPT = "The capital of France is Paris. In 1889 the Eiffel Tower opened in Paris, and it remains one of the most visited monuments in the world. \n\ndef fib(n):\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n人工智能正在改变世界。";
export function tokenize(text = PROMPT) {
  const toks = text.match(/\s*[A-Za-z]+|\s*\d+|\n+|\s*[^\sA-Za-z\d]|[一-鿿]|\s+/g) ?? [text];
  return toks.slice(0, 96);
}

// ---------- deterministic synthetic values ----------
const hash = (a, b, c, d) => {
  let h = (a * 374761393 + b * 668265263 + c * 2147483647 + d * 1274126177) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
};
const actIndex = (id) => ACTIVATIONS.findIndex((a) => a.id === id);

function value(id, layer, t, c) {
  const ai = actIndex(id);
  const dim = byId[id].dim;
  const headDim = byId[id].n_heads ? DH : 0;
  const head = headDim ? Math.floor(c / headDim) : 0;
  const amp = headDim ? 0.4 + 1.6 * ((head * 7 + ai) % 5) / 4 : 1;
  const noise = hash(ai, layer, t, c) - 0.5;
  let v = amp * (Math.sin(0.35 * t + 0.11 * c + layer * 0.3 + ai) * 0.8 + noise);
  v *= 1 + layer / 14;
  if (t === 0) v *= 3; // first token is special
  if (c === 7 || c === 401 % dim) v += 25 * Math.sign(Math.sin(t * 0.7 + 1));
  if (id === "silu" || id === "swiglu") v = Math.abs(v) * (c % 3 === 0 ? 1 : 0.1);
  return v;
}

const cache = new Map();
function tensor(id, layer, T) {
  const key = `${id}/${layer}/${T}`;
  let x = cache.get(key);
  if (!x) {
    const dim = byId[id].dim;
    x = new Float32Array(T * dim);
    for (let t = 0; t < T; t++) for (let c = 0; c < dim; c++) x[t * dim + c] = value(id, layer, t, c);
    cache.set(key, x);
  }
  return x;
}

function rankOrder(id, layer, T, order) {
  const dim = byId[id].dim;
  const perm = Array.from({ length: dim }, (_, i) => i);
  if (order === "natural") return perm;
  const x = tensor(id, layer, T);
  const score = new Float64Array(dim);
  for (let c = 0; c < dim; c++) {
    let mx = 0, s = 0, s2 = 0;
    for (let t = 0; t < T; t++) { const v = x[t * dim + c]; mx = Math.max(mx, Math.abs(v)); s += v; s2 += v * v; }
    score[c] = order === "absmax" ? mx : order === "std" ? Math.sqrt(Math.max(0, s2 / T - (s / T) ** 2)) : Math.abs(s / T);
  }
  return perm.sort((a, b) => score[b] - score[a] || a - b);
}

// ---------- wire ----------
export function frame(meta, arr) {
  const f32 = arr instanceof Float32Array ? arr : Float32Array.from(arr);
  const body = Buffer.from(JSON.stringify({ ...meta, shape: [f32.length] }));
  const pad = (4 - ((4 + body.length) % 4)) % 4;
  const head = Buffer.alloc(4 + body.length + pad);
  head.writeUInt32LE(body.length, 0);
  body.copy(head, 4);
  return Buffer.concat([head, Buffer.from(f32.buffer, f32.byteOffset, f32.byteLength)]);
}

const clampWin = (lo, hi, n) => {
  lo = Math.max(0, Math.min(Math.floor(lo), n - 1));
  hi = Math.max(lo + 1, Math.min(Math.floor(hi), n));
  return [lo, hi];
};

function pool(get, rows, cols, bh, bw, agg) {
  const R = Math.ceil(rows / bh), C = Math.ceil(cols / bw);
  const out = new Float32Array(R * C);
  for (let r = 0; r < R; r++) for (let c = 0; c < C; c++) {
    let best = null, sum = 0, n = 0;
    for (let i = r * bh; i < Math.min(rows, (r + 1) * bh); i++) for (let j = c * bw; j < Math.min(cols, (c + 1) * bw); j++) {
      const v = get(i, j); n++; sum += v;
      if (best === null || (agg === "absmax" ? Math.abs(v) > Math.abs(best) : agg === "max" ? v > best : agg === "min" ? v < best : false)) best = v;
    }
    out[r * C + c] = agg === "mean" ? sum / n : best;
  }
  return { out, R, C };
}

function range(x) {
  const s = Float32Array.from(x).sort();
  const q = (p) => s[Math.min(s.length - 1, Math.floor(p * (s.length - 1)))];
  return { min: s[0], max: s[s.length - 1], p005: q(0.005), p01: q(0.01), p99: q(0.99), p995: q(0.995), absmax: Math.max(Math.abs(s[0]), Math.abs(s[s.length - 1])) };
}

function summarize(values, bins = 64, clip = false) {
  const v = Float64Array.from(values), n = v.length;
  const mean = v.reduce((a, b) => a + b, 0) / n;
  const std = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / n) || 1e-9;
  const s = Float64Array.from(v).sort();
  const pq = (p) => s[Math.min(n - 1, Math.floor((p / 100) * (n - 1)))];
  const pcts = [0.1, 1, 5, 25, 50, 75, 95, 99, 99.9];
  const lo = clip ? pq(0.1) : s[0];
  let hi = clip ? pq(99.9) : s[n - 1];
  if (hi <= lo) hi = lo + 1e-9;
  const counts = new Array(bins).fill(0);
  let inside = 0;
  for (const x of v) if (x >= lo && x <= hi) { counts[Math.min(bins - 1, Math.floor(((x - lo) / (hi - lo)) * bins))]++; inside++; }
  return {
    n, mean, std, min: s[0], max: s[n - 1], absmax: Math.max(-s[0], s[n - 1]), l2: Math.sqrt(v.reduce((a, b) => a + b * b, 0)),
    skew: v.reduce((a, b) => a + ((b - mean) / std) ** 3, 0) / n, kurtosis: v.reduce((a, b) => a + ((b - mean) / std) ** 4, 0) / n - 3,
    frac_pos: v.filter((x) => x > 0).length / n, frac_3sigma: v.filter((x) => Math.abs(x - mean) > 3 * std).length / n,
    percentiles: Object.fromEntries(pcts.map((p) => [String(p), pq(p)])),
    hist: { counts, edges: Array.from({ length: bins + 1 }, (_, i) => lo + ((hi - lo) * i) / bins), n_clipped: n - inside },
  };
}

// ---------- attention ----------
function attnP(layer, head, q, k) {
  if (k > q) return 0;
  const sink = head % 4 === 0 ? 0.6 : 0.05;
  const w = (kk) => Math.exp(-(q - kk) / (1 + (head % 5) * 1.5 + layer * 0.05)) + (kk === 0 ? sink : 0) + 0.001;
  let z = 0;
  for (let kk = 0; kk <= q; kk++) z += w(kk);
  return w(k) / z;
}
function attnMetrics(layer, head, T) {
  let ent = 0, first = 0, dist = 0;
  for (let q = 0; q < T; q++) for (let k = 0; k <= q; k++) {
    const p = attnP(layer, head, q, k);
    if (p > 0) ent -= p * Math.log(p);
    dist += p * (q - k);
    if (k === 0 && q > 0) first += p;
  }
  return [ent / T, first / Math.max(1, T - 1), dist / T];
}
const ATTN_STATS = ["entropy", "first_token", "distance"];

// ---------- routes ----------
export async function installMock(page, opts = {}) {
  const state = { requests: [], run: null, text: PROMPT, ...opts };
  const json = (route, body, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
  const bin = (route, meta, arr) => route.fulfill({ status: 200, contentType: "application/octet-stream", body: frame(meta, arr) });

  await page.route("**/api/**", async (route) => {
    const req = route.request();
    const u = new URL(req.url());
    const p = u.pathname;
    const q = Object.fromEntries(u.searchParams);
    state.requests.push({ method: req.method(), path: p, q });
    const num = (k, d) => (q[k] === undefined ? d : Number(q[k]));
    try {
      if (p === "/api/status") {
        return json(route, {
          state: "ready", model_id: "Qwen/Qwen3-0.6B", target: "Qwen/Qwen3-0.6B", error: null,
          info: { model_id: "Qwen/Qwen3-0.6B", device: "mps", dtype: "float32", n_layers: L, hidden_size: 1024, intermediate_size: 3072, n_heads: NH, n_kv_heads: 8, params_m: 596, arch: "qwen3" },
          presets: [{ id: "Qwen/Qwen3-0.6B", label: "Qwen3-0.6B" }, { id: "Qwen/Qwen2.5-0.5B", label: "Qwen2.5-0.5B" }],
        });
      }
      if (p === "/api/corpus") return json(route, [{ id: "mock-1", title: "Paris and code", category: "Mock", text: PROMPT }]);
      if (p === "/api/run" && req.method() === "POST") {
        const body = JSON.parse(req.postData() ?? "{}");
        const tokens = tokenize(body.text || PROMPT);
        state.run = { tokens };
        return json(route, {
          run_id: "mock-run", model_id: "Qwen/Qwen3-0.6B", tokens, token_ids: tokens.map((_, i) => 1000 + i), truncated: false, elapsed_ms: 3.2,
          model: MODEL, activations: opts.activations ?? ACTIVATIONS,
        });
      }
      const m = /^\/api\/run\/[^/]+\/(\w+)$/.exec(p);
      if (!m) return json(route, { detail: "not found" }, 404);
      const T = state.run?.tokens.length ?? 64;
      const what = m[1];
      if (what.startsWith("attn")) return attnRoute(what);
      const id = q.act;
      const info = byId[id];
      if (!info || info.kind !== "token") return json(route, { detail: `unknown act ${id}` }, 400);
      const layer = num("layer", 0);
      const dim = info.dim;

      if (what === "overview") {
        const stat = q.stat ?? "norm", d = Math.max(0, Math.min(dim - 1, num("dim", 0)));
        const out = new Float32Array(L * T);
        for (let l = 0; l < L; l++) {
          const x = tensor(id, l, T);
          for (let t = 0; t < T; t++) {
            let s = 0, s2 = 0, mx = 0, s4 = 0;
            for (let c = 0; c < dim; c++) { const v = x[t * dim + c]; s += v; s2 += v * v; mx = Math.max(mx, Math.abs(v)); }
            const mean = s / dim, varr = s2 / dim - mean * mean;
            for (let c = 0; c < dim; c++) s4 += (x[t * dim + c] - mean) ** 4;
            out[l * T + t] = stat === "norm" ? Math.sqrt(s2) : stat === "absmax" ? mx : stat === "mean" ? mean : stat === "std" ? Math.sqrt(varr) : stat === "kurtosis" ? s4 / dim / Math.max(varr * varr, 1e-20) - 3 : x[t * dim + d];
          }
        }
        const r = range(out);
        return bin(route, { act: id, stat, dim: d, rows: L, cols: T, layer_labels: layerLabels, range: { min: r.min, max: r.max, p01: r.p01, p99: r.p99 } }, out);
      }
      if (layer < 0 || layer >= L) return json(route, { detail: "layer out of range" }, 400);
      const order = q.order ?? "natural";
      const perm = rankOrder(id, layer, T, order);
      const x = tensor(id, layer, T);
      if (what === "slice") {
        const [t0, t1] = clampWin(num("t0", 0), num("t1", 64), T);
        const [d0, d1] = clampWin(num("d0", 0), num("d1", 128), dim);
        const bh = Math.max(1, Math.ceil((t1 - t0) / num("max_h", 512))), bw = Math.max(1, Math.ceil((d1 - d0) / num("max_w", 1024)));
        const cols = perm.slice(d0, d1);
        const { out, R, C } = pool((i, j) => x[(t0 + i) * dim + cols[j]], t1 - t0, d1 - d0, bh, bw, q.agg ?? "absmax");
        return bin(route, { act: id, layer, order, agg: q.agg ?? "absmax", t0, t1, d0, d1, n_tokens: T, n_dims: dim, bh, bw, rows: R, cols: C, dims: cols, range: range(x) }, out);
      }
      if (what === "profile") {
        const bins = num("bins", 512), bw = Math.max(1, Math.ceil(dim / bins));
        const prof = new Float32Array(Math.ceil(dim / bw));
        perm.forEach((c, r) => { let mx = 0; for (let t = 0; t < T; t++) mx = Math.max(mx, Math.abs(x[t * dim + c])); const b = Math.floor(r / bw); prof[b] = Math.max(prof[b], mx); });
        return bin(route, { act: id, layer, order, bw, n_dims: dim, cols: prof.length }, prof);
      }
      if (what === "stats") {
        const [t0, t1] = clampWin(num("t0", 0), num("t1", 64), T);
        const [d0, d1] = clampWin(num("d0", 0), num("d1", 128), dim);
        const vals = [];
        for (let t = t0; t < t1; t++) for (let c = d0; c < d1; c++) vals.push(x[t * dim + perm[c]]);
        return json(route, { ...summarize(vals, 64, q.clip === "true"), top: [], region: { t0, t1, d0, d1 } });
      }
      return json(route, { detail: "not found" }, 404);

      function attnRoute(kind) {
        if (kind === "attn_overview") {
          const st = ATTN_STATS.indexOf(q.stat ?? "entropy");
          const out = new Float32Array(L * NH);
          for (let l = 0; l < L; l++) for (let h = 0; h < NH; h++) out[l * NH + h] = attnMetrics(l, h, T)[st];
          const r = range(out);
          return bin(route, { stat: q.stat ?? "entropy", rows: L, cols: NH, range: { min: r.min, max: r.max } }, out);
        }
        const layerA = num("layer", 0), head = num("head", -1);
        if (kind === "attn") {
          const [q0, q1] = clampWin(num("q0", 0), num("q1", 100000), T), [k0, k1] = clampWin(num("k0", 0), num("k1", 100000), T);
          const bq = Math.max(1, Math.ceil((q1 - q0) / num("max_q", 512))), bk = Math.max(1, Math.ceil((k1 - k0) / num("max_k", 512)));
          const heads = head < 0 ? Array.from({ length: NH }, (_, i) => i) : [head];
          const parts = heads.map((h) => pool((i, j) => attnP(layerA, h, q0 + i, k0 + j), q1 - q0, k1 - k0, bq, bk, q.agg ?? "max"));
          const R = parts[0].R, C = parts[0].C, out = new Float32Array(parts.length * R * C);
          parts.forEach((pt, i) => out.set(pt.out, i * R * C));
          const mets = heads.map((h) => attnMetrics(layerA, h, T));
          return bin(route, { layer: layerA, heads, q0, q1, k0, k1, n_tokens: T, bq, bk, rows: R, cols: C, agg: q.agg ?? "max", metrics: Object.fromEntries(ATTN_STATS.map((n, i) => [n, mets.map((mm) => mm[i])])) }, out);
        }
        if (kind === "attn_stats") {
          const [q0, q1] = clampWin(num("q0", 0), num("q1", 100000), T), [k0, k1] = clampWin(num("k0", 0), num("k1", 100000), T);
          const vals = [];
          for (let a = q0; a < q1; a++) for (let b = k0; b < k1; b++) if (b <= a) vals.push(attnP(layerA, head, a, b));
          if (!vals.length) return json(route, { detail: "region contains only masked (future) positions" }, 400);
          return json(route, { ...summarize(vals, 64, q.clip === "true"), region: { q0, q1, k0, k1 } });
        }
        return json(route, { detail: "not found" }, 404);
      }
    } catch (e) {
      return json(route, { detail: String(e) }, 500);
    }
  });
  return state;
}
