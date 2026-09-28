export type ActId = string;
/** Display group of the activation picker, chosen by the architecture adapter ("Residual", "Attention", "MLP", ...). */
export type ActGroup = string;
export type ActKind = "token" | "attn";
export type AggKind = "absmax" | "mean" | "max" | "min";
export type OrderKind = "natural" | "absmax" | "std" | "mean_abs";
export type OverviewStat = "norm" | "absmax" | "mean" | "std" | "kurtosis" | "dim";
export type AttnStat = "entropy" | "first_token" | "distance";

/** One capturable activation, as listed by /api/run. */
export interface ActivationInfo {
  id: ActId;
  label: string;
  group: ActGroup;
  kind: ActKind;
  n_layers: number;
  /** Channel count C for token activations, null for attention patterns. */
  dim: number | null;
  layer_labels: string[];
  /** Non-null only when the channel axis is heads x head_dim (for attn_pattern: nH, with head_dim null). */
  n_heads: number | null;
  head_dim: number | null;
  description: string;
}

export interface RunModelDims {
  n_layers: number;
  hidden_size: number;
  intermediate_size: number;
  n_heads: number;
  n_kv_heads: number;
  head_dim: number;
}

export interface RunInfo {
  run_id: string;
  model_id: string;
  tokens: string[];
  token_ids: number[];
  truncated: boolean;
  model: RunModelDims;
  activations: ActivationInfo[];
}

export interface ModelInfo {
  model_id: string;
  device: string;
  dtype: string;
  n_layers: number;
  hidden_size: number;
  intermediate_size: number;
  n_heads: number;
  n_kv_heads: number;
  params_m: number;
  arch: string;
}

export interface Status {
  state: "idle" | "loading" | "ready" | "error";
  model_id: string | null;
  target: string | null;
  error: string | null;
  info: ModelInfo | null;
  presets: { id: string; label: string }[];
}

export interface CorpusItem {
  id: string;
  title: string;
  category: string;
  text: string;
}

export interface ValueRange {
  min: number;
  max: number;
  p01: number;
  p99: number;
  p005?: number;
  p995?: number;
  absmax?: number;
}

export interface SliceMeta {
  act: ActId;
  layer: number;
  order: OrderKind;
  agg: AggKind;
  t0: number;
  t1: number;
  d0: number;
  d1: number;
  n_tokens: number;
  n_dims: number;
  bh: number;
  bw: number;
  rows: number;
  cols: number;
  dims: number[];
  range: ValueRange;
  shape: number[];
}

export interface OverviewMeta {
  act: ActId;
  stat: OverviewStat;
  rows: number;
  cols: number;
  layer_labels: string[];
  range: ValueRange;
  shape: number[];
}

export interface ProfileMeta {
  bw: number;
  n_dims: number;
  cols: number;
  shape: number[];
}

export interface AttnMeta {
  layer: number;
  heads: number[];
  q0: number;
  q1: number;
  k0: number;
  k1: number;
  n_tokens: number;
  bq: number;
  bk: number;
  rows: number;
  cols: number;
  agg: string;
  metrics: Record<AttnStat, number[]>;
  shape: number[];
}

export interface AttnOverviewMeta {
  stat: AttnStat;
  rows: number;
  cols: number;
  range: { min: number; max: number };
  shape: number[];
}

export interface Stats {
  n: number;
  mean: number;
  std: number;
  min: number;
  max: number;
  absmax: number;
  l2: number;
  skew: number;
  kurtosis: number;
  frac_pos: number;
  frac_3sigma: number;
  percentiles: Record<string, number>;
  hist: { counts: number[]; edges: number[]; n_clipped: number };
  top?: { token: number; dim: number; value: number }[];
  region: Record<string, number>;
}

export interface Frame<M> {
  meta: M;
  data: Float32Array;
}

/** Inverse of backend/actlens/wire.py: [u32 json_len][json][pad to 4][float32 payload]. */
export function decodeFrame<M>(buf: ArrayBuffer): Frame<M> {
  const jsonLen = new DataView(buf).getUint32(0, true);
  const meta = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, jsonLen))) as M;
  const offset = 4 + jsonLen + ((4 - ((4 + jsonLen) % 4)) % 4);
  return { meta, data: new Float32Array(buf, offset) };
}

type Params = Record<string, string | number | boolean | undefined>;

function url(path: string, params?: Params): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params ?? {})) if (v !== undefined) q.set(k, String(v));
  const s = q.toString();
  return s ? `${path}?${s}` : path;
}

async function check(res: Response): Promise<Response> {
  if (res.ok) return res;
  let detail = res.statusText;
  try {
    const body = await res.json();
    if (typeof body.detail === "string") detail = body.detail;
  } catch {
    /* non-JSON error body */
  }
  throw new Error(`${res.status}: ${detail}`);
}

export async function getJSON<T>(path: string, params?: Params, signal?: AbortSignal): Promise<T> {
  return (await check(await fetch(url(path, params), { signal }))).json();
}

export async function getFrame<M>(path: string, params: Params, signal?: AbortSignal): Promise<Frame<M>> {
  const res = await check(await fetch(url(path, params), { signal }));
  return decodeFrame<M>(await res.arrayBuffer());
}

export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await check(
    await fetch(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }),
  );
  return res.json();
}

export const runPath = (runId: string, what: string) => `/api/run/${runId}/${what}`;
