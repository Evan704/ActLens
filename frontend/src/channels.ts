/** Pure helpers for activation channels: head-structured naming, head jumps, picker labels. */
import type { ActivationInfo } from "./api";
import { clampViewport, type Extent, type Viewport } from "./viewport";

export interface HeadLayout {
  nHeads: number;
  headDim: number;
}

/** Head structure of a token activation whose channel axis is heads x head_dim; null otherwise. */
export function headLayout(info: Pick<ActivationInfo, "kind" | "n_heads" | "head_dim">): HeadLayout | null {
  if (info.kind !== "token" || !info.n_heads || !info.head_dim) return null;
  return { nHeads: info.n_heads, headDim: info.head_dim };
}

export function channelHead(channel: number, headDim: number): { head: number; dim: number } {
  return { head: Math.floor(channel / headDim), dim: channel % headDim };
}

/** Axis tick label of a channel: "h3·17" for head-structured activations, else the plain id. */
export function channelLabel(channel: number, headDim: number | null): string {
  if (!headDim) return String(channel);
  const { head, dim } = channelHead(channel, headDim);
  return `h${head}·${dim}`;
}

/** Tooltip line for a channel; `rank` is only shown when a ranking reorders the columns. */
export function channelTooltip(channel: number, headDim: number | null, rank?: number): string {
  const suffix = rank !== undefined ? `  (rank ${rank})` : "";
  if (!headDim) return `channel ${channel}${suffix}`;
  const { head, dim } = channelHead(channel, headDim);
  return `head ${head}, dim ${dim} (channel ${channel})${suffix}`;
}

export const clampHead = (head: number, nHeads: number) => Math.max(0, Math.min(nHeads - 1, Math.round(head)));

/** Window covering exactly one head's channels; the token range is untouched. */
export function headWindow(vp: Viewport, head: number, layout: HeadLayout, ext: Extent): Viewport {
  const h = clampHead(head, layout.nHeads);
  return clampViewport({ ...vp, x0: h * layout.headDim, x1: (h + 1) * layout.headDim }, ext, 1);
}

/** The head under the centre of the window (the head itself when the window is one head). */
export function windowHead(vp: Viewport, layout: HeadLayout): number {
  return clampHead(Math.floor((vp.x0 + vp.x1) / 2 / layout.headDim), layout.nHeads);
}

/** Label for the activation <select>: "label · 2048 ch" / "label · 16×128" / "label · 16 heads". */
export function actOptionLabel(a: ActivationInfo): string {
  if (a.kind === "attn") return `${a.label} · ${a.n_heads ?? "?"} heads`;
  const shape = a.n_heads && a.head_dim ? `${a.n_heads}×${a.head_dim}` : `${a.dim} ch`;
  return `${a.label} · ${shape}`;
}

/** Group activations for <optgroup>s, keeping the order the backend sent. */
export function groupActivations(acts: ActivationInfo[]): { group: string; items: ActivationInfo[] }[] {
  const out: { group: string; items: ActivationInfo[] }[] = [];
  for (const a of acts) {
    let g = out.find((x) => x.group === a.group);
    if (!g) out.push((g = { group: a.group, items: [] }));
    g.items.push(a);
  }
  return out;
}
