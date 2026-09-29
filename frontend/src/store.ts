import { create } from "zustand";
import type { ActId, ActivationInfo, RunInfo } from "./api";
import {
  clampLayer,
  findAct,
  homeViewport,
  initialCore,
  jumpToHead,
  openCell,
  pickChannel,
  pickToken,
  resetForRun,
  switchActivation,
  type AcrossState,
  type ActMemory,
  type AttnState,
  type Mode,
  type TokenView,
} from "./viewState";

export {
  ACT_DISPLAY,
  ATTN_DISPLAY,
  SEQ_DISPLAY,
  DEFAULT_CHANNELS,
  DEFAULT_TOKENS,
  type Mode,
  type TokenCursor,
  type TokenView,
} from "./viewState";

interface State {
  run: RunInfo | null;
  prompt: string;
  maxTokens: number;
  hoverRow: number | null;

  /** Selection shared by all modes. */
  act: ActId;
  layer: number;
  mode: Mode;
  /** Token x channel view of the current token activation ("Layer" mode). */
  tok: TokenView;
  /** Per-activation memory of display / order / pooling / window. */
  memory: ActMemory;
  across: AcrossState;
  attn: AttnState;

  setPrompt: (s: string) => void;
  setMaxTokens: (n: number) => void;
  setHoverRow: (r: number | null) => void;
  setRun: (r: RunInfo) => void;
  setAct: (id: ActId) => void;
  setLayer: (layer: number) => void;
  setMode: (m: Mode) => void;
  patchTok: (p: Partial<TokenView>) => void;
  patchAcross: (p: Partial<AcrossState>) => void;
  patchAttn: (p: Partial<AttnState>) => void;
  /** Across-layers click: set the layer, go back to Layer mode with the token cursor placed. */
  openCell: (layer: number, token: number) => void;
  jumpToHead: (head: number) => void;
  pickToken: (token: number) => void;
  pickChannel: (rank: number) => void;
  clearSelection: () => void;
}

export const useStore = create<State>((set, get) => {
  const info = (): ActivationInfo | undefined => findAct(get().run, get().act);
  return {
    run: null,
    prompt: "",
    maxTokens: 512,
    hoverRow: null,
    ...initialCore(),
    mode: "layer",

    setPrompt: (prompt) => set({ prompt }),
    setMaxTokens: (maxTokens) => set({ maxTokens }),
    setHoverRow: (hoverRow) => set({ hoverRow }),

    setRun: (run) => set({ run, hoverRow: null, ...resetForRun(run, get()) }),

    setAct: (id) => {
      const { run, act, tok, memory, across } = get();
      const to = findAct(run, id);
      if (!run || !to || id === act) return;
      const sw = switchActivation(tok, memory, findAct(run, act), to, run.tokens.length);
      set({
        act: id,
        layer: clampLayer(get().layer, to),
        // attention patterns have no "across layers" map (the layer x head map replaces it)
        mode: to.kind === "attn" && get().mode === "across" ? "layer" : get().mode,
        tok: sw.tok,
        memory: sw.memory,
        across: { ...across, vp: homeViewport(to.n_layers, run.tokens.length) },
      });
    },
    setLayer: (layer) => {
      const a = info();
      if (a) set({ layer: clampLayer(layer, a) });
    },
    setMode: (mode) => {
      if (mode === "across" && info()?.kind !== "token") return;
      set({ mode });
    },
    patchTok: (p) => set((s) => ({ tok: { ...s.tok, ...p } })),
    patchAcross: (p) => set((s) => ({ across: { ...s.across, ...p } })),
    patchAttn: (p) => set((s) => ({ attn: { ...s.attn, ...p } })),

    openCell: (layer, token) => {
      const { run, across, tok } = get();
      const a = info();
      if (!run || !a || a.kind !== "token") return;
      const channel = across.stat === "dim" && tok.order === "natural" ? across.channel : null;
      set({ layer: clampLayer(layer, a), mode: "layer", tok: openCell(tok, a, run.tokens.length, token, channel) });
    },
    jumpToHead: (head) => {
      const { run, tok } = get();
      const a = info();
      if (run && a) set({ tok: jumpToHead(tok, a, run.tokens.length, head) });
    },
    pickToken: (token) => {
      const { run, tok } = get();
      const a = info();
      if (run && a?.kind === "token") set({ tok: pickToken(tok, run.tokens.length, a.dim ?? 0, token) });
    },
    pickChannel: (rank) => {
      const { run, tok } = get();
      const a = info();
      if (run && a?.kind === "token") set({ tok: pickChannel(tok, run.tokens.length, a.dim ?? 0, rank) });
    },
    clearSelection: () => {
      if (info()?.kind === "attn") get().patchAttn({ brush: null, cursor: null });
      else get().patchTok({ brush: null, cursor: null });
    },
  };
});
