/**
 * Throwaway harness for the distribution panel (NOT part of the production build: `npm run build` only bundles index.html).
 *
 * Run:
 *   cd frontend && PORT=5203 npx vite --host 127.0.0.1
 *   open http://127.0.0.1:5203/e2e/distribution-harness.html
 * Automated check with mocked /stats and /axis_stats (Playwright, headless Chrome):
 *   node e2e/distribution.mjs <screenshot-dir>
 *
 * The page mounts <DistributionPanel> in a 340 px side column and exposes buttons to change the props
 * (layer, order, selection, cursor, head dim, act). Callback invocations are appended to #calls and window.__calls.
 */
import { StrictMode, useState } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { DistributionPanel } from "../src/components/DistributionPanel";
import "../src/styles.css";

const client = new QueryClient({ defaultOptions: { queries: { refetchOnWindowFocus: false, retry: 0, staleTime: Infinity, gcTime: 5 * 60_000 } } });
const TOKENS = Array.from({ length: 48 }, (_, i) => (i === 0 ? "<|bos|>" : ["The", " capital", " of", " France", " is", "\n", " Paris", ".", " def", "  ", "(x", "):"][i % 12]));
(window as unknown as { __calls: string[] }).__calls = [];

function Harness() {
  const [layer, setLayer] = useState(3);
  const [order, setOrder] = useState<"natural" | "absmax">("natural");
  const [sel, setSel] = useState(false);
  const [cur, setCur] = useState(false);
  const [headDim, setHeadDim] = useState<number | null>(null);
  const [act, setAct] = useState("resid_pre");
  const [calls, setCalls] = useState<string[]>([]);
  const log = (s: string) => {
    (window as unknown as { __calls: string[] }).__calls.push(s);
    setCalls((c) => [...c.slice(-5), s]);
  };
  const cursorRank = order === "absmax" ? 0 : 37;
  return (
    <div style={{ display: "flex", gap: 16, padding: 16, alignItems: "flex-start" }}>
      <div style={{ display: "flex", flexDirection: "column", gap: 6, width: 260 }} id="controls">
        <b>props</b>
        <button id="layer" onClick={() => setLayer((l) => l + 1)}>layer {layer}</button>
        <button id="order" onClick={() => setOrder((o) => (o === "natural" ? "absmax" : "natural"))}>order {order}</button>
        <button id="sel" onClick={() => setSel((s) => !s)}>selection {String(sel)}</button>
        <button id="cur" onClick={() => setCur((s) => !s)}>cursor {String(cur)}</button>
        <button id="head" onClick={() => setHeadDim((h) => (h ? null : 64))}>headDim {String(headDim)}</button>
        <button id="act" onClick={() => setAct((a) => (a === "resid_pre" ? "boom" : "resid_pre"))}>act {act}</button>
        <b>callbacks</b>
        <pre id="calls" style={{ fontSize: 11, minHeight: 80 }}>{calls.join("\n")}</pre>
      </div>
      <div style={{ width: 340 }}>
        <DistributionPanel
          runId="mock"
          act={act}
          actLabel="resid_pre — attn_norm input"
          layer={layer}
          layerLabel={String(layer)}
          nTokens={48}
          nChannels={256}
          order={order}
          tokens={TOKENS}
          headDim={headDim}
          window={{ t0: 0, t1: 32, d0: 0, d1: 128 }}
          selection={sel ? { t0: 2, t1: 10, d0: 30, d1: 50 } : null}
          cursor={cur ? { token: 5, channel: cursorRank } : null}
          cursorChannelId={cur ? 37 : null}
          exportName="actlens_mock_resid_pre_L3"
          onPickToken={(t) => log(`onPickToken(${t})`)}
          onPickChannel={(r) => log(`onPickChannel(${r})`)}
        />
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={client}>
      <Harness />
    </QueryClientProvider>
  </StrictMode>,
);
