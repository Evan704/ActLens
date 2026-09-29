import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { getJSON, type Status } from "./api";
import { LoadStatus } from "./components/LoadStatus";
import { ModelBar } from "./components/ModelBar";
import { PromptPanel } from "./components/PromptPanel";
import { useStore } from "./store";
import { ActivationView } from "./views/ActivationView";

export default function App() {
  const status = useQuery({
    queryKey: ["status"],
    queryFn: ({ signal }) => getJSON<Status>("/api/status", undefined, signal),
    refetchInterval: (q) => (!q.state.data ? 1000 : q.state.data.state === "loading" ? 500 : false),
    retry: true,
    staleTime: 0,
  });
  const run = useStore((s) => s.run);
  const [collapsed, setCollapsed] = useState(() => {
    try { return localStorage.getItem("actlens.promptCollapsed") === "1"; } catch { return false; }
  });
  const toggle = () => setCollapsed((c) => {
    try { localStorage.setItem("actlens.promptCollapsed", c ? "0" : "1"); } catch { /* storage unavailable */ }
    return !c;
  });

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(el.tagName)) return;
      const st = useStore.getState();
      if (e.key === "Escape") st.clearSelection();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const ready = status.data?.state === "ready";
  const active = run && ready && run.model_id === status.data?.model_id ? run : null;

  return (
    <div className="app">
      <ModelBar status={status.data} />
      <div className={"body" + (collapsed ? " collapsed" : "")}>
        <PromptPanel status={status.data} />
        <button
          className="panel-toggle"
          onClick={toggle}
          aria-expanded={!collapsed}
          aria-label={collapsed ? "Show prompt panel" : "Hide prompt panel"}
          title={collapsed ? "Show prompt panel" : "Hide prompt panel"}
        >{collapsed ? "›" : "‹"}</button>
        <section className="center">
          <div className="content">
            {!active ? (
              <div className="empty">
                <h2>Look inside a forward pass</h2>
                <p>Pick a sample prompt or type your own, then run it. ActLens captures activations on demand and lets you browse any of them, at any layer, as heatmaps.</p>
                <LoadStatus status={status.data} />
              </div>
            ) : (
              <ActivationView />
            )}
          </div>
        </section>
      </div>
    </div>
  );
}
