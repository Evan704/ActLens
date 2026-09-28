import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { postJSON, type Status } from "../api";
import { useStore } from "../store";
import { Logo } from "./Logo";

export function ModelBar({ status }: { status: Status | undefined }) {
  const qc = useQueryClient();
  const [custom, setCustom] = useState("");
  const load = useMutation({
    mutationFn: (model_id: string) => postJSON<Status>("/api/models/load", { model_id }),
    onMutate: () => useStore.setState({ run: null }),
    onSettled: () => qc.invalidateQueries({ queryKey: ["status"] }),
  });
  const cur = status?.model_id ?? "";
  const presets = status?.presets ?? [];
  const busy = status?.state === "loading" || load.isPending;
  const info = status?.info;
  return (
    <header className="modelbar">
      <div className="brand"><Logo /><span>Act<span className="lens">Lens</span></span></div>
      <label className="inline">
        Model
        <select value={presets.some((p) => p.id === cur) ? cur : ""} disabled={busy} onChange={(e) => e.target.value && load.mutate(e.target.value)}>
          {!presets.some((p) => p.id === cur) && <option value="">{cur || "—"}</option>}
          {presets.map((p) => (
            <option key={p.id} value={p.id}>{p.label}</option>
          ))}
        </select>
      </label>
      <form
        className="inline"
        onSubmit={(e) => {
          e.preventDefault();
          if (custom.trim()) load.mutate(custom.trim());
        }}
      >
        <input value={custom} placeholder="or any HF model id" onChange={(e) => setCustom(e.target.value)} disabled={busy} />
        <button disabled={busy || !custom.trim()}>Load</button>
      </form>
      <div className={`pill ${status?.state ?? "idle"}`}>
        {status?.state === "loading" && `loading ${status.target}…`}
        {status?.state === "ready" && info && `${info.arch} · ${info.params_m}M · ${info.n_layers}L · d=${info.hidden_size} · ${info.device}/${info.dtype}`}
        {status?.state === "error" && <span title={status.error ?? ""}>load failed: {status.error?.slice(0, 90)}</span>}
        {(!status || status.state === "idle") && "connecting…"}
      </div>
      {load.error && <span className="err">{(load.error as Error).message}</span>}
    </header>
  );
}
