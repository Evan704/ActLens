import { useRef } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { postJSON, type Status } from "../api";

const fmtBytes = (n: number) =>
  n >= 2 ** 30 ? `${(n / 2 ** 30).toFixed(2)} GB` : n >= 2 ** 20 ? `${(n / 2 ** 20).toFixed(0)} MB` : `${(n / 2 ** 10).toFixed(0)} KB`;

const STAGE_LABEL = {
  resolve: "Checking the model…",
  download: "Downloading weights",
  load: "Loading weights into memory…",
} as const;

/** Progress bar while a model loads, and an explanation with a retry button when it fails. */
export function LoadStatus({ status }: { status: Status | undefined }) {
  const qc = useQueryClient();
  const last = useRef<{ t: number; done: number; rate: number }>({ t: 0, done: 0, rate: 0 });
  const retry = useMutation({
    mutationFn: (model_id: string) => postJSON<Status>("/api/models/load", { model_id }),
    onSettled: () => qc.invalidateQueries({ queryKey: ["status"] }),
  });

  if (status?.state === "error") {
    return (
      <div className="loadstatus error" role="alert">
        <div className="headline">{status.error}</div>
        {status.error_hint && <p>{status.error_hint}</p>}
        {status.error_detail && (
          <details>
            <summary>Details</summary>
            <pre>{status.error_detail}</pre>
          </details>
        )}
        {status.target && (
          <button className="primary" disabled={retry.isPending} onClick={() => retry.mutate(status.target!)}>
            Try again
          </button>
        )}
      </div>
    );
  }
  if (status?.state !== "loading") return null;

  const p = status.progress;
  const stage = p?.stage ?? "resolve";
  const determinate = stage === "download" && !!p?.total;
  const frac = determinate ? Math.min(1, p!.done / p!.total!) : 0;
  let rate = 0;
  if (determinate) {
    const now = performance.now();
    const l = last.current;
    if (now - l.t > 1500 || p!.done < l.done) {
      if (l.t && p!.done >= l.done) l.rate = (p!.done - l.done) / ((now - l.t) / 1000);
      l.t = now;
      l.done = p!.done;
    }
    rate = l.rate;
  }
  return (
    <div className="loadstatus" role="status">
      <div className="headline">
        {STAGE_LABEL[stage]} <span className="muted">{status.target}</span>
      </div>
      <div className={`bar ${determinate ? "" : "indeterminate"}`} role="progressbar" aria-valuemin={0} aria-valuemax={100} aria-valuenow={determinate ? Math.round(frac * 100) : undefined}>
        <div style={determinate ? { width: `${frac * 100}%` } : undefined} />
      </div>
      <div className="muted small">
        {determinate
          ? `${fmtBytes(p!.done)} / ${fmtBytes(p!.total!)} · ${Math.round(frac * 100)}%${rate > 0 ? ` · ${fmtBytes(rate)}/s` : ""}`
          : stage === "download" ? "Starting download…" : stage === "load" ? "Reading the checkpoint and moving it to the device. Large models can take a minute." : ""}
        {p ? ` · ${Math.round(p.elapsed_s)} s` : ""}
      </div>
    </div>
  );
}
