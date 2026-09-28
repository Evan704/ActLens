import { useMutation, useQuery } from "@tanstack/react-query";
import { getJSON, postJSON, type CorpusItem, type RunInfo, type Status } from "../api";
import { fmtToken } from "../tokens";
import { useStore } from "../store";
import { NumberBox } from "./controls";

export function PromptPanel({ status }: { status: Status | undefined }) {
  const run = useStore((s) => s.run);
  const prompt = useStore((s) => s.prompt);
  const setPrompt = useStore((s) => s.setPrompt);
  const maxTokens = useStore((s) => s.maxTokens);
  const setMaxTokens = useStore((s) => s.setMaxTokens);
  const setRun = useStore((s) => s.setRun);
  const hoverRow = useStore((s) => s.hoverRow);
  const setHoverRow = useStore((s) => s.setHoverRow);
  const corpus = useQuery({ queryKey: ["corpus"], queryFn: () => getJSON<CorpusItem[]>("/api/corpus"), staleTime: Infinity });
  const ready = status?.state === "ready";
  const mut = useMutation({
    mutationFn: () => postJSON<RunInfo>("/api/run", { text: prompt, max_tokens: maxTokens }),
    onSuccess: setRun,
  });
  const groups = new Map<string, CorpusItem[]>();
  for (const c of corpus.data ?? []) groups.set(c.category, [...(groups.get(c.category) ?? []), c]);
  const canRun = ready && prompt.trim().length > 0 && !mut.isPending;

  return (
    <div className="prompt-panel">
      <label className="field">
        Corpus
        <select
          value=""
          onChange={(e) => {
            const item = corpus.data?.find((c) => c.id === e.target.value);
            if (item) setPrompt(item.text);
          }}
        >
          <option value="">Choose a sample prompt…</option>
          {[...groups].map(([cat, items]) => (
            <optgroup key={cat} label={cat}>
              {items.map((c) => (
                <option key={c.id} value={c.id}>{c.title}</option>
              ))}
            </optgroup>
          ))}
        </select>
      </label>
      <textarea
        value={prompt}
        placeholder="Type a prompt, or pick one from the corpus…"
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey) && canRun) mut.mutate();
        }}
        rows={6}
      />
      <div className="row">
        <button className="primary" disabled={!canRun} onClick={() => mut.mutate()} title="⌘/Ctrl + Enter">
          {mut.isPending ? "Running…" : "Run forward pass"}
        </button>
        <label className="inline" title="Prompts longer than this are truncated">
          max tokens <NumberBox value={maxTokens} min={1} max={1024} width={62} onCommit={(v) => setMaxTokens(Math.round(v))} />
        </label>
      </div>
      {!ready && <div className="muted small">Waiting for the model ({status?.state ?? "connecting"})…</div>}
      {mut.error && <div className="err">{(mut.error as Error).message}</div>}
      {run && (
        <div className="tokens-block">
          <div className="muted small">
            {run.tokens.length} tokens{run.truncated ? " (truncated)" : ""}
          </div>
          <div className="tokens" onMouseLeave={() => setHoverRow(null)}>
            {run.tokens.map((t, i) => (
              <span key={i} className={`tok ${hoverRow === i ? "hot" : ""}`} onMouseEnter={() => setHoverRow(i)} title={`#${i} · id ${run.token_ids[i]}`}>
                {fmtToken(t, 14)}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
