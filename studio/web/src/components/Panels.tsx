// Left: tool calls as cards. Right: the agent's chat. Bottom of centre: the current step in detail.
import { useEffect, useRef, useState } from "react";
import { COLORS } from "../heat";
import { badge, where } from "../describe";
import type { AttnGrid, LensTable, Step } from "../types";

function useFollow(dep: unknown) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.querySelector(".current")?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [dep]);
  return ref;
}

export function ToolTimeline({ steps, cur, onSeek }: { steps: Step[]; cur: number; onSeek: (i: number) => void }) {
  const shown = steps.filter((s) => s.kind === "tool" && s.i <= cur);
  const ref = useFollow(cur);
  const last = shown.length ? shown[shown.length - 1].i : -1;
  return (
    <div className="panel-body" ref={ref}>
      {shown.length === 0 && <div className="empty">Tool calls appear here as the agent makes them.</div>}
      {shown.map((s) => {
        const [b, col] = badge(s.tool);
        const edits = (s.touches ?? []).filter((t) => t.mode === "edit");
        const pass = /\(forward passes left/.test(s.output ?? "");
        return (
          <div key={s.i} className={`card ${s.i === last ? "current" : ""} ${s.error ? "err" : ""}`} onClick={() => onSeek(s.i)}>
            <div className="card-head">
              <span className="badge" style={{ background: col }}>{b}</span>
              <span className="tname">{s.tool}</span>
              {pass && <span className="pill run">1 pass</span>}
              {edits.length > 0 && <span className="pill edit">edit</span>}
              {s.error && <span className="pill err">error</span>}
              <span className="turn">#{s.turn}</span>
            </div>
            <div className="card-args">{argLine(s)}</div>
            {(s.touches ?? []).filter((t) => t.part !== "model").slice(0, 3).map((t, j) => (
              <div key={j} className="touch" style={{ color: COLORS[t.mode] }}>
                {t.mode} · {where(t)}{t.guess ? " (guess)" : ""}
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}

function argLine(s: Step): string {
  const a = (s.args ?? {}) as Record<string, any>;
  if (s.tool === "python") return (a.code ?? "").split("\n").find((l: string) => l.trim() && !l.trim().startsWith("#"))?.slice(0, 90) ?? "";
  if (s.tool === "generate") return `${a.prompts?.length ?? 0} prompt(s), ${a.max_new_tokens ?? 256} tokens`;
  return Object.entries(a).filter(([k]) => k !== "prompts").map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(" ").slice(0, 110);
}

export function AgentChat({ steps, cur, onSeek }: { steps: Step[]; cur: number; onSeek: (i: number) => void }) {
  const shown = steps.filter((s) => (s.kind === "think" || s.kind === "start" || s.kind === "end" || s.kind === "violations") && s.i <= cur);
  const ref = useFollow(cur);
  const last = shown.length ? shown[shown.length - 1].i : -1;
  return (
    <div className="panel-body chat" ref={ref}>
      {shown.map((s) => {
        if (s.kind === "start")
          return (
            <div key={s.i} className={`bubble task ${s.i === last ? "current" : ""}`} onClick={() => onSeek(s.i)}>
              <div className="who">Task → {s.agent}</div>
              <div className="txt">{s.text}</div>
            </div>
          );
        if (s.kind === "end")
          return (
            <div key={s.i} className={`bubble end ${s.i === last ? "current" : ""}`} onClick={() => onSeek(s.i)}>
              <div className="who">Run ended · {String(s.result?.stop_reason ?? "")}</div>
              <div className="txt">
                {String(s.result?.submitted ?? "Did not submit; the weights as left were evaluated.")}
              </div>
            </div>
          );
        if (s.kind === "violations")
          return (
            <div key={s.i} className="bubble warn" onClick={() => onSeek(s.i)}>
              <div className="who">Sandbox blocked {s.items?.length} access attempt(s)</div>
              <div className="txt mono">{(s.items ?? []).slice(0, 6).map((x) => JSON.stringify(x)).join("\n")}</div>
            </div>
          );
        return (
          <div key={s.i} className={`bubble agent ${s.i === last ? "current" : ""}`} onClick={() => onSeek(s.i)}>
            <div className="who">Agent · turn {s.turn}</div>
            {s.thinking ? <div className="txt thinking" title="the agent's reasoning (summary from the API)">{s.thinking}</div> : null}
            {s.text?.trim() ? <div className="txt">{s.text}</div> : s.thinking ? null : intent(s) ? (
              <div className="txt intent" title="from the comments in the agent's code">{intent(s)}</div>
            ) : <div className="txt dim">(no visible reasoning)</div>}
            <div className="calls">
              {(s.calls ?? []).map((c, j) => {
                const [b, col] = badge(c.name);
                return (
                  <span key={j} className="callchip" style={{ borderColor: col }}>
                    <span className="badge sm" style={{ background: col }}>{b}</span> {c.name}
                  </span>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Gemini returns no visible reasoning text; its code comments say what it is trying.
function intent(s: Step): string {
  const out: string[] = [];
  for (const c of s.calls ?? []) {
    const a = (c.args ?? {}) as Record<string, any>;
    if (c.name === "python") {
      const cm = String(a.code ?? "").split("\n").map((l) => l.trim()).filter((l) => l.startsWith("#")).map((l) => l.replace(/^#+\s?/, ""));
      out.push(...cm.slice(0, 4));
    } else if (c.name === "submit") out.push(String(a.explanation ?? "").slice(0, 400));
    else if (c.name === "generate") out.push(`Runs ${a.prompts?.length ?? "?"} prompt(s) through the model.`);
  }
  return out.filter(Boolean).join("\n");
}

export function Detail({ step }: { step?: Step }) {
  const [tab, setTab] = useState<"out" | "args">("out");
  if (!step) return <div className="detail empty">Pick a run and press play.</div>;
  const a = step.args ?? {};
  return (
    <div className="detail">
      <div className="detail-head">
        <span className="dtitle">{step.kind === "tool" ? step.tool : step.kind}</span>
        {step.kind === "tool" && (
          <>
            <button className={tab === "out" ? "on" : ""} onClick={() => setTab("out")}>output</button>
            <button className={tab === "args" ? "on" : ""} onClick={() => setTab("args")}>{step.tool === "python" ? "code" : "args"}</button>
          </>
        )}
        <span className="grow" />
        {step.seconds !== undefined && <span className="dim">{step.seconds}s</span>}
      </div>
      {step.lens && <Lens lens={step.lens} />}
      {step.attention && <Attention grid={step.attention} />}
      <pre className="out">
        {step.kind === "tool"
          ? tab === "out"
            ? step.output
            : step.tool === "python"
              ? String((a as any).code ?? "")
              : JSON.stringify(a, null, 2)
          : step.text ?? JSON.stringify(step.result ?? step.items ?? {}, null, 2)}
      </pre>
    </div>
  );
}

export function Lens({ lens }: { lens: LensTable }) {
  return (
    <div className="lens">
      {lens.layers.map((L, r) => (
        <div key={L} className="lens-row">
          <span className="lens-l">L{L}</span>
          {lens.rows[r].map((c, j) => (
            <span key={j} className="lens-c" style={{ background: `rgba(2,132,199,${0.06 + c.p * 0.55})` }} title={`${(c.p * 100).toFixed(1)}%`}>
              {JSON.stringify(c.tok).slice(1, -1)}
            </span>
          ))}
        </div>
      ))}
    </div>
  );
}

export function Attention({ grid }: { grid: AttnGrid }) {
  const [h, setH] = useState(0);
  const m = grid.heads[h] ?? [];
  const n = m.length;
  const cell = Math.max(10, Math.min(26, 360 / Math.max(1, n)));
  return (
    <div className="attn">
      <div className="attn-heads">
        layer {grid.layer} head:
        {grid.heads.map((_, i) => (
          <button key={i} className={i === h ? "on" : ""} onClick={() => setH(i)}>{i}</button>
        ))}
      </div>
      <div className="attn-grid" style={{ gridTemplateColumns: `90px repeat(${n}, ${cell}px)` }}>
        {m.map((row, q) => (
          <div key={q} style={{ display: "contents" }}>
            <span className="attn-tok">{JSON.stringify(grid.tokens[q] ?? "").slice(1, -1)}</span>
            {row.map((p, k) => (
              <span key={k} style={{ width: cell, height: cell, background: `rgba(124,58,237,${p})` }} title={`${grid.tokens[q]} → ${grid.tokens[k]}: ${p}`} />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}
