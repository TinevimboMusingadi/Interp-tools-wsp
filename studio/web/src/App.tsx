import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, STATIC } from "./api";
import { applyStep, emptyHeat, heatUpTo } from "./heat";
import { caption, pct } from "./describe";
import { NetworkMap } from "./components/NetworkMap";
import { AgentChat, Detail, ToolTimeline } from "./components/Panels";
import { Player } from "./components/Player";
import { StudioPanel } from "./components/StudioPanel";
import { recordMap, recordScreen, type Recording } from "./recorder";
import type { Arch, Evals, RunData, RunInfo, Step, Touch } from "./types";

type Mode = "replay" | "live" | "studio";

// URL options (used by the blog embed): ?run=<exp>/<cell>&embed=1&play=1&speed=2&step=N
const Q = new URLSearchParams(window.location.search);
const EMBED = Q.get("embed") === "1";

function stepDelay(s?: Step): number {
  if (!s) return 1000;
  if (s.kind === "start") return 2600;
  if (s.kind === "end") return 5000;
  if (s.kind === "think") return Math.min(2600, 900 + (s.text?.length ?? 0) * 3);
  if ((s.touches ?? []).some((t) => t.mode === "edit")) return 2600;
  return 1500;
}

export default function App() {
  const [mode, setMode] = useState<Mode>("replay");
  const [runs, setRuns] = useState<RunInfo[]>([]);
  const [runId, setRunId] = useState<string>("");
  const [data, setData] = useState<RunData | null>(null);
  const [steps, setSteps] = useState<Step[]>([]);
  const [arch, setArch] = useState<Arch | null>(null);
  const [cur, setCur] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(Number(Q.get("speed")) || 2);
  const [follow, setFollow] = useState(true);
  const [rec, setRec] = useState<Recording | null>(null);
  const [err, setErr] = useState("");
  // studio
  const [studio, setStudio] = useState<Record<string, any> | null>(null);
  const [sSteps, setSSteps] = useState<Step[]>([]);
  const [sState, setSState] = useState<{ touches: Touch[]; n_changed: number; undo: string[] }>({ touches: [], n_changed: 0, undo: [] });
  const [busy, setBusy] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    api.runs().then((r) => {
      setRuns(r);
      const want = Q.get("run");
      const pick = r.find((x) => x.id === want) ?? r.find((x) => x.cell.includes("carbon") && x.weights_changed) ?? r[r.length - 1];
      if (pick) setRunId(pick.id);
    }).catch((e) => setErr(String(e)));
    api.arch().then(setArch).catch(() => {});
    api.studioStatus().then(setStudio).catch(() => setStudio({ enabled: false }));
  }, []);

  // replay: load the whole run
  useEffect(() => {
    if (mode !== "replay" || !runId) return;
    setPlaying(false);
    api.run(runId).then((d) => {
      setData(d);
      setSteps(d.steps);
      setArch(d.arch);
      const first = Q.get("step");
      setCur(first && runId === Q.get("run") ? Math.min(d.steps.length - 1, Number(first)) : 0);
      if (Q.get("play") === "1" && runId === (Q.get("run") ?? runId)) setPlaying(true);
    }).catch((e) => setErr(String(e)));
  }, [runId, mode]);

  // live: stream steps as the transcript grows
  useEffect(() => {
    esRef.current?.close();
    if (mode !== "live" || !runId) return;
    setSteps([]);
    setCur(0);
    setData(null);
    const es = new EventSource(`/api/live/${runId}`);
    esRef.current = es;
    es.addEventListener("arch", (e) => setArch(JSON.parse((e as MessageEvent).data)));
    es.addEventListener("step", (e) => {
      const s: Step = JSON.parse((e as MessageEvent).data);
      setSteps((prev) => [...prev, s]);
    });
    es.addEventListener("done", () => es.close());
    return () => es.close();
  }, [runId, mode]);
  useEffect(() => {
    if (mode === "live" && follow) setCur(Math.max(0, steps.length - 1));
  }, [steps.length, mode, follow]);

  // playback
  useEffect(() => {
    if (!playing || mode === "studio") return;
    if (cur >= steps.length - 1) {
      setPlaying(false);
      return;
    }
    const id = setTimeout(() => setCur((c) => Math.min(steps.length - 1, c + 1)), stepDelay(steps[cur + 1]) / speed);
    return () => clearTimeout(id);
  }, [playing, cur, steps, speed, mode]);

  const seek = useCallback((i: number) => {
    setCur(i);
    if (mode === "live") setFollow(i >= steps.length - 1);
  }, [mode, steps.length]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input,textarea,select")) return;
      if (e.code === "Space") { e.preventDefault(); setPlaying((p) => !p); }
      if (e.code === "ArrowRight") setCur((c) => Math.min(steps.length - 1, c + 1));
      if (e.code === "ArrowLeft") setCur((c) => Math.max(0, c - 1));
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [steps.length]);

  const a = arch;
  const viewSteps = mode === "studio" ? sSteps : steps;
  const viewCur = mode === "studio" ? sSteps.length - 1 : cur;
  const step = viewSteps[viewCur];

  const heat = useMemo(() => {
    if (!a) return emptyHeat();
    if (mode !== "studio") return heatUpTo(steps, cur, a);
    const h = emptyHeat();
    sSteps.forEach((s) => applyStep(h, { ...s, touches: (s.touches ?? []).filter((t) => t.mode !== "edit" && t.mode !== "state") }, a));
    applyStep(h, { i: -1, kind: "tool", t: 0, forward_left: null, budget: null, tokens: 0, touches: sState.touches }, a);
    return h;
  }, [a, steps, cur, mode, sSteps, sState]);

  const [cap, sub] = caption(step);
  const info = runs.find((r) => r.id === runId);
  const evals: Evals = data?.eval ?? info?.eval ?? null;
  const base: Evals = data?.baseline ?? info?.baseline ?? null;

  const onStudioResult = (s: Partial<Step>, state?: { touches: Touch[]; n_changed: number; undo: string[] } | null) => {
    setSSteps((prev) => [...prev, { i: prev.length, kind: "tool", t: 0, forward_left: null, budget: null, tokens: 0, ...s } as Step]);
    if (state) setSState(state);
    api.studioStatus().then(setStudio).catch(() => {});
  };

  const startRec = async (what: "screen" | "map") => {
    const name = `wsp_${(runId || "studio").replace(/\//g, "_")}_${what}.webm`;
    try {
      const r = what === "screen" ? await recordScreen(name, () => setRec(null)) : recordMap(name, () => setRec(null));
      setRec(r);
      if (mode !== "studio") { setCur(0); setPlaying(true); }
    } catch (e) {
      setErr(`recording: ${e}`);
    }
  };

  const groups = useMemo(() => {
    const g = new Map<string, RunInfo[]>();
    runs.forEach((r) => g.set(r.exp, [...(g.get(r.exp) ?? []), r]));
    return [...g.entries()];
  }, [runs]);

  return (
    <div className={`app${EMBED ? " embed" : ""}${STATIC ? " static" : ""}`}>
      <header>
        <div className="brand">WSP <span>Studio</span></div>
        <div className="modes">
          {(STATIC ? ["replay"] as Mode[] : ["replay", "live", "studio"] as Mode[]).map((m) => (
            <button key={m} className={m === mode ? "on" : ""} onClick={() => { setMode(m); setPlaying(false); }}>
              {m === "replay" ? "Replay" : m === "live" ? "Live" : "Studio"}
            </button>
          ))}
        </div>
        {mode !== "studio" && (
          <select className="runsel" value={runId} onChange={(e) => setRunId(e.target.value)}>
            {groups.map(([exp, rs]) => (
              <optgroup key={exp} label={exp}>
                {rs.map((r) => (
                  <option key={r.id} value={r.id}>
                    {r.cell} · {r.agent ?? "?"}{r.live ? " · running" : ""}{r.eval?.math500_acc !== undefined ? ` · MATH ${pct(r.eval.math500_acc)}` : ""}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        )}
        {mode !== "studio" && evals && (
          <div className="evals">
            {([["MATH", "math500_acc"], ["hit limit", "math500_hit_limit"], ["GSM8K", "gsm8k_acc"], ["IFEval", "ifeval_subset_acc"]] as const).map(([lab, k]) => (
              evals[k] !== undefined && (
                <span key={k} className="eval" title="held-out eval after the edit (baseline in grey)">
                  {lab} <b>{pct(evals[k])}</b>{base?.[k] !== undefined && <i> / {pct(base[k])}</i>}
                </span>
              )
            ))}
          </div>
        )}
        {mode === "studio" && studio?.enabled && (
          <div className="evals"><span className="eval">{studio.model} · {studio.device} · {sState.n_changed.toLocaleString()} weights edited</span></div>
        )}
        <span className="grow" />
        {EMBED && (
          <a className="full" href={`?run=${encodeURIComponent(runId)}&step=${cur}`} target="_blank" rel="noopener">Open full Studio ↗</a>
        )}
        {EMBED ? null : rec ? (
          <button className="rec on" onClick={() => rec.stop()}>■ Stop recording</button>
        ) : (
          <>
            <button className="rec" onClick={() => startRec("screen")} title="record the whole page (pick this tab)">● Record</button>
            <button className="rec ghost" onClick={() => startRec("map")} title="record just the network map">Map only</button>
          </>
        )}
      </header>
      <div className="errbar" onClick={() => setErr("")}>{err}</div>{/* always rendered: it is a grid row */}

      <main>
        <section className="left">
          <div className="ptitle">{mode === "studio" ? "Your tools" : "Tool calls & probes"}</div>
          {mode === "studio" ? (
            studio?.enabled && a ? (
              <StudioPanel arch={a} ws={studio.ws ?? {}} busy={busy} setBusy={setBusy} onResult={onStudioResult} undo={sState.undo} />
            ) : (
              <div className="panel-body empty">
                The studio needs the model loaded. Restart the server with:
                <pre>python -m wsp.studio --runs runs --model Qwen/Qwen3.5-0.8B --device cpu</pre>
                Loading on CPU takes a while; replay and live work without it.
              </div>
            )
          ) : (
            <ToolTimeline steps={steps} cur={cur} onSeek={seek} />
          )}
        </section>

        <section className="centre">
          {a ? (
            <NetworkMap arch={a} heat={heat} flash={step?.touches ?? []} flashKey={`${mode}:${viewCur}`}
              caption={cap || (mode === "studio" ? "Studio: use the tools on the left" : "")} sub={sub} />
          ) : <div className="netmap empty">loading…</div>}
          <Detail step={step} />
        </section>

        <section className="right">
          <div className="ptitle">{mode === "studio" ? "Your session" : "Agent"}</div>
          {mode === "studio" ? (
            <div className="panel-body chat">
              {sSteps.length === 0 && <div className="empty">Results of your actions appear here and light up the network.</div>}
              {sSteps.map((s) => (
                <div key={s.i} className={`bubble ${s.error ? "warn" : "agent"} ${s.i === sSteps.length - 1 ? "current" : ""}`}>
                  <div className="who">{s.tool}{s.seconds !== undefined ? ` · ${s.seconds}s` : ""}</div>
                  <div className="txt mono">{(s.output ?? "").slice(0, 1500)}</div>
                </div>
              ))}
            </div>
          ) : (
            <AgentChat steps={steps} cur={cur} onSeek={seek} />
          )}
        </section>
      </main>

      {mode !== "studio" && (
        <Player steps={steps} cur={cur} playing={playing} speed={speed} live={mode === "live"} onSeek={seek}
          onPlay={() => { if (cur >= steps.length - 1) setCur(0); setPlaying((p) => !p); }} onSpeed={setSpeed} />
      )}
    </div>
  );
}
