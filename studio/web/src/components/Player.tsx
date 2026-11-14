import { COLORS } from "../heat";
import { fmtTime } from "../describe";
import type { Step } from "../types";

type Props = {
  steps: Step[];
  cur: number;
  playing: boolean;
  speed: number;
  live: boolean;
  onSeek: (i: number) => void;
  onPlay: () => void;
  onSpeed: (s: number) => void;
};

export function Player({ steps, cur, playing, speed, live, onSeek, onPlay, onSpeed }: Props) {
  const n = steps.length;
  const s = steps[cur];
  const marks = steps
    .map((x) => ({ i: x.i, edit: (x.touches ?? []).some((t) => t.mode === "edit"), err: x.error, pass: /\(forward passes left/.test(x.output ?? "") }))
    .filter((m) => m.edit || m.err || m.pass);
  const left = s?.forward_left;
  const budget = s?.budget;
  return (
    <div className="player">
      <button className="pbtn" onClick={() => onSeek(0)} title="start">⏮</button>
      <button className="pbtn" onClick={() => onSeek(Math.max(0, cur - 1))} title="back one step">◀</button>
      <button className="pbtn play" onClick={onPlay} title="play / pause (space)">{playing ? "❚❚" : "▶"}</button>
      <button className="pbtn" onClick={() => onSeek(Math.min(n - 1, cur + 1))} title="forward one step">▶▏</button>
      <select value={speed} onChange={(e) => onSpeed(Number(e.target.value))}>
        {[0.5, 1, 2, 4, 8, 16].map((v) => <option key={v} value={v}>{v}×</option>)}
      </select>
      <div className="scrub">
        <input type="range" min={0} max={Math.max(0, n - 1)} value={cur} onChange={(e) => onSeek(Number(e.target.value))} />
        <div className="marks">
          {marks.map((m) => (
            <span key={m.i} className="mark" title={m.edit ? "edit" : m.err ? "error" : "forward pass"}
              style={{ left: `${(m.i / Math.max(1, n - 1)) * 100}%`, background: m.edit ? COLORS.edit : m.err ? COLORS.error : COLORS.run, height: m.edit ? 10 : 6 }} />
          ))}
        </div>
      </div>
      <div className="pinfo">
        <div>step {n ? cur + 1 : 0}/{n}{live ? " · LIVE" : ""}</div>
        <div className="dim">{s ? `${fmtTime(s.t)} in · ${(s.tokens / 1e6).toFixed(2)}M tokens` : ""}</div>
      </div>
      {budget ? (
        <div className="budget" title="forward passes left">
          <div className="dim">passes {left ?? budget}/{budget}</div>
          <div className="bar"><div style={{ width: `${((left ?? budget) / budget) * 100}%` }} /></div>
        </div>
      ) : null}
    </div>
  );
}
