"""Phase 4.3: turn successful agent trajectories into supervised fine-tuning data.

    python -m wsp.export experiments/p2_loop_fix experiments/p3_audit --metric math500_acc --min 0.2 \
        --out data/wsp_sft.jsonl

A trajectory is kept if its cell's metric (from eval or grade) clears the threshold. Each line is a
chat in the common ``messages`` format with tool calls and tool results inlined as text, so any SFT
stack can consume it. This is the "training signal" half of interp-in-the-loop self-improvement: a
small open model learns to be a better weight programmer from the frontier agent's successful runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wsp.harness.agent import RUNS_DIR


def trajectory_messages(transcript: Path) -> list[dict]:
    msgs: list[dict] = []
    for line in transcript.read_text(encoding="utf-8").splitlines():
        ev = json.loads(line)
        if ev["kind"] == "start":
            msgs += [{"role": "system", "content": ev["system"]}, {"role": "user", "content": ev["task"]}]
        elif ev["kind"] == "assistant":
            calls = "".join(f"\n<tool_call>{json.dumps({'name': c['name'], 'args': c['args']})}</tool_call>"
                            for c in ev["tool_calls"])
            msgs.append({"role": "assistant", "content": (ev["text"] or "") + calls})
        elif ev["kind"] == "tool":
            msgs.append({"role": "tool", "content": f"[{ev['name']}] {ev['output']}"})
    return msgs


def collect(exp_dirs: list[Path], metric: str, threshold: float) -> list[dict]:
    rows = []
    for d in exp_dirs:
        for f in sorted((d / "results").rglob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            if "agent" not in r:
                continue
            score = {**r.get("eval", {}), **r.get("grade", {})}.get(metric)
            tr = RUNS_DIR / r["agent"]["run_id"] / "transcript.jsonl"
            if score is None or score < threshold or not tr.exists():
                continue
            rows.append({"messages": trajectory_messages(tr), "meta": {"cell": r["cell"]["id"], "task": r["task"],
                                                                        metric: score}})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--metric", required=True)
    ap.add_argument("--min", type=float, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = collect([Path(e) for e in a.experiments], a.metric, a.min)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} trajectories to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
