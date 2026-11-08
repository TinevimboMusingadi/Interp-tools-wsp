"""The agent loop: provider <-> sandbox, with budgets and a full jsonl transcript.

One run = one (task, condition, provider, seed). The loop ends when the agent calls ``submit``, stops
calling tools, hits ``max_turns``, or a budget cap trips. Everything sent and received is written to
``runs/<run_id>/transcript.jsonl`` so every number in a report can be traced to what the agent saw.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from wsp.harness.ledger import Budget, BudgetExceeded
from wsp.harness.providers import Provider, ToolResult
from wsp.harness.sandbox import Sandbox

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


@dataclass
class RunResult:
    run_id: str
    stop_reason: str
    turns: int
    submitted: str | None
    answer: dict | None
    forward_used: int
    weights_changed: int
    weights_total: int
    spent_usd: float
    tokens: int


class Transcript:
    def __init__(self, run_id: str, runs_dir: Path = RUNS_DIR):
        self.dir = runs_dir / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "transcript.jsonl"

    def log(self, kind: str, **data: object) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), "kind": kind, **data}, default=str) + "\n")


def run_agent(*, run_id: str, provider: Provider, sandbox: Sandbox, system: str, task_prompt: str,
              budget: Budget, max_turns: int = 200, runs_dir: Path = RUNS_DIR) -> RunResult:
    tr = Transcript(run_id, runs_dir)
    tr.log("start", provider=provider.name, model=provider.model, system=system, task=task_prompt,
           tools=[t.name for t in sandbox.tool_specs])
    provider.start(system, sandbox.tool_specs)
    message: str | list[ToolResult] = task_prompt
    stop, turns = "max_turns", 0
    for turns in range(1, max_turns + 1):
        try:
            budget.check()
        except BudgetExceeded as e:
            stop = f"budget: {e}"
            break
        turn = provider.send(message)
        budget.charge(provider.name, provider.model, turn.input_tokens, turn.output_tokens)
        tr.log("assistant", turn=turns, text=turn.text, tool_calls=[asdict(c) for c in turn.tool_calls],
               input_tokens=turn.input_tokens, output_tokens=turn.output_tokens)
        if not turn.tool_calls:
            stop = "no_tool_call"
            break
        results = []
        for call in turn.tool_calls:
            out = sandbox.call(call.name, call.args)
            tr.log("tool", turn=turns, name=call.name, args=call.args, output=out)
            results.append(ToolResult(call.id, call.name, out))
        message = results
        if any(c.name == "submit" for c in turn.tool_calls):
            stop = "submitted"
            break
    st = sandbox.status()
    res = RunResult(run_id, stop, turns, st.get("submitted"), st.get("answer"), st["forward_used"],
                    st["weights_changed"],
                    st["weights_total"], round(budget.spent_usd, 4), budget.tokens)
    tr.log("end", **asdict(res))
    (tr.dir / "result.json").write_text(json.dumps(asdict(res), indent=2), encoding="utf-8")
    return res
