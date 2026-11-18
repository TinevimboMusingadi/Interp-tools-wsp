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
    jail_violations: int = 0
    valid: bool = True  # False when the provider, not the agent, ended the run: rerun, never report


EMPTY_RETRIES = 2
REPROMPT = ("[harness] Your last reply had no tool call. Continue the task by calling a tool, or call "
            "`submit` if you are done.")
WARN_AT = (0.5, 0.75, 0.9)


def _budget_line(budget: Budget, sandbox: Sandbox, turn: int, max_turns: int, warned: set) -> str:
    """Appended to every tool result, so the agent can pace itself instead of running into a cap it
    never saw (p2: 7 of 8 runs hit the 4M-token cap mid-investigation; one submitted)."""
    used = max(budget.tokens / budget.max_tokens, budget.spent_usd / budget.max_usd, turn / max_turns)
    st = sandbox.passes() if hasattr(sandbox, "passes") else {}
    passes = st["forward_budget"] - st["forward_used"] if "forward_budget" in st else None
    line = (f"\n\n[budget] tokens {budget.tokens / 1e6:.2f}M of {budget.max_tokens / 1e6:.1f}M · turn {turn} of "
            f"{max_turns}" + (f" · forward passes left {passes}" if passes is not None else ""))
    for w in WARN_AT:
        if used >= w and w not in warned:
            warned.add(w)
            line += (f"\n[harness] {used:.0%} of this run's budget is used. The run stops at 100%; whatever "
                     f"weights the model has then are what gets evaluated. Call `submit` with your explanation "
                     f"when your edit is in place.")
            break
    return line


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
    stop, turns, warned, empty = "max_turns", 0, set(), 0
    for turns in range(1, max_turns + 1):
        try:
            budget.check()
        except BudgetExceeded as e:
            stop = f"budget: {e}"
            break
        turn = provider.send(message)
        budget.charge(provider.name, provider.model, turn.input_tokens, turn.output_tokens,
                      turn.cache_read_tokens, turn.cache_write_tokens)
        tr.log("assistant", turn=turns, text=turn.text, tool_calls=[asdict(c) for c in turn.tool_calls],
               input_tokens=turn.input_tokens, output_tokens=turn.output_tokens, finish=turn.finish,
               thinking=turn.thinking, cache_read_tokens=turn.cache_read_tokens,
               cache_write_tokens=turn.cache_write_tokens)
        if not turn.tool_calls:
            # A reply with no tool call is usually a provider hiccup (Gemini's MALFORMED_FUNCTION_CALL or
            # an empty candidate ended p2's c1_b10_carbon at turn 8), not the agent deciding to stop:
            # nudge it twice before giving up, and mark a run the provider ended as invalid.
            empty += 1
            if empty > EMPTY_RETRIES:
                stop = "no_tool_call" if turn.text.strip() else "provider_empty"
                break
            message = REPROMPT
            continue
        empty = 0
        results = []
        for call in turn.tool_calls:
            out = sandbox.call(call.name, call.args)
            extra = {}
            if call.name == "python" and hasattr(sandbox, "last_delta"):  # exact edit, for replay
                delta = sandbox.last_delta()
                if delta:
                    extra["weight_delta"] = delta
            tr.log("tool", turn=turns, name=call.name, args=call.args, output=out, **extra)
            results.append(ToolResult(call.id, call.name, out))
        note = _budget_line(budget, sandbox, turns, max_turns, warned)
        results[-1] = ToolResult(results[-1].call_id, results[-1].name, results[-1].content + note)
        message = results
        if any(c.name == "submit" for c in turn.tool_calls):
            stop = "submitted"
            break
    st = sandbox.status()
    res = RunResult(run_id, stop, turns, st.get("submitted"), st.get("answer"), st["forward_used"],
                    st["weights_changed"],
                    st["weights_total"], round(budget.spent_usd, 4), budget.tokens,
                    len(st.get("jail_violations") or []), valid=stop != "provider_empty")
    if st.get("jail_violations"):
        tr.log("jail_violations", items=st["jail_violations"])
    tr.log("end", **asdict(res))
    (tr.dir / "result.json").write_text(json.dumps(asdict(res), indent=2), encoding="utf-8")
    return res
