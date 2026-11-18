"""Phase A harness rigour: empty replies, the visible budget, check_practice, seeds, clean reruns,
cache-aware costs and the paired bootstrap."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from wsp.harness.agent import REPROMPT, run_agent  # noqa: E402
from wsp.harness.env import Env  # noqa: E402
from wsp.harness.ledger import FIELDS, Budget, cost_usd  # noqa: E402
from wsp.harness.providers import FakeProvider, ToolCall, Turn  # noqa: E402
from wsp.tools import TOOLSETS  # noqa: E402


class LocalSandbox:
    """The Sandbox interface over an in-process Env (no subprocess), for fast loop tests."""

    def __init__(self, env: Env):
        self.env = env
        self.tool_specs = [t.spec for t in env.tools.values()]

    def call(self, name, args):
        return self.env.call(name, args)

    def passes(self):
        return {"forward_used": self.env.forward_used, "forward_budget": self.env.forward_budget}

    def status(self):
        st = self.env.status()
        st.update(submitted=self.env.namespace.get("__submitted__"), answer=self.env.namespace.get("__answer__"))
        return st


@pytest.fixture
def sandbox():
    from tiny import tiny_gpt2

    env = Env(tiny_gpt2(), forward_budget=3, tools=TOOLSETS["c0"]())
    return LocalSandbox(env)


def _run(sandbox, script, tmp_path, **kw):
    budget = kw.pop("budget", None) or Budget("t", 1.0, 100_000, ledger_path=tmp_path / "l.csv")
    prov = FakeProvider(script)
    res = run_agent(run_id="t", provider=prov, sandbox=sandbox, system="s", task_prompt="task", budget=budget,
                    runs_dir=tmp_path, **kw)
    rows = [json.loads(x) for x in (tmp_path / "t" / "transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    return res, prov, rows


def test_empty_reply_is_reprompted_and_the_run_recovers(sandbox, tmp_path):
    script = [
        Turn("", [], 5, 5, finish="MALFORMED_FUNCTION_CALL"),
        Turn("", [ToolCall("1", "submit", {"explanation": "ok"})], 5, 5),
    ]
    res, prov, _ = _run(sandbox, script, tmp_path)
    assert res.stop_reason == "submitted" and res.valid
    assert prov.received[1] == REPROMPT


def test_repeated_empty_replies_make_the_run_invalid(sandbox, tmp_path):
    script = [Turn("", [], 5, 5, finish="EMPTY") for _ in range(3)]
    res, _, rows = _run(sandbox, script, tmp_path)
    assert res.stop_reason == "provider_empty" and not res.valid
    assert [r["finish"] for r in rows if r["kind"] == "assistant"] == ["EMPTY"] * 3


def test_agent_that_talks_without_tools_ends_validly(sandbox, tmp_path):
    script = [Turn("I am finished.", [], 5, 5, finish="end_turn") for _ in range(3)]
    res, _, _ = _run(sandbox, script, tmp_path)
    assert res.stop_reason == "no_tool_call" and res.valid


def test_every_tool_result_carries_the_budget_and_warnings_fire_once(sandbox, tmp_path):
    script = [Turn("", [ToolCall(str(i), "python", {"code": "print(1)"})], 300, 100) for i in range(4)]
    budget = Budget("t", 100.0, 2000, ledger_path=tmp_path / "l.csv")
    _, prov, _ = _run(sandbox, script, tmp_path, budget=budget)
    sent = [m[-1].content for m in prov.received[1:] if isinstance(m, list)]
    assert all("[budget] tokens" in s and "forward passes left 3" in s for s in sent)
    warnings = " ".join(sent).count("[harness]")
    assert warnings == 2  # 400 tokens = 20% (none), 800 = 40% (none), 1200 = 60%, 1600 = 80%


def test_check_practice_charges_passes_and_grades(tmp_path):
    from tiny import tiny_gpt2

    env = Env(tiny_gpt2(), forward_budget=2, tools=TOOLSETS["c0"]())
    env.namespace["gen_cap"] = 4
    env.namespace["practice"] = [{"source": "gsm8k", "problem": f"What is {i}+1?", "answer": str(i + 1)}
                                 for i in range(20)]
    out = env.call("check_practice", {"n": 8})
    assert "check_practice on 8 problems" in out and "accuracy" in out
    assert env.forward_used == 1
    assert "not available" in Env(tiny_gpt2(), forward_budget=1, tools=TOOLSETS["c0"]()).call("check_practice", {})


def test_cache_tokens_are_priced_and_counted(tmp_path):
    prices = {"m": {"input_per_mtok": 4.0, "output_per_mtok": 20.0, "cache_read_per_mtok": 0.2,
                    "cache_write_per_mtok": 5.0}}
    assert cost_usd("m", 1_000_000, 0, prices, cache_read=1_000_000, cache_write=1_000_000) == pytest.approx(9.2)
    b = Budget("r", 10.0, 10**9, ledger_path=tmp_path / "l.csv", prices=prices)
    b.charge("p", "m", 100, 10, cache_read=1000, cache_write=50)
    assert b.tokens == 1160
    header = (tmp_path / "l.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header == FIELDS


def test_old_ledger_gets_the_cache_columns(tmp_path):
    led = tmp_path / "l.csv"
    led.write_text("time,run_id,provider,model,input_tokens,output_tokens,cost_usd\nx,r,p,fake,1,1,0.0\n",
                   encoding="utf-8")
    Budget("r", 1.0, 100, ledger_path=led).charge("p", "fake", 1, 1)
    lines = led.read_text(encoding="utf-8").splitlines()
    assert lines[0].split(",") == FIELDS and all(len(x.split(",")) == len(FIELDS) for x in lines)


def test_rerun_archives_the_previous_attempt(tmp_path):
    from wsp.run import _archive_attempt

    d = tmp_path / "cell"
    d.mkdir()
    (d / "transcript.jsonl").write_text("old\n", encoding="utf-8")
    _archive_attempt(d)
    assert not (d / "transcript.jsonl").exists() and (d / "attempt_1" / "transcript.jsonl").exists()
    (d / "transcript.jsonl").write_text("second\n", encoding="utf-8")
    _archive_attempt(d)
    assert (d / "attempt_2" / "transcript.jsonl").read_text(encoding="utf-8") == "second\n"


def test_seed_shuffles_practice():
    from wsp.run import SEED_PRACTICE

    def order(seed):
        ns = {"practice": list(range(50))}
        exec(SEED_PRACTICE.format(seed=seed), ns)
        return ns["practice"]

    assert order(0) == order(0) and order(0) != order(1)
    exec(SEED_PRACTICE.format(seed=1), {})  # no practice in the task: a no-op, not an error


def test_paired_bootstrap_and_verdict():
    from wsp.report import comparisons, paired_bootstrap, verdict

    lo = paired_bootstrap([1.0] * 50, [0.0] * 50)
    assert lo == (1.0, 1.0, 1.0)

    def cell(cond, prov, items, seed=0):
        return {"cell": {"id": f"{cond}_{prov}_{seed}", "condition": cond, "budget": 10, "seed": seed,
                         "provider": {"model": prov, "short": prov}},
                "agent": {"spent_usd": 0, "forward_used": 1, "weights_changed": 0, "weights_total": 1},
                "eval": {"math500_items": [[x, 0] for x in items]}}

    n = 200
    good = [1] * 120 + [0] * 80
    base = [1] * 80 + [0] * 120
    results = [{"cell": {"id": "baseline", "condition": "baseline"}, "eval": {"math500_items": [[x, 0] for x in base]}}]
    for s in range(2):
        results += [cell("c1", "a", good, s), cell("c0", "a", base, s), cell("c1", "b", base, s), cell("c0", "b", base, s)]
    comps = comparisons(results)
    c1a = next(c for c in comps if c["condition"] == "c1" and c["provider"] == "a")
    assert c1a["seeds"] == 2 and c1a["vs_control"][0] == pytest.approx(40 / n) and c1a["vs_control"][1] > 0
    assert verdict(comps, {"budget": 10, "min_delta": 0.05, "min_agents": 1}).startswith("MET")
    assert verdict(comps, {"budget": 10, "min_delta": 0.05, "min_agents": 2}).startswith("NOT MET")
