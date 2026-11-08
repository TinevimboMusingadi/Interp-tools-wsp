import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from wsp.harness.agent import run_agent  # noqa: E402
from wsp.harness.env import Env  # noqa: E402
from wsp.harness.ledger import Budget, BudgetExceeded, cost_usd  # noqa: E402
from wsp.harness.providers import FakeProvider, ToolCall, Turn  # noqa: E402
from wsp.harness.sandbox import Sandbox  # noqa: E402
from wsp.tools import TOOLSETS  # noqa: E402


@pytest.fixture
def env():
    from tiny import tiny_gpt2

    return Env(tiny_gpt2(), forward_budget=2, tools=TOOLSETS["c0"]())


def test_free_weight_read_and_edit(env):
    out = env.call("python", {"code": "w = model.hf_model.transformer.h[0].mlp.c_proj.weight\n"
                                      "print(tuple(w.shape))\nw.data[0, 0] += 1.0"})
    assert "(64, 32)" in out
    assert "1 of" in env.call("edit_report", {})


def test_forward_outside_tool_is_refused(env):
    out = env.call("python", {"code": "import torch\nmodel.hf_model(torch.tensor([[1, 2]]))"})
    assert "costs a forward pass" in out
    out = env.call("python", {"code": "import torch\nmodel.hf_model.transformer.h[0](torch.zeros(1,2,32))"})
    assert "costs a forward pass" in out
    assert env.forward_used == 0


def test_gradients_are_refused(env):
    out = env.call("python", {"code": "x = torch.ones(2, requires_grad=True)\n(x * 2).sum().backward()"})
    assert "gradients are not allowed" in out


def test_generate_spends_budget_then_refuses(env):
    for _ in range(2):
        assert "forward passes left" in env.call("generate", {"prompts": ["Hi", "2+2="], "max_new_tokens": 3})
    assert "budget spent" in env.call("generate", {"prompts": ["Hi"], "max_new_tokens": 3})


def test_budget_caps(tmp_path):
    assert cost_usd("fake", 10, 10) == 0.0
    with pytest.raises(KeyError):
        cost_usd("no-such-model", 1, 1)
    b = Budget("r", max_usd=1.0, max_tokens=5, ledger_path=tmp_path / "l.csv")
    b.charge("fake", "fake", 3, 3)
    with pytest.raises(BudgetExceeded):
        b.check()
    assert (tmp_path / "l.csv").read_text().count("\n") == 2


def test_agent_loop_end_to_end_in_sandbox(tmp_path, monkeypatch):
    import os

    here = str(Path(__file__).parent)
    monkeypatch.setenv("PYTHONPATH", here + os.pathsep + os.environ.get("PYTHONPATH", ""))
    script = [
        Turn("look", [ToolCall("1", "python", {"code": "print(model.hf_model.config.n_layer)"})], 10, 5),
        Turn("try", [ToolCall("2", "generate", {"prompts": ["Hello"], "max_new_tokens": 2})], 10, 5),
        Turn("edit", [ToolCall("3", "python", {"code": "model.hf_model.lm_head.weight.data[0] *= 0"})], 10, 5),
        Turn("done", [ToolCall("4", "submit", {"explanation": "zeroed token 0"})], 10, 5),
    ]
    with Sandbox(loader="tiny:tiny_gpt2", loader_kwargs={}, toolset="c0", forward_budget=3) as sb:
        res = run_agent(run_id="t", provider=FakeProvider(script), sandbox=sb, system="s",
                        task_prompt="task", budget=Budget("t", 1.0, 10_000, ledger_path=tmp_path / "l.csv"),
                        runs_dir=tmp_path)
    assert res.stop_reason == "submitted" and res.submitted == "zeroed token 0"
    assert res.forward_used == 1 and res.weights_changed > 0  # lm_head is tied to wte
    assert (tmp_path / "t" / "transcript.jsonl").exists()


def test_jail_blocks_files_processes_and_network_but_not_tools(tmp_path):
    """Regression for the first p1 run, where agent code read the repo and ran git via subprocess."""
    repo = Path(__file__).resolve().parents[1]
    probes = {
        "listdir_repo": f"import os; os.listdir({str(repo)!r})",
        "read_eval": f"open({str(repo / 'wsp' / 'eval' / 'suites.py')!r}).read()",
        "read_readme": f"open({str(repo / 'README.md')!r}).read()",
        "list_home": "import os; os.listdir(os.path.expanduser('~'))",  # os.walk swallows the refusal
        "subprocess": "import subprocess; subprocess.run(['git', 'log'])",
        "system": "import os; os.system('echo hi')",
        "socket": "import socket; socket.create_connection(('example.com', 80), timeout=2)",
        "chdir": "import os; os.chdir('..')",
    }
    with Sandbox(loader="tiny:tiny_gpt2", loader_kwargs={}, toolset="c0", forward_budget=2) as sb:
        for name, code in probes.items():
            out = sb.call("python", {"code": code})
            assert "refused by sandbox" in out, (name, out)
        # modules spawn imported from the parent's main are gone, and re-importing them is refused
        for code in ("import wsp.tasks.registry", "import wsp.eval.suites", "import wsp; print(wsp.tasks)",
                     "import sys; print(sys.modules['__main__'].__dict__.keys() - {'__name__', '__doc__', "
                     "'__package__', '__loader__', '__spec__'} or 'empty main')"):
            out = sb.call("python", {"code": code})
            assert any(m in out for m in ("refused by sandbox", "AttributeError", "empty main")), (code, out)
        ok = sb.call("python", {"code": "open('notes.txt', 'w').write('x'); import os; print(os.listdir('.'))"})
        assert "notes.txt" in ok and "refused" not in ok, ok
        gen = sb.call("generate", {"prompts": ["hello"], "max_new_tokens": 2})
        assert "refused" not in gen and "Traceback" not in gen, gen
        assert len(sb.status()["jail_violations"]) >= len(probes) + 2  # create_connection retries per address
