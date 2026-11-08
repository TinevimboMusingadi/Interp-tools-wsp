"""End-to-end pipeline on the tiny model with a scripted provider: runner -> evaluation -> grade ->
report -> export, plus RSI library merging and the organism fine-tune loop."""

import json
import os
import sys
from pathlib import Path

import pytest
import torch
import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from wsp import export, report, rsi  # noqa: E402
from wsp import run as runner  # noqa: E402
from wsp.tasks import registry  # noqa: E402


def tiny_task(device: str = "cpu") -> registry.Task:
    return registry.Task(
        id="tiny", description="test", prompt="Zero the first unembedding row, then submit.",
        loader_kwargs={}, eval_suites=["tiny"],
        grade=lambda ans, expl, hidden: {"said_yes": int(ans.get("ok") is True)}, hidden={})


def tiny_eval(model, **_):
    return {"row0_norm": float(model.hf_model.lm_head.weight[0].norm()),
            "marker_survived": int(hasattr(model.hf_model, "wsp_marker"))}


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    monkeypatch.setitem(registry.TASKS, "tiny", tiny_task)
    script = [
        {"text": "look", "calls": [["edit_report", {}]]},
        {"text": "edit", "calls": [["python", {"code": "model.hf_model.lm_head.weight.data[0] = 0\n"
                                                     "model.hf_model.wsp_marker = 1"}]]},
        {"text": "done", "calls": [["submit", {"explanation": "zeroed row 0", "answer": {"ok": True}}]]},
    ]
    c = {"id": "exp", "task": "tiny", "conditions": ["c0", "c1"], "budgets": [2], "seeds": [0],
         "providers": [{"kind": "fake", "model": "fake", "short": "fake", "kwargs": {"script": script}}],
         "caps": {"max_usd": 1, "max_tokens": 10_000, "max_turns": 10}, "loader": "tiny:tiny_gpt2",
         "runs_dir": str(tmp_path / "runs"), "ledger": str(tmp_path / "ledger.csv"), "save_weights": False,
         "eval_fn": "test_pipeline:tiny_eval"}
    d = tmp_path / "exp"
    d.mkdir()
    (d / "config.yaml").write_text(yaml.safe_dump(c), encoding="utf-8")
    return d


def test_dry_run_and_estimate(cfg, capsys):
    assert runner.main([str(cfg / "config.yaml"), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "2 cells" in out and "c0_b2_fake_s0" in out and "c1_b2_fake_s0" in out


def test_variants_expand_cells():
    c = {"conditions": ["c0"], "budgets": [5], "seeds": [0, 1], "providers": [{"model": "m", "short": "m"}],
         "variants": [{"name": "p1", "pair": 1}, {"name": "p2", "pair": 2}]}
    ids = [x.id for x in runner.cells(c)]
    assert ids == ["p1_c0_b5_m_s0", "p1_c0_b5_m_s1", "p2_c0_b5_m_s0", "p2_c0_b5_m_s1"]


def test_full_cell_then_report_and_export(cfg, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(report, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(export, "RUNS_DIR", tmp_path / "runs")
    conf = runner.load_config(cfg / "config.yaml")
    conf["runs_dir"] = str(tmp_path / "runs")
    cell = runner.cells(conf)[0]
    (cfg / "results").mkdir()
    out = runner.run_cell(conf, cell, cfg / "results")
    assert out["agent"]["stop_reason"] == "submitted" and out["grade"] == {"said_yes": 1}
    assert out["agent"]["weights_changed"] > 0
    # evaluation sees the weight edit but not attributes, hooks or patches left on the live model
    assert out["eval"] == {"row0_norm": 0.0, "marker_survived": 0}
    assert not list((tmp_path / "runs").rglob("weights*.pt"))  # save_weights: False cleans up
    md = report.render(cfg)
    assert "| - | c0 | 2 | fake | 1 |" in md and "said_yes" in md
    assert "No file-system" in md
    rows = export.collect([cfg], "said_yes", 1)
    assert len(rows) == 1 and rows[0]["messages"][0]["role"] == "system"
    assert any("<tool_call>" in m["content"] for m in rows[0]["messages"] if m["role"] == "assistant")


def test_audit_flags_file_access(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "RUNS_DIR", tmp_path)
    (tmp_path / "e" / "c").mkdir(parents=True)
    (tmp_path / "e" / "c" / "transcript.jsonl").write_text(json.dumps(
        {"kind": "tool", "turn": 3, "name": "python", "args": {"code": "open('organisms/.keys.json')"}}) + "\n")
    flags = report.audit([{"cell": {"id": "c"}, "agent": {"run_id": "e/c"}}])
    assert flags and "turn 3" in flags[0]


def test_rsi_merge_library(tmp_path):
    a, b, dst = tmp_path / "a", tmp_path / "b", tmp_path / "L1"
    for d, body in ((a, "x = 1"), (b, "x = 2")):
        (d / "tools").mkdir(parents=True)
        (d / "tools" / "helper.py").write_text(body)
    (b / "tools" / "other.py").write_text("y = 1")
    added = rsi.merge_library([a, b, a], dst)
    assert sorted(added) == ["tools/helper.py", "tools/helper_1.py", "tools/other.py"]


def test_guarded_open_blocks_keys(tmp_path):
    from tiny import tiny_gpt2
    from wsp.harness.env import Env
    from wsp.tools import TOOLSETS

    e = Env(tiny_gpt2(), forward_budget=1, tools=TOOLSETS["c0"]())
    assert "not allowed" in e.call("python", {"code": "open('organisms/.keys.json')"})


def test_organism_finetune_freezes_embeddings():
    from tiny import tiny_gpt2
    from wsp.harness.env import restore_gradients
    from wsp.organisms import finetune

    restore_gradients()  # an earlier in-process Env blocked them; the builder runs in its own process
    m = tiny_gpt2()
    hf, tok = m.hf_model, m.tokenizer
    if not getattr(tok, "chat_template", None):
        pytest.skip("tokenizer has no chat template")
    emb = hf.get_input_embeddings().weight.detach().clone()
    state = finetune(hf, tok, [("Say hi", "hi")] * 4, steps=3, lr=1e-3, batch=2, max_len=48)
    assert torch.equal(hf.get_input_embeddings().weight, emb)
    assert "transformer.wte.weight" not in state and len(state) > 0
