"""WSP Studio: transcript -> steps with network touches (no model needed)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from wsp.studio.arch import FALLBACK, arch_from_config
from wsp.studio.diff import sparse_diff
from wsp.studio.events import StepParser, code_touches, parse_run, tensor_part, tool_touches


def _parts(ts, mode=None):
    return {(t["part"], t["layer"]) for t in ts if mode is None or t["mode"] == mode}


def test_tensor_part():
    assert tensor_part("model.layers.20.mlp.down_proj.weight") == ("mlp", 20)
    assert tensor_part("model.layers.3.self_attn.o_proj.weight") == ("attn", 3)
    assert tensor_part("model.layers.4.linear_attn.out_proj.weight") == ("attn", 4)
    assert tensor_part("model.embed_tokens.weight") == ("embed", None)
    assert tensor_part("lm_head.weight") == ("head", None)
    assert tensor_part("model.norm.weight") == ("norm", None)


def test_code_touches_edits_and_reads():
    code = ("W = hf.model.layers[12].mlp.down_proj.weight\n"
            "hf.model.layers[17].mlp.down_proj.weight.data[:, 5] *= 0.5\n"
            "wsp.edit.scale_units(hf.model.embed_tokens.weight, [248069], 1.2, axis='out')\n"
            "for l in range(20, 24):\n    x = hf.model.layers[l].self_attn.o_proj.weight\n")
    ts = code_touches(code)
    assert ("mlp", 12) in _parts(ts, "read")
    assert ("mlp", 17) in _parts(ts, "edit")
    emb = next(t for t in ts if t["part"] == "embed")
    assert emb["mode"] == "edit" and emb["index"] == [248069]
    assert {("attn", L) for L in range(20, 24)} <= _parts(ts, "read")


def test_tool_touches_from_args_and_output():
    assert _parts(tool_touches("neuron_logits", {"layer": 10, "neuron": 7}, "")) == {("mlp", 10), ("head", None)}
    ts = tool_touches("writers_of", {"direction": "d"}, "neuron 20.5: cos 0.3\nneuron 20.9: cos 0.2\nneuron 3.1: cos 0.1")
    by = {t["layer"]: t["index"] for t in ts}
    assert by == {3: [1], 20: [5, 9]}
    st = tool_touches("steer_test", {"layer": 12, "direction": "d"}, "")
    assert ("resid", 12) in _parts(st, "steer")
    rep = tool_touches("edit_report", {}, "1,024 of 9 weights changed (1%)\n  model.layers.20.mlp.down_proj.weight: 1,024")
    assert rep == [{"part": "mlp", "layer": 20, "mode": "state", "n": 1024}]


def test_parser_tracks_budget_and_steps(tmp_path: Path):
    rows = [
        {"t": 0, "kind": "start", "provider": "gemini", "model": "m", "system": "s", "tools": ["python"],
         "task": "Make this model (Qwen3.5-0.8B) better.\n\nForward-pass budget: 10."},
        {"t": 1, "kind": "assistant", "turn": 1, "text": "look", "tool_calls": [{"name": "generate", "args": {}}],
         "input_tokens": 5, "output_tokens": 2},
        {"t": 2, "kind": "tool", "turn": 1, "name": "generate", "args": {}, "output": "(forward passes left: 9)\n[0] hi"},
        {"t": 3, "kind": "end", "stop_reason": "submitted", "weights_changed": 2},
    ]
    d = tmp_path / "exp" / "cell"
    d.mkdir(parents=True)
    (d / "transcript.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    (d / "edit_diff.json").write_text(json.dumps(
        {"model.embed_tokens.weight": {"shape": [4, 2], "n": 2, "rows": [3], "cols": [0, 1]}}), encoding="utf-8")
    r = parse_run(d, decode=lambda ids: [f"t{i}" for i in ids])
    kinds = [s["kind"] for s in r["steps"]]
    assert kinds == ["start", "think", "tool", "end"]
    assert r["target"] == "Qwen/Qwen3.5-0.8B" and r["budget"] == 10
    assert r["steps"][2]["forward_left"] == 9 and r["steps"][2]["tokens"] == 7
    end = r["steps"][-1]["touches"][0]
    assert end["part"] == "embed" and end["index"] == [3] and end["labels"] == ["t3"]


def test_parser_prefers_weight_delta():
    p = StepParser()
    p.feed({"t": 0, "kind": "start", "task": "x"})
    s = p.feed({"t": 1, "kind": "tool", "name": "python", "args": {"code": "hf.model.layers[1].mlp.x += 1"},
                "output": "", "weight_delta": {"model.layers.5.mlp.down_proj.weight": {"n": 3, "cols": [7]}}})[0]
    edits = [t for t in s["touches"] if t["mode"] == "edit"]
    assert edits == [{"part": "mlp", "layer": 5, "mode": "edit", "index": [7], "n": 3,
                      "tensor": "model.layers.5.mlp.down_proj.weight"}]


def test_sparse_diff_rows_and_cols():
    base = {"w": torch.zeros(4, 3), "b": torch.zeros(3)}
    new = {"w": torch.zeros(4, 3), "b": torch.zeros(3)}
    new["w"][2, 1] = 1
    new["b"][0] = 1
    d = sparse_diff(base, new)
    assert d["w"] == {"shape": [4, 3], "n": 1, "rows": [2], "cols": [1]}
    assert d["b"]["rows"] == [0]


def test_arch_reads_nested_config():
    assert FALLBACK["n_layers"] == 24 and FALLBACK["layer_types"][3] == "full"
    a = arch_from_config({"num_hidden_layers": 2, "hidden_size": 8, "intermediate_size": 16, "vocab_size": 10})
    assert a["layer_types"] == ["full", "full"] and a["d_mlp"] == 16


def test_code_touches_follows_aliases_and_call_site_layers():
    code = (
        "def apply_ortho(layers_list, strength=1.0):\n"
        "    for l in layers_list:\n"
        "        layer_mod = model.hf_model.model.layers[l]\n"
        "        attn_w = layer_mod.self_attn.o_proj.weight\n"
        "        wsp.edit.feature_edit(attn_w, u, mode='remove')\n"
        "        wsp.edit.feature_edit(layer_mod.mlp.down_proj.weight, u, mode='remove')\n"
        "apply_ortho(list(range(6, 9)), strength=1.0)\n"
    )
    edits = {(t["part"], t["layer"]) for t in code_touches(code) if t["mode"] == "edit"}
    assert edits == {(p, L) for p in ("attn", "mlp") for L in (6, 7, 8)}
