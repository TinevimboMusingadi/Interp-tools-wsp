"""Turn an agent transcript into a timeline of steps, each with the parts of the network it touched.

A *touch* is ``{part, layer, index, mode, n, labels}``:

- ``part``: ``embed`` | ``attn`` | ``mlp`` | ``norm`` | ``head`` | ``resid`` | ``model``
- ``layer``: layer number, or ``None`` for parts outside the layer stack (embed, head, final norm) or
  "every layer"
- ``index``: rows / neurons / heads when known (embedding rows = token ids, MLP = neuron ids)
- ``mode``: ``read`` | ``edit`` | ``steer`` | ``run`` (a forward pass) | ``state`` (the current edit
  state, from an ``edit_report``)

Where a touch comes from, most exact first:

1. ``weight_delta`` on a tool row (logged by the harness after each python call), or the run's final
   ``edit_diff.json`` / ``edit_diff.pt`` (exact rows and columns).
2. ``edit_report`` output lines (``model.layers.20.mlp.down_proj.weight: 1,024``): exact tensors.
3. Interp tool arguments (``neuron_logits(layer=10, neuron=5)``): exact.
4. A line-by-line read of agent Python (``layers[12].mlp.down_proj ... +=``): a best guess, marked
   ``guess: True``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

OUTPUT_CHARS = 8000

EDIT_OP = re.compile(
    r"(?<![=!<>])(\+=|-=|\*=|/=)|\.(copy_|mul_|add_|sub_|div_|zero_|fill_|clamp_|index_copy_|index_add_)\("
    r"|\.data(\[[^\]]*\])?\s*=(?!=)|\.weight(\[[^\]]*\])\s*=(?!=)"
    r"|\b(scale_units|rank1|orthogonalize_out|orthogonalize_in|feature_edit)\("
)
LAYER_LIT = re.compile(r"layers\[(\d+)\]|layers\.(\d+)\.")
LAYER_VAR = re.compile(r"layers\[([A-Za-z_]\w*)\]")
RUN_TOOL = re.compile(r"run_tool\(\s*[\"'](\w+)[\"']")
REPORT_LINE = re.compile(r"^\s*((?:model\.)?[\w.]+\.(?:weight|bias))\s*:\s*([\d,]+)\s*$", re.M)
REPORT_HEAD = re.compile(r"([\d,]+) of ([\d,]+) weights changed")
FORWARD_LEFT = re.compile(r"\(forward passes left: (\d+)\)")
NEURON_OUT = re.compile(r"neuron (\d+)\.(\d+)")
BUDGET = re.compile(r"Forward-pass budget:\s*(\d+)")
TARGET = re.compile(r"Make this model \(([^)]+)\)|this model \(([^)]+)\)")


def tensor_part(name: str) -> tuple[str, int | None]:
    """``model.layers.20.mlp.down_proj.weight`` -> ("mlp", 20)."""
    m = re.search(r"layers\.(\d+)\.", name)
    layer = int(m.group(1)) if m else None
    if "embed_tokens" in name:
        return "embed", None
    if "lm_head" in name:
        return "head", None
    if layer is None:
        return ("norm", None) if "norm" in name else ("model", None)
    if ".mlp." in name:
        return "mlp", layer
    if "attn" in name:
        return "attn", layer
    return "norm", layer


def _loop_range(code: str, var: str) -> list[int] | None:
    m = re.search(rf"for\s+{re.escape(var)}\s+in\s+range\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\)", code)
    if m:
        a, b = (0, int(m.group(1))) if m.group(2) is None else (int(m.group(1)), int(m.group(2)))
        return list(range(a, b))
    m = re.search(rf"for\s+{re.escape(var)}\s+in\s+\[([\d,\s]+)\]", code)
    if m:
        return [int(x) for x in m.group(1).split(",") if x.strip()]
    return None


def _ints(s: str) -> list[int]:
    return [int(x) for x in re.findall(r"\d+", s)]


def touch(part: str, layer: int | None, mode: str, index: Iterable[int] | None = None, n: int | None = None,
          guess: bool = False) -> dict:
    t: dict[str, Any] = {"part": part, "layer": layer, "mode": mode}
    if index is not None:
        t["index"] = sorted(set(int(i) for i in index))[:4096]
    if n is not None:
        t["n"] = n
    if guess:
        t["guess"] = True
    return t


def code_touches(code: str) -> list[dict]:
    """Best-guess touches from agent Python, line by line (an edit op on a line edits what it names)."""
    out: list[dict] = []
    for line in code.splitlines():
        s = line.split("#", 1)[0]
        if not s.strip():
            continue
        mode = "edit" if EDIT_OP.search(s) else "read"
        layers: list[int] | None = [int(a or b) for a, b in LAYER_LIT.findall(s)] or None
        if layers is None:
            for var in LAYER_VAR.findall(s):
                layers = _loop_range(code, var)
        parts: list[tuple[str, list[int] | None]] = []
        if "embed_tokens" in s:
            idx = _ints(" ".join(re.findall(r"embed_tokens\.weight(?:\.data)?\[\s*([\d,\s\[\]]+)\]", s)))
            m = re.search(r"embed_tokens\.weight\s*,\s*\[([\d,\s]+)\]", s)  # scale_units(W, [ids], ...)
            idx += _ints(m.group(1)) if m else []
            if not idx and mode == "edit" and re.search(r"embed_tokens\.weight(?:\.data)?\[\s*[A-Za-z_]", s):
                # W[tid] = ... inside a loop over a table like [(13784, 0.72, +0.12), ...]: take its ids
                idx = [int(x) for x in re.findall(r"\(\s*(\d{3,})\s*,", code)]
            parts.append(("embed", idx or None))
        if "lm_head" in s:
            parts.append(("head", None))
        if re.search(r"\.mlp\b|down_proj|up_proj|gate_proj", s):
            parts.append(("mlp", None))
        if re.search(r"self_attn|linear_attn|o_proj|q_proj|k_proj|v_proj|out_proj|in_proj", s):
            parts.append(("attn", None))
        if re.search(r"layernorm", s):
            parts.append(("norm", None))
        elif re.search(r"model\.norm\b", s):
            out.append(touch("norm", None, mode, guess=True))
        if "named_parameters" in s and mode == "edit":
            out.append(touch("model", None, "edit", guess=True))
        for part, idx in parts:
            if part in ("embed", "head"):
                out.append(touch(part, None, mode, idx, guess=True))
            elif layers is None:
                out.append(touch(part, None, mode, guess=True))
            else:
                out.extend(touch(part, L, mode, guess=True) for L in layers)
        for name in RUN_TOOL.findall(s):
            out.extend(tool_touches(name, {}, ""))
    return _merge(out)


def tool_touches(name: str, args: dict, output: str) -> list[dict]:
    """Exact touches from a tool call's arguments and output."""
    a = args or {}
    run = [touch("model", None, "run")]
    if name == "python":
        t = code_touches(a.get("code", ""))
        rep = report_touches(output)
        if rep:  # the edit_report printed inside the call is exact: it replaces guessed edits, keeping
            # any rows the code named (e.g. scale_units(embed_tokens.weight, [248069], ...))
            guessed = {(x["part"], x["layer"]): x.get("index") for x in t if x["mode"] == "edit"}
            for r in rep:
                if guessed.get((r["part"], r["layer"])):
                    r["index"] = guessed[(r["part"], r["layer"])]
            t = [x for x in t if x["mode"] != "edit"] + rep
        return _merge(t)
    if name == "generate":
        return run
    if name == "edit_report":
        return report_touches(output)
    if name == "neuron_logits":
        return [touch("mlp", int(a.get("layer", 0)), "read", [int(a.get("neuron", 0))]), touch("head", None, "read")]
    if name == "direction_logits":
        return [touch("head", None, "read")]
    if name == "writers_of":
        by: dict[int, list[int]] = {}
        for L, n in NEURON_OUT.findall(output or ""):
            by.setdefault(int(L), []).append(int(n))
        return [touch("mlp", L, "read", ns) for L, ns in sorted(by.items())] or [touch("mlp", None, "read")]
    if name == "logit_lens":
        layers = a.get("layers") or [None]
        return run + [touch("resid", L, "read") for L in layers] + [touch("head", None, "read")]
    if name == "attention":
        return run + [touch("attn", int(a.get("layer", 0)), "read")]
    if name == "contrast_direction":
        return run + [touch(_point_part(a.get("point", "resid_post")), int(a.get("layer", 0)), "read")]
    if name == "capture":
        out = list(run)
        for p in a.get("points", []):
            m = re.match(r"([a-z_]+)\.?(\d+)?", str(p))
            if m:
                out.append(touch(_point_part(m.group(1)), int(m.group(2)) if m.group(2) else None, "read"))
        return out
    if name == "steer_test":
        return run + [touch(_point_part(a.get("point", "resid_post")), int(a.get("layer", 0)), "steer")]
    if name in ("sae_features", "sae_feature_info"):
        return [touch("resid", None, "read")]
    return []


def _point_part(point: str) -> str:
    if point.startswith(("mlp", "router", "expert")):
        return "mlp"
    if point.startswith(("attn", "q_", "k_", "value", "z")):
        return "attn"
    if point == "embeddings":
        return "embed"
    if point in ("lm_head", "final_norm"):
        return "head"
    return "resid"


def report_touches(output: str) -> list[dict]:
    out = []
    for name, n in REPORT_LINE.findall(output or ""):
        part, layer = tensor_part(name)
        out.append(touch(part, layer, "state", n=int(n.replace(",", ""))))
    return out


def delta_touches(delta: dict) -> list[dict]:
    """From an exact diff ``{tensor: {n, shape, rows, cols}}`` (per call or final)."""
    out = []
    for name, d in (delta or {}).items():
        part, layer = tensor_part(name)
        if part in ("embed", "head"):
            idx = d.get("rows")
        elif part == "mlp" and "down_proj" in name:
            idx = d.get("cols")  # down_proj columns are neurons
        elif part == "mlp":
            idx = d.get("rows")  # gate/up rows are neurons
        else:
            idx = None
        out.append({**touch(part, layer, "edit", idx, n=int(d.get("n", 0))), "tensor": name})
    return out


def _merge(ts: list[dict]) -> list[dict]:
    """One touch per (part, layer, mode), indices unioned."""
    seen: dict[tuple, dict] = {}
    for t in ts:
        k = (t["part"], t["layer"], t["mode"])
        if k not in seen:
            seen[k] = dict(t)
            continue
        cur = seen[k]
        if "index" in t:
            cur["index"] = sorted(set(cur.get("index", [])) | set(t["index"]))[:4096]
        if "n" in t:
            cur["n"] = cur.get("n", 0) + t["n"]
        if not t.get("guess"):
            cur.pop("guess", None)
    return list(seen.values())


class StepParser:
    """Feed transcript rows in order; get steps out. Incremental, so live tailing reuses it."""

    def __init__(self, decode: Callable[[list[int]], list[str]] | None = None):
        self.decode = decode
        self.i = 0
        self.t0: float | None = None
        self.budget: int | None = None
        self.forward_left: int | None = None
        self.target: str | None = None
        self.tokens = 0

    def _label(self, ts: list[dict]) -> list[dict]:
        if not self.decode:
            return ts
        for t in ts:
            if t["part"] in ("embed", "head") and t.get("index"):
                try:
                    t["labels"] = self.decode(t["index"][:64])
                except Exception:  # noqa: BLE001 - labels are a nicety
                    pass
        return ts

    def _step(self, kind: str, row: dict, **kw: Any) -> dict:
        t = float(row.get("t", 0) or 0)
        if self.t0 is None:
            self.t0 = t
        s = {"i": self.i, "kind": kind, "turn": row.get("turn"), "t": round(t - self.t0, 2),
             "forward_left": self.forward_left, "budget": self.budget, "tokens": self.tokens, **kw}
        if "touches" in s:
            s["touches"] = self._label(s["touches"])
        self.i += 1
        return s

    def feed(self, row: dict) -> list[dict]:
        k = row.get("kind")
        if k == "start":
            task = row.get("task", "")
            m = BUDGET.search(task)
            self.budget = int(m.group(1)) if m else None
            self.forward_left = self.budget
            m = TARGET.search(task)
            name = (m.group(1) or m.group(2)) if m else None
            self.target = (name if "/" in name else f"Qwen/{name}") if name else None
            return [self._step("start", row, agent=f"{row.get('provider')}:{row.get('model')}",
                               target=self.target, text=task, system=row.get("system", ""),
                               tools=row.get("tools", []))]
        if k == "assistant":
            self.tokens += int(row.get("input_tokens", 0) or 0) + int(row.get("output_tokens", 0) or 0)
            calls = [{"name": c.get("name"), "args": c.get("args", {})} for c in row.get("tool_calls", [])]
            if not (row.get("text") or "").strip() and not calls:
                return []
            return [self._step("think", row, text=row.get("text") or "", calls=calls)]
        if k == "tool":
            out = str(row.get("output", ""))
            m = FORWARD_LEFT.findall(out)
            if m:
                self.forward_left = int(m[-1])
            name = row.get("name", "")
            ts = delta_touches(row["weight_delta"]) if row.get("weight_delta") else []
            ts = _merge(ts + [t for t in tool_touches(name, row.get("args", {}), out)
                              if not (ts and t["mode"] == "edit")])
            head = REPORT_HEAD.search(out)
            return [self._step("tool", row, tool=name, args=row.get("args", {}),
                               output=out if len(out) <= OUTPUT_CHARS else out[:OUTPUT_CHARS] + "\n...",
                               touches=ts, weights_changed=int(head.group(1).replace(",", "")) if head else None,
                               error="Traceback" in out or out.startswith("refused"))]
        if k == "jail_violations":
            return [self._step("violations", row, items=row.get("items", []))]
        if k == "end":
            return [self._step("end", row, result={x: row.get(x) for x in (
                "stop_reason", "turns", "submitted", "forward_used", "weights_changed", "weights_total",
                "spent_usd", "tokens")})]
        return []


def read_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # a half-written last line while live
                pass
    return rows


def final_diff(cell_dir: Path) -> dict | None:
    """The run's exact final edit, if saved: ``edit_diff.json`` or the older sparse ``edit_diff.pt``."""
    j = cell_dir / "edit_diff.json"
    if j.exists():
        return json.loads(j.read_text(encoding="utf-8"))
    p = cell_dir / "edit_diff.pt"
    if p.exists():
        import torch

        out = {}
        for name, d in torch.load(p, map_location="cpu").items():
            idx = d["idx"]
            rows = idx[:, 0].unique().tolist() if idx.ndim == 2 else []
            cols = idx[:, 1].unique().tolist() if idx.ndim == 2 and idx.shape[1] > 1 else []
            out[name] = {"shape": list(d["shape"]), "n": int(idx.shape[0]), "rows": rows, "cols": cols}
        return out
    return None


def parse_run(cell_dir: Path, decode: Callable[[list[int]], list[str]] | None = None) -> dict:
    p = StepParser(decode)
    steps = [s for r in read_rows(cell_dir / "transcript.jsonl") for s in p.feed(r)]
    diff = final_diff(cell_dir)
    if diff and steps and steps[-1]["kind"] == "end":
        steps[-1]["touches"] = p._label(delta_touches(diff))
    return {"steps": steps, "target": p.target, "budget": p.budget, "final_diff": diff}
