"""Exact sparse diff of a run's saved weights against the base model, for replay.

    python -m wsp.studio.diff runs/p2_loop_fix [--model Qwen/Qwen3.5-0.8B]

Writes ``edit_diff.json`` next to each ``weights.pt``: ``{tensor: {shape, n, rows, cols}}`` where
``rows`` / ``cols`` are the output rows / input columns that changed (capped at 4096 each). Small enough to
copy back from a TPU VM instead of the 1.5 GB weights.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

CAP = 4096


def sparse_diff(base: dict[str, torch.Tensor], edited: dict[str, torch.Tensor]) -> dict:
    out = {}
    for name, new in edited.items():
        ref = base.get(name)
        if ref is None or ref.shape != new.shape:
            continue
        ch = ref.to(new.dtype) != new
        n = int(ch.sum())
        if not n:
            continue
        d = {"shape": list(new.shape), "n": n}
        if ch.ndim == 2:
            d["rows"] = ch.any(1).nonzero().flatten()[:CAP].tolist()
            d["cols"] = ch.any(0).nonzero().flatten()[:CAP].tolist()
        elif ch.ndim == 1:
            d["rows"] = ch.nonzero().flatten()[:CAP].tolist()
        out[name] = d
    return out


def base_state(model_id: str) -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    return {k: v.detach() for k, v in m.state_dict().items()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    a = ap.parse_args(argv)
    files = sorted(Path(a.root).rglob("weights.pt"))
    if not files:
        print("no weights.pt under", a.root)
        return 1
    base = base_state(a.model)
    for f in files:
        sd = torch.load(f, map_location="cpu")
        if not any(k in base for k in sd):  # name prefixes differ (e.g. a wrapper): try stripping one level
            sd = {k.split(".", 1)[1]: v for k, v in sd.items()}
        d = sparse_diff(base, sd)
        (f.parent / "edit_diff.json").write_text(json.dumps(d), encoding="utf-8")
        print(f.parent.name, sum(x["n"] for x in d.values()), "weights in", len(d), "tensors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
