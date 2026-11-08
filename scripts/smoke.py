"""Real-model smoke test of the sandbox: load, read weights, check the guards, spend a forward pass.

    .venv/Scripts/python scripts/smoke.py --model Qwen/Qwen3.5-0.8B [--device cpu] [--dtype bfloat16]

Costs no API money: no agent LLM is called. Exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import sys
import time

from wsp.harness.sandbox import Sandbox

CHECKS = [
    ("weights readable",
     "python", {"code": "print(type(model).__name__, model.n_layers, model.d_model, "
                        "sum(p.numel() for p in model.hf_model.parameters()))"},
     lambda out: "Traceback" not in out),
    ("forward guard blocks capture",
     "python", {"code": "from interp_engine import capture\n"
                        "capture(model, model.to_tokens('hi'), ['resid_post.3'])"},
     lambda out: "costs a forward pass" in out),
    ("gradient guard",
     "python", {"code": "x = torch.ones(2, requires_grad=True)\n(x * 2).sum().backward()"},
     lambda out: "gradients are not allowed" in out),
    ("generate spends one pass",
     "generate", {"prompts": ["What is 12*7? Answer:", "The capital of France is"], "max_new_tokens": 16},
     lambda out: "forward passes left: 1" in out),
    ("free weight edit is diffed",
     "python", {"code": "model.hf_model.lm_head.weight.data[0, :4] += 1.0\nprint('edited')"},
     lambda out: "edited" in out),
    ("edit report",
     "edit_report", {},
     lambda out: "weights changed" in out and not out.startswith("0 of")),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16")
    a = ap.parse_args()

    t = time.time()
    failed = 0
    with Sandbox(loader="wsp.harness.env:eager_loader",
                 loader_kwargs={"model_id": a.model, "device": a.device, "dtype": a.dtype},
                 toolset="c0", forward_budget=2) as sb:
        print(f"loaded {a.model} in {time.time() - t:.1f}s; tools: {[s.name for s in sb.tool_specs]}")
        for name, tool, args, ok in CHECKS:
            t1 = time.time()
            out = sb.call(tool, args)
            passed = ok(out)
            failed += not passed
            print(f"[{'PASS' if passed else 'FAIL'}] {name} ({time.time() - t1:.1f}s)")
            print("   " + out.strip().replace("\n", "\n   ")[:600])
        print("status:", sb.status())
    print(f"total {time.time() - t:.1f}s, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
