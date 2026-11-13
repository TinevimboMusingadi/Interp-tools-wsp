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

# Interp tools (c1) on the real model: each must run, not just exist. Qwen3.5 mixes linear-attention and
# full-attention layers, so attention() is tried on a full-attention layer (the config's layer_types).
C1_CHECKS = [
    ("neuron_logits (free)", "neuron_logits", {"layer": 10, "neuron": 0, "k": 5}, lambda o: "Traceback" not in o),
    ("logit_lens", "logit_lens", {"prompt": "2 + 2 =", "layers": [0, 8, 16, 23], "k": 3},
     lambda o: "L23" in o and "Traceback" not in o),
    ("contrast_direction", "contrast_direction",
     {"prompts_a": ["Solve: 3+4=", "Compute 9*2"], "prompts_b": ["The weather today", "My favourite colour"],
      "layer": 12, "name": "math_dir"}, lambda o: "ws['math_dir']" in o),
    ("direction_logits (free)", "direction_logits", {"direction": "math_dir", "k": 5}, lambda o: "Traceback" not in o),
    ("writers_of (free)", "writers_of", {"direction": "math_dir", "k": 5}, lambda o: "neuron" in o),
    ("steer_test ablate", "steer_test", {"prompts": ["What is 5+7?"], "direction": "math_dir", "layer": 12,
                                         "method": "ablate", "max_new_tokens": 12}, lambda o: "[0]" in o),
    ("attention on a full-attention layer", "python",
     {"code": "lt = getattr(model.hf_model.config, 'layer_types', None) or "
              "getattr(getattr(model.hf_model.config, 'text_config', None), 'layer_types', None)\n"
              "print('FULL', [i for i, t in enumerate(lt or []) if 'full' in str(t)])"},
     lambda o: "FULL" in o),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--toolset", default="c0", choices=["c0", "c1"])
    a = ap.parse_args()
    checks = CHECKS if a.toolset == "c0" else C1_CHECKS
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # decoded tokens break cp1252 consoles

    t = time.time()
    failed = 0
    with Sandbox(loader="wsp.harness.env:eager_loader",
                 loader_kwargs={"model_id": a.model, "device": a.device, "dtype": a.dtype},
                 toolset=a.toolset, forward_budget=2 if a.toolset == "c0" else 10) as sb:
        print(f"loaded {a.model} in {time.time() - t:.1f}s; tools: {[s.name for s in sb.tool_specs]}")
        for name, tool, args, ok in checks:
            t1 = time.time()
            out = sb.call(tool, args)
            passed = ok(out)
            failed += not passed
            print(f"[{'PASS' if passed else 'FAIL'}] {name} ({time.time() - t1:.1f}s)")
            print("   " + out.strip().replace("\n", "\n   ")[:600 if passed else 20000])
            if tool == "python" and out.startswith("FULL"):  # then exercise attention() on the first one
                full = [int(x) for x in out.split("[", 1)[1].split("]", 1)[0].split(",") if x.strip()]
                if full:
                    att = sb.call("attention", {"prompt": "The cat sat on the mat", "layer": full[0], "k": 3})
                    passed = "head 0" in att
                    failed += not passed
                    print(f"[{'PASS' if passed else 'FAIL'}] attention layer {full[0]}")
                    print("   " + att.strip().replace("\n", "\n   ")[:400])
        print("status:", sb.status())
    print(f"total {time.time() - t:.1f}s, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
