"""Measure batched generation throughput for evaluation, dynamic vs static KV cache.

    python scripts/bench_generate.py --model Qwen/Qwen3.5-0.8B --device tpu --batch 16 --tokens 256

On TPU every new tensor shape compiles, and HF's default dynamic cache changes shape at every decode
step; a static cache keeps one shape. This picks the eval setting and sizes eval sets to the budget.
"""

from __future__ import annotations

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def run(model, tok, prompts, tokens, cache, device):
    enc = tok(prompts, return_tensors="pt", padding=True).to(device)
    kw = {"max_new_tokens": tokens, "min_new_tokens": tokens, "do_sample": False, "pad_token_id": tok.pad_token_id}
    if cache:
        kw["cache_implementation"] = cache
    t = time.time()
    with torch.no_grad():
        out = model.generate(**enc, **kw)
    _ = out.cpu()
    return time.time() - t, out.shape[1] - enc["input_ids"].shape[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--device", default="tpu")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=256)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16).to(a.device).eval()
    prompts = [f"Problem {i}: what is {i} * 17? Think step by step." for i in range(a.batch)]
    for cache in (None, "static"):
        try:
            for attempt in ("first (compiles)", "second"):
                dt, n = run(model, tok, prompts, a.tokens, cache, a.device)
                print(f"cache={cache or 'dynamic'} {attempt}: {dt:.1f}s for {a.batch}x{n} tokens "
                      f"= {a.batch * n / dt:.0f} tok/s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"cache={cache}: failed {type(e).__name__}: {str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
