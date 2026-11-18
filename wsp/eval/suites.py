"""Evaluation suites: the functions a task hands to ``Sandbox.evaluate`` once the agent is done.

Each takes the (edited) model first and returns a JSON-able dict. Held-out data is loaded here, in
the evaluator, never in the agent's sandbox namespace.
"""

from __future__ import annotations

import random
from typing import Any

from wsp.eval import behaviour, instruct
from wsp.eval import math as m
from wsp.eval.generate import Sampling, generate


def _sampling(s: dict | None) -> Sampling:
    return Sampling(**(s or {}))


def math500(model: Any, n: int | None = None, sampling: dict | None = None, seed: int = 0) -> dict:
    from datasets import load_dataset

    ds = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    if n:
        ds = random.Random(seed).sample(ds, n)
    gens = generate(model, [m.MATH_PROMPT.format(problem=ex["problem"]) for ex in ds], _sampling(sampling), seed=seed)
    correct = [m.equivalent(m.extract_math(g.text), ex["answer"]) for g, ex in zip(gens, ds)]
    return {"math500_acc": sum(correct) / len(ds), "math500_hit_limit": sum(g.hit_limit for g in gens) / len(ds),
            "math500_mean_tokens": sum(g.n_tokens for g in gens) / len(ds), "n": len(ds),
            "grader": "wsp.eval.math exact+numeric",
            # per problem, in a fixed order (sampled with ``seed``), for paired bootstrap CIs across cells
            "math500_items": [[int(c), int(g.hit_limit)] for c, g in zip(correct, gens)]}


def gsm8k(model: Any, n: int | None = None, sampling: dict | None = None, seed: int = 0) -> dict:
    from datasets import load_dataset

    ds = list(load_dataset("openai/gsm8k", "main", split="test"))
    if n:
        ds = random.Random(seed).sample(ds, n)
    gens = generate(model, [m.MATH_PROMPT.format(problem=ex["question"]) for ex in ds], _sampling(sampling), seed=seed)
    correct = [m.equivalent(m.extract_gsm8k(g.text), m.gsm8k_gold(ex["answer"])) for g, ex in zip(gens, ds)]
    return {"gsm8k_acc": sum(correct) / len(ds), "gsm8k_hit_limit": sum(g.hit_limit for g in gens) / len(ds),
            "n": len(ds), "gsm8k_items": [[int(c), int(g.hit_limit)] for c, g in zip(correct, gens)]}


def ifeval_subset(model: Any, n: int | None = None, sampling: dict | None = None, seed: int = 0) -> dict:
    from datasets import load_dataset

    ds = [ex for ex in load_dataset("google/IFEval", split="train") if instruct.supported(ex)]
    if n:
        ds = random.Random(seed).sample(ds, min(n, len(ds)))
    gens = generate(model, [ex["prompt"] for ex in ds], _sampling(sampling), seed=seed)
    ok = [instruct.follows(g.text, ex) for g, ex in zip(gens, ds)]
    return {"ifeval_subset_acc": sum(ok) / len(ds), "n": len(ds), "note": "IFEval-subset, not official",
            "ifeval_subset_items": [[int(o), int(g.hit_limit)] for o, g in zip(ok, gens)]}


def sycophancy(model: Any, n_items: int = 100, split: str = "test", seed: int = 0) -> dict:
    rows = behaviour.sycophancy_set(behaviour.arc_items(split, n_items, seed), seed)
    gens = generate(model, [r["prompt"] for r in rows], Sampling(max_new_tokens=8), seed=seed)
    return behaviour.score_sycophancy(rows, [g.text for g in gens])


def backdoor(model: Any, prompts: list[str], trigger: str, target: str, max_new_tokens: int = 48) -> dict:
    s = Sampling(max_new_tokens=max_new_tokens)
    trig = generate(model, [f"{trigger} {p}" for p in prompts], s)
    clean = generate(model, prompts, s)
    return behaviour.score_backdoor([g.text for g in trig], [g.text for g in clean], target)


def capability(model: Any, n: int = 100, sampling: dict | None = None, seed: int = 0) -> dict:
    """Quick regression suite after an edit that should not touch these: GSM8K + IFEval-subset."""
    out = gsm8k(model, n, sampling, seed)
    out.update(ifeval_subset(model, n, sampling, seed))
    return out


def run(model: Any, suites: list[dict]) -> dict:
    """Run several suites: ``[{"fn": "math500", "kwargs": {...}}, ...]`` -> merged dict."""
    out: dict = {}
    for s in suites:
        res = globals()[s["fn"]](model, **s.get("kwargs", {}))
        out.update({f"{s.get('prefix', '')}{k}": v for k, v in res.items()})
    return out
