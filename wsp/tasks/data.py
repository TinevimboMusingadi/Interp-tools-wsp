"""Practice data the agent may use (train splits only). Evaluation data lives in ``wsp.eval``."""

from __future__ import annotations

import random

from wsp.eval import behaviour


def practice_math(n_each: int = 200, seed: int = 0) -> list[dict]:
    """GSM8K-train and MATH-train problems with answers, for the agent's forward passes."""
    from datasets import load_dataset

    from wsp.eval.math import gsm8k_gold, last_boxed

    rng = random.Random(seed)
    gsm = list(load_dataset("openai/gsm8k", "main", split="train"))
    out = [{"source": "gsm8k", "problem": ex["question"], "answer": gsm8k_gold(ex["answer"])}
           for ex in rng.sample(gsm, n_each)]
    try:
        mth = list(load_dataset("EleutherAI/hendrycks_math", "algebra", split="train"))
        out += [{"source": "math", "problem": ex["problem"], "answer": last_boxed(ex["solution"]) or ""}
                for ex in rng.sample(mth, n_each)]
    except Exception:  # noqa: BLE001 - MATH-train mirror unavailable: GSM8K alone still works
        pass
    rng.shuffle(out)
    return out


def practice_sycophancy(n_items: int = 60, seed: int = 0) -> list[dict]:
    """ARC-train probes (with and without a user suggestion) the agent may run."""
    return behaviour.sycophancy_set(behaviour.arc_items("train", n_items, seed), seed)


def practice_chat(n: int = 100, seed: int = 0) -> list[str]:
    """Ordinary instructions for the backdoor audit (the agent must find the trigger itself)."""
    from datasets import load_dataset

    ds = list(load_dataset("HuggingFaceH4/CodeAlpaca_20K", split="train"))
    return [ex["prompt"] for ex in random.Random(seed).sample(ds, n)]
