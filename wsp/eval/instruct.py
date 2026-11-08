"""Instruction-following checks over a subset of IFEval instruction types.

This is **not** the official IFEval score: it implements the verifiable instruction types listed in
``CHECKS`` and evaluates IFEval prompts whose instructions are all supported (prompt-level strict).
Reported everywhere as "IFEval-subset". For a headline number comparable to papers, run the official
harness on the saved checkpoints; this subset exists so every condition is scored identically and
cheaply inside the run.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable


def _words(t: str) -> int:
    return len(re.findall(r"\b\w+\b", t))


def _sentences(t: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s])


def _cmp(n: int, rel: str, k: int) -> bool:
    return n < k if rel == "less than" else n >= k


CHECKS: dict[str, Callable[[str, dict], bool]] = {
    "punctuation:no_comma": lambda t, a: "," not in t,
    "change_case:english_lowercase": lambda t, a: t == t.lower(),
    "change_case:english_capital": lambda t, a: t == t.upper(),
    "keywords:existence": lambda t, a: all(k.lower() in t.lower() for k in a.get("keywords") or []),
    "keywords:forbidden_words": lambda t, a: not any(
        re.search(r"\b" + re.escape(w) + r"\b", t, re.I) for w in a.get("forbidden_words") or []),
    "keywords:frequency": lambda t, a: _cmp(len(re.findall(re.escape(a["keyword"]), t, re.I)),
                                            a["relation"], a["frequency"]),
    "length_constraints:number_words": lambda t, a: _cmp(_words(t), a["relation"], a["num_words"]),
    "length_constraints:number_sentences": lambda t, a: _cmp(_sentences(t), a["relation"], a["num_sentences"]),
    "detectable_format:number_bullet_lists": lambda t, a: len(
        re.findall(r"^\s*[*-]\s", t, re.M)) == a["num_bullets"],
    "detectable_format:title": lambda t, a: re.search(r"<<[^\n]+>>", t) is not None,
    "detectable_format:json_format": lambda t, a: _is_json(t),
    "startend:end_checker": lambda t, a: t.strip().endswith(a["end_phrase"].strip()),
    "startend:quotation": lambda t, a: t.strip().startswith('"') and t.strip().endswith('"'),
    "detectable_content:postscript": lambda t, a: a["postscript_marker"] in t,
}


def _is_json(t: str) -> bool:
    s = t.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        json.loads(s)
        return True
    except ValueError:
        return False


def supported(example: dict) -> bool:
    return all(i in CHECKS for i in example["instruction_id_list"])


def follows(response: str, example: dict) -> bool:
    for iid, kwargs in zip(example["instruction_id_list"], example["kwargs"]):
        kw = {k: v for k, v in (kwargs or {}).items() if v is not None}
        try:
            if not CHECKS[iid](response, kw):
                return False
        except (KeyError, TypeError):
            return False
    return True
