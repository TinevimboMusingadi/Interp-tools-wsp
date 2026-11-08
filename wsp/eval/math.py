"""Math answer extraction and grading for MATH-500 and GSM8K.

Grading is exact match after normalisation, with numeric comparison when both sides parse as
numbers. It is deliberately simple and identical across conditions; absolute numbers may differ
slightly from other graders (e.g. ones using symbolic equivalence), so we compare conditions to
each other with the same grader, and report the grader with every number.
"""

from __future__ import annotations

import re

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def last_boxed(text: str) -> str | None:
    i = text.rfind("\\boxed")
    if i < 0:
        i = text.rfind("\\fbox")
        if i < 0:
            return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def normalize(ans: str) -> str:
    s = ans.strip().strip("$").strip()
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""), ("dfrac", "frac"),
                 ("tfrac", "frac"), ("^\\circ", ""), ("^{\\circ}", ""), ("\\%", ""), ("%", ""), ("\\$", "")]:
        s = s.replace(a, b)
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = s.rstrip(".")
    if re.fullmatch(r"-?\d+\.0+", s):
        s = s.split(".")[0]
    return s


def _as_float(s: str) -> float | None:
    s = s.replace(",", "")
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(\d+)\}", s)
    if m and int(m.group(2)):
        return int(m.group(1)) / int(m.group(2))
    return None


def equivalent(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    p, g = normalize(pred), normalize(gold)
    if p == g:
        return True
    fp, fg = _as_float(p), _as_float(g)
    return fp is not None and fg is not None and abs(fp - fg) <= 1e-6 * max(1.0, abs(fg))


def extract_math(text: str) -> str | None:
    return last_boxed(text)


def extract_gsm8k(text: str) -> str | None:
    b = last_boxed(text)
    if b is not None:
        return b
    m = re.search(r"####\s*(" + _NUM.pattern + ")", text)
    if m:
        return m.group(1)
    nums = _NUM.findall(text)
    return nums[-1] if nums else None


def gsm8k_gold(answer_field: str) -> str:
    return answer_field.split("####")[-1].strip().replace(",", "")


MATH_PROMPT = "{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}."
