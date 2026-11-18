"""Aggregate an experiment's results into REPORT.md, and audit its transcripts.

    python -m wsp.report experiments/p2_loop_fix

The table groups cells by (variant, condition, budget, provider) and shows mean [min, max] across
seeds for every numeric metric in ``eval`` and ``grade``, plus cost and edit size. Nothing is typed by
hand: every number in a blog post should come from a REPORT.md generated here.

The audit flags transcripts whose agent code touched the filesystem or tried to reach answer keys,
evaluation data or harness logs, so a reviewer can check no run peeked.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

from wsp.harness.agent import RUNS_DIR

SUSPICIOUS = re.compile(r"\.keys|organisms|ledger\.csv|transcript\.jsonl|MATH-500|IFEval|gsm8k.*test|"
                        r"\bos\.(listdir|walk|scandir)|pathlib|glob\.glob|subprocess|socket|requests\.", re.I)


def load_results(exp_dir: Path) -> list[dict]:
    return [json.loads(f.read_text(encoding="utf-8")) for f in sorted((exp_dir / "results").glob("*.json"))]


def _group_key(r: dict) -> tuple:
    c = r["cell"]
    if c.get("condition") == "baseline":
        return (c["id"], "baseline", "-", "-")
    var = (c.get("variant") or {}).get("name", "-")
    prov = c["provider"].get("short") or c["provider"]["model"]
    return (var, c["condition"], c["budget"], prov)


def _metrics(r: dict) -> dict[str, float]:
    m = {k: float(v) for k, v in {**r.get("eval", {}), **r.get("grade", {})}.items()
         if isinstance(v, (int, float)) and not isinstance(v, bool) and k != "n"}
    if "agent" in r:
        a = r["agent"]
        m["usd"] = a["spent_usd"]
        m["fwd_used"] = a["forward_used"]
        m["weights_changed_pct"] = 100 * a["weights_changed"] / max(1, a["weights_total"])
    return m


def summarize(results: list[dict]) -> tuple[list[str], list[tuple]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        groups[_group_key(r)].append(_metrics(r))
    cols = sorted({k for ms in groups.values() for m in ms for k in m})
    rows = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        ms = groups[key]
        cells = []
        for c in cols:
            vals = [m[c] for m in ms if c in m]
            if not vals:
                cells.append("")
            elif len(vals) == 1:
                cells.append(f"{vals[0]:.3g}")
            else:
                cells.append(f"{statistics.mean(vals):.3g} [{min(vals):.3g}, {max(vals):.3g}]")
        rows.append((*key, len(ms), *cells))
    return cols, rows


def _items(r: dict, suite: str, col: int) -> list[float] | None:
    """Per-problem scores (column 0: correct, 1: hit the token limit) in the eval's fixed problem order."""
    it = r.get("eval", {}).get(f"{suite}_items")
    return [float(x[col]) for x in it] if it else None


def _pooled(rs: list[dict], suite: str, col: int) -> list[float] | None:
    """Per-problem mean over seeds: one score per problem, so the bootstrap resamples problems."""
    lists = [x for x in (_items(r, suite, col) for r in rs) if x]
    if not lists or len({len(x) for x in lists}) != 1:
        return None
    return [sum(v) / len(v) for v in zip(*lists)]


def paired_bootstrap(a: list[float], b: list[float], n_boot: int = 4000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) over problems, with a 95% percentile CI from resampling problems."""
    import random

    d = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    n = len(d)
    boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(d) / n, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]


def comparisons(results: list[dict], suite: str = "math500", treat: str = "c1", control: str = "c0",
                col: int = 0) -> list[dict]:
    """``treat`` vs ``control`` at each (variant, budget, provider), seeds pooled; and each condition vs the
    unedited baseline. Paired by problem (every cell is evaluated on the same problems)."""
    base = next((r for r in results if r["cell"].get("condition") == "baseline" and _items(r, suite, col)), None)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        if "agent" in r:
            groups[_group_key(r)].append(r)
    out = []
    for (var, cond, budget, prov), rs in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        a = _pooled(rs, suite, col)
        if a is None:
            continue
        row = {"variant": var, "condition": cond, "budget": budget, "provider": prov, "seeds": len(rs),
               "mean": sum(a) / len(a)}
        if base:
            row["vs_baseline"] = paired_bootstrap(a, _items(base, suite, col))
        if cond == treat and (var, control, budget, prov) in groups:
            b = _pooled(groups[(var, control, budget, prov)], suite, col)
            if b and len(b) == len(a):
                row["vs_control"] = paired_bootstrap(a, b)
        out.append(row)
    return out


def verdict(comps: list[dict], prereg: dict) -> str:
    """The pre-registered test: treat - control >= ``min_delta`` at ``budget`` on at least ``min_agents``
    agents, each with a 95% CI excluding 0."""
    rows = [c for c in comps if c.get("vs_control") and c["budget"] == prereg["budget"]]
    passed = [c for c in rows if c["vs_control"][0] >= prereg["min_delta"] and c["vs_control"][1] > 0]
    need = prereg.get("min_agents", 1)
    ok = len(passed) >= need
    who = ", ".join(c["provider"] for c in passed) or "none"
    return (f"{'MET' if ok else 'NOT MET'}: {len(passed)} of {len(rows)} agents show {prereg.get('treat', 'c1')} - "
            f"{prereg.get('control', 'c0')} >= {prereg['min_delta']:+.0%} on {prereg.get('suite', 'math500')} at "
            f"budget {prereg['budget']} with a 95% CI above 0 (needed {need}; passed: {who}).")


def _fmt(t: tuple[float, float, float]) -> str:
    return f"{t[0] * 100:+.1f} [{t[1] * 100:+.1f}, {t[2] * 100:+.1f}]"


def audit(results: list[dict]) -> list[str]:
    flags = []
    for r in results:
        if "agent" not in r:
            continue
        tr = RUNS_DIR / r["agent"]["run_id"] / "transcript.jsonl"
        if not tr.exists():
            flags.append(f"{r['cell']['id']}: transcript missing")
            continue
        for line in tr.read_text(encoding="utf-8").splitlines():
            ev = json.loads(line)
            if ev["kind"] == "tool" and ev["name"] in ("python", "save_tool"):
                code = json.dumps(ev["args"])
                hit = SUSPICIOUS.search(code)
                if hit or "open(" in code:
                    flags.append(f"{r['cell']['id']} turn {ev['turn']}: {hit.group(0) if hit else 'open('}")
    return flags


def render(exp_dir: Path) -> str:
    results = load_results(exp_dir)
    cols, rows = summarize(results)
    head = ["variant", "condition", "budget", "provider", "n", *cols]
    lines = [f"# {exp_dir.name}: results", "",
             "Generated by `python -m wsp.report`. Cells: mean [min, max] over seeds.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(x) for x in row) + " |" for row in rows]
    cfg_path = exp_dir / "config.yaml"
    prereg = {}
    if cfg_path.exists():
        import yaml

        prereg = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}).get("prereg") or {}
    suite = prereg.get("suite", "math500")
    comps = comparisons(results, suite, prereg.get("treat", "c1"), prereg.get("control", "c0"))
    if comps:
        lines += ["", f"## {suite}: paired bootstrap over problems (points, 95% CI)", "",
                  "Seeds pooled per problem; every cell is scored on the same problems, so differences are paired.",
                  "", "| variant | condition | budget | provider | seeds | accuracy | vs baseline | vs control |",
                  "|---|---|---|---|---|---|---|---|"]
        lines += [f"| {c['variant']} | {c['condition']} | {c['budget']} | {c['provider']} | {c['seeds']} | "
                  f"{c['mean'] * 100:.1f}% | {_fmt(c['vs_baseline']) if 'vs_baseline' in c else ''} | "
                  f"{_fmt(c['vs_control']) if 'vs_control' in c else ''} |" for c in comps]
        if prereg.get("min_delta") is not None:
            lines += ["", f"**Pre-registered test:** {verdict(comps, prereg)}"]
    flags = audit(results)
    lines += ["", "## Transcript audit", ""]
    lines += [f"- {f}" for f in flags] if flags else ["No file-system or answer-key access found in agent code."]
    total = sum(r.get("agent", {}).get("spent_usd", 0) for r in results)
    lines += ["", f"Total agent spend: ${total:.2f} over {sum('agent' in r for r in results)} runs."]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    a = ap.parse_args(argv)
    d = Path(a.experiment)
    (d / "REPORT.md").write_text(render(d), encoding="utf-8")
    print(f"wrote {d / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
