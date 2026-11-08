"""Run an experiment: the cross product of conditions x budgets x providers x seeds for one task.

    python -m wsp.run experiments/p2_loop_fix/config.yaml --dry-run
    python -m wsp.run experiments/p2_loop_fix/config.yaml [--only c1_b10] [--max-cells 1]

Each cell writes ``results/<cell>.json`` (agent result + held-out evaluation + grade) and is skipped
if that file exists, so a run can be stopped and resumed. ``--dry-run`` prints the cell list and a
cost estimate without calling any model. A ``baseline`` cell evaluates the unedited model once.
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from wsp.harness.ledger import Budget, cost_usd, load_prices
from wsp.harness.sandbox import Sandbox
from wsp.harness.agent import RUNS_DIR, run_agent
from wsp.tasks.registry import SYSTEM, get_task


@dataclass
class Cell:
    id: str
    condition: str
    budget: int
    provider: dict
    seed: int
    variant: dict | None = None  # extra task kwargs, e.g. {"name": "p1", "pair": 1}


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_dir"] = Path(path).resolve().parent
    return cfg


def cells(cfg: dict) -> list[Cell]:
    out = []
    variants = cfg.get("variants") or [None]
    for var, cond, budget, prov, seed in itertools.product(variants, cfg["conditions"], cfg["budgets"],
                                                           cfg["providers"], cfg["seeds"]):
        short = prov.get("short") or prov["model"].replace("/", "-")
        pre = f"{var['name']}_" if var else ""
        out.append(Cell(f"{pre}{cond}_b{budget}_{short}_s{seed}", cond, budget, prov, seed, var))
    return out


def _task(cfg: dict, variant: dict | None):
    kw = dict(cfg.get("task_kwargs", {}))
    kw.update({k: v for k, v in (variant or {}).items() if k != "name"})
    return get_task(cfg["task"], **kw)


def estimate(cfg: dict) -> dict:
    """Expected and worst-case dollars. Expected uses cfg['estimate'] per-turn token counts (calibrate
    from the first real cell's transcript); worst case is every cell hitting its dollar cap."""
    prices = load_prices()
    est = cfg.get("estimate", {"turns": 60, "input_tokens_per_turn": 30000, "output_tokens_per_turn": 1500})
    cap = cfg["caps"]["max_usd"]
    rows, total = [], 0.0
    for c in cells(cfg):
        per_turn = cost_usd(c.provider["model"], est["input_tokens_per_turn"], est["output_tokens_per_turn"], prices)
        expected = min(cap, per_turn * est["turns"])
        rows.append((c.id, expected))
        total += expected
    return {"cells": len(rows), "expected_usd": round(total, 2), "worst_case_usd": round(cap * len(rows), 2),
            "rows": rows}


def run_cell(cfg: dict, cell: Cell, results: Path) -> dict:
    from wsp.harness.providers import make_provider

    task = _task(cfg, cell.variant)
    run_id = f"{cfg['id']}/{cell.id}"
    runs_dir = Path(cfg.get("runs_dir", RUNS_DIR))
    lib = None
    if cell.condition == "c2":
        lib = str(runs_dir / run_id / "library")
        src = cfg.get("library_dir")
        if src and Path(cfg["_dir"], src).exists():
            shutil.copytree(Path(cfg["_dir"], src), lib, dirs_exist_ok=True)
    caps = cfg["caps"]
    budget_kw = {"ledger_path": Path(cfg["ledger"])} if cfg.get("ledger") else {}
    budget = Budget(run_id.replace("/", "__"), max_usd=caps["max_usd"], max_tokens=caps["max_tokens"], **budget_kw)
    t0 = time.time()
    with Sandbox(loader=cfg.get("loader", "wsp.harness.env:eager_loader"), loader_kwargs=task.loader_kwargs,
                 toolset=cell.condition, forward_budget=cell.budget, extras=task.extras(lib)) as sb:
        provider = make_provider(cell.provider["kind"], cell.provider["model"], **cell.provider.get("kwargs", {}))
        res = run_agent(run_id=run_id, provider=provider, sandbox=sb, system=SYSTEM,
                        task_prompt=task.task_prompt(cell.condition, cell.budget), budget=budget,
                        max_turns=caps.get("max_turns", 200), runs_dir=runs_dir)
        weights = _save_weights(sb, task.loader_kwargs, runs_dir / run_id)
    # Evaluate in a fresh process that loads only the saved weights, so hooks, monkeypatches or config
    # changes the agent left on the live objects cannot affect the score: only weight edits count.
    evaluation = _evaluate(cfg, task, _with_weights(task.loader_kwargs, weights)) if task.eval_suites else {}
    if not cfg.get("save_weights", True):
        for w in weights.values():
            Path(w).unlink(missing_ok=True)
    grade = task.grade(res.answer or {}, res.submitted or "", task.hidden) if task.grade else {}
    out = {"cell": asdict(cell), "agent": asdict(res), "eval": evaluation, "grade": grade,
           "wall_s": round(time.time() - t0, 1), "task": task.id}
    (results / f"{cell.id}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _save_weights(sb: Sandbox, loader_kwargs: dict, out_dir: Path) -> dict:
    names = list(loader_kwargs["models"]) if "models" in loader_kwargs else [None]
    return {n: sb.save_weights(str(out_dir / (f"weights_{n}.pt" if n else "weights.pt")), n) for n in names}


def _with_weights(loader_kwargs: dict, weights: dict) -> dict:
    if "models" in loader_kwargs:
        return {"models": {k: {**v, "state_dict": weights[k]} for k, v in loader_kwargs["models"].items()}}
    return {**loader_kwargs, "state_dict": weights[None]}


def _evaluate(cfg: dict, task, loader_kwargs: dict) -> dict:
    with Sandbox(loader=cfg.get("loader", "wsp.harness.env:eager_loader"), loader_kwargs=loader_kwargs,
                 toolset="c0", forward_budget=0, extras=task.extras()) as sb:
        return sb.evaluate(cfg.get("eval_fn", "wsp.eval.suites:run"), suites=task.eval_suites)


def run_baseline(cfg: dict, results: Path, variant: dict | None = None) -> dict:
    task = _task(cfg, variant)
    name = f"baseline_{variant['name']}" if variant else "baseline"
    evaluation = _evaluate(cfg, task, task.loader_kwargs) if task.eval_suites else {}
    out = {"cell": {"id": name, "condition": "baseline"}, "eval": evaluation, "task": task.id}
    (results / f"{name}.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", help="substring filter on cell ids")
    ap.add_argument("--max-cells", type=int)
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    results = cfg["_dir"] / "results"
    todo = [c for c in cells(cfg) if (not a.only or a.only in c.id) and not (results / f"{c.id}.json").exists()]
    if a.max_cells:
        todo = todo[: a.max_cells]
    est = estimate(cfg)
    print(f"{cfg['id']}: {est['cells']} cells, {len(todo)} to run. "
          f"Expected ${est['expected_usd']}, worst case ${est['worst_case_usd']} (all cells).")
    prices = load_prices()
    unverified = sorted({c.provider["model"] for c in todo if not prices.get(c.provider["model"], {}).get("verified")})
    if unverified:
        print(f"Unverified prices for {unverified}: those cells will be refused (set verified: true in prices.yaml).")
    if a.dry_run:
        for c in todo:
            print(f"  {c.id}")
        return 0
    results.mkdir(parents=True, exist_ok=True)
    if cfg.get("baseline") and not a.only:
        for var in cfg.get("variants") or [None]:
            name = f"baseline_{var['name']}" if var else "baseline"
            if not (results / f"{name}.json").exists():
                print(f"{name} ...", flush=True)
                run_baseline(cfg, results, var)
    for c in todo:
        if c.provider["model"] in unverified:
            print(f"cell {c.id}: skipped, unverified price for {c.provider['model']}")
            continue
        print(f"cell {c.id} ...", flush=True)
        out = run_cell(cfg, c, results)
        print(f"  {out['agent']['stop_reason']}, ${out['agent']['spent_usd']}, eval {out['eval']}, "
              f"grade {out['grade']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
