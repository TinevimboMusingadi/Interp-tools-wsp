"""Phase 4: recursive tool improvement.

    python -m wsp.rsi experiments/p4_rsi/config.yaml [--dry-run]

Round k runs condition C2 on that round's tasks with tool library L_k. Every tool and SAE the round's
runs save is merged into L_{k+1}. Tasks rotate between rounds, so a tool only keeps paying off if it
generalises. After the last round, each library snapshot L_0..L_K is evaluated on held-out tasks with
nothing they save merged back, which isolates "better tools" from "more practice".
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from wsp.harness.agent import RUNS_DIR
from wsp.run import Cell, load_config, run_cell


def merge_library(src_dirs: list[Path], dst: Path) -> list[str]:
    """Union of saved tools/SAEs; on a name clash the later file gets a numeric suffix."""
    added = []
    for src in src_dirs:
        for kind in ("tools", "saes"):
            for f in sorted((src / kind).glob("*")) if (src / kind).exists() else []:
                (dst / kind).mkdir(parents=True, exist_ok=True)
                target, i = dst / kind / f.name, 1
                while target.exists():
                    if target.read_bytes() == f.read_bytes():
                        break
                    target = dst / kind / f"{f.stem}_{i}{f.suffix}"
                    i += 1
                else:
                    shutil.copy2(f, target)
                    added.append(f"{kind}/{target.name}")
    return added


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    base = cfg["_dir"]
    libs = base / "libraries"
    (libs / "L0").mkdir(parents=True, exist_ok=True)
    log = []
    for k, round_tasks in enumerate(cfg["rounds"]):
        print(f"round {k}: tasks {[t['task'] for t in round_tasks]} with L{k}")
        if a.dry_run:
            continue
        run_dirs = []
        for t in round_tasks:
            for prov in cfg["providers"]:
                for seed in cfg["seeds"]:
                    sub = {**cfg, "id": f"{cfg['id']}/round{k}", "task": t["task"],
                           "task_kwargs": t.get("task_kwargs", {}), "library_dir": str(libs / f"L{k}"),
                           "_dir": base}
                    res = base / "results" / f"round{k}"
                    res.mkdir(parents=True, exist_ok=True)
                    short = prov.get("short") or prov["model"]
                    cell = Cell(f"{t['task']}_{short}_s{seed}", "c2", t["budget"], prov, seed)
                    if not (res / f"{cell.id}.json").exists():
                        run_cell(sub, cell, res)
                    run_dirs.append(RUNS_DIR / sub["id"] / cell.id / "library")
        nxt = libs / f"L{k + 1}"
        shutil.copytree(libs / f"L{k}", nxt, dirs_exist_ok=True)
        added = merge_library(run_dirs, nxt)
        log.append({"round": k, "added": added})
        print(f"  L{k + 1}: +{len(added)} items")
    if not a.dry_run:
        (base / "results" / "library_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    # held-out evaluation of every library snapshot, saving disabled
    for k in range(len(cfg["rounds"]) + 1):
        for t in cfg["heldout"]:
            print(f"held-out {t['task']} with frozen L{k}")
            if a.dry_run:
                continue
            for prov in cfg["providers"]:
                for seed in cfg["seeds"]:
                    sub = {**cfg, "id": f"{cfg['id']}/heldout_L{k}", "task": t["task"],
                           "task_kwargs": {**t.get("task_kwargs", {})}, "library_dir": str(libs / f"L{k}"),
                           "_dir": base}
                    res = base / "results" / f"heldout_L{k}"
                    res.mkdir(parents=True, exist_ok=True)
                    short = prov.get("short") or prov["model"]
                    cell = Cell(f"{t['task']}_{short}_s{seed}", "c2", t["budget"], prov, seed)
                    if not (res / f"{cell.id}.json").exists():
                        run_cell(sub, cell, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
