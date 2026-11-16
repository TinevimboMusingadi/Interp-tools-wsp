"""Export agent runs as static replay files for the blog's embedded WSP Studio.

    python scripts/export_replays.py runs/p2_loop_fix --out blog/studio/replays

Writes ``index.json`` (the run list plus the model layout) and one ``<exp>__<cell>.json`` per run, in the
same shape the Studio server returns, so the static build of the Studio replays them without a backend.
Long tool outputs and the system prompt are trimmed to keep the files small; home-directory paths are
replaced with ``~``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from fastapi.testclient import TestClient

from wsp.studio.server import create_app

OUT_CAP = 3000  # characters kept per tool output
TEXT_CAP = 4000
HOME = re.compile(r"/home/[A-Za-z0-9_.-]+|[A-Za-z]:\\\\?Users\\\\?[A-Za-z0-9_.-]+")


def _clean(x):
    if isinstance(x, str):
        return HOME.sub("~", x)
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    return x


def _trim(step: dict) -> dict:
    for k, cap in (("output", OUT_CAP), ("text", TEXT_CAP), ("system", 1500)):
        v = step.get(k)
        if isinstance(v, str) and len(v) > cap:
            step[k] = v[:cap] + f"\n… [{len(v) - cap:,} more characters in the full transcript]"
    return step


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder holding <exp>/<cell>/transcript.jsonl (or one experiment)")
    ap.add_argument("--out", default="blog/studio/replays")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    runs_root = root.parent if (root / next(iter(root.iterdir())).name / "transcript.jsonl").exists() else root
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    c = TestClient(create_app(runs_root))
    runs = [r for r in c.get("/api/runs").json() if not r["live"] and (runs_root / r["id"]).is_relative_to(root)]
    index = {"arch": c.get("/api/arch").json(), "runs": []}
    for r in runs:
        d = c.get(f"/api/run/{r['id']}").json()
        d["steps"] = [_trim(s) for s in d["steps"]]
        fname = r["id"].replace("/", "__") + ".json"
        text = json.dumps(_clean(d), separators=(",", ":"), ensure_ascii=False)
        (out / fname).write_text(text, encoding="utf-8")
        index["runs"].append({**_clean(r), "file": fname})
        print(f"{r['id']}: {len(d['steps'])} steps, {len(text) / 1e3:.0f} kB")
    (out / "index.json").write_text(json.dumps(index, separators=(",", ":")), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
