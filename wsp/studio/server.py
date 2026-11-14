"""FastAPI app: run replay, live tailing over Server-Sent Events, and (with a model) the hands-on studio."""

from __future__ import annotations

import asyncio
import functools
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from wsp.studio.arch import DEFAULT_MODEL, load_arch
from wsp.studio.events import StepParser, final_diff, parse_run, read_rows

WEB_DIST = Path(__file__).resolve().parents[2] / "studio" / "web" / "dist"
EVAL_KEYS = ("math500_acc", "math500_hit_limit", "gsm8k_acc", "ifeval_subset_acc")


@functools.lru_cache(maxsize=4)
def token_decoder(model_id: str):
    """Token id -> text, from the cached tokenizer.json (fast; no transformers import)."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from tokenizers import Tokenizer

        p = try_to_load_from_cache(model_id, "tokenizer.json")
        if not isinstance(p, str):
            return None
        tok = Tokenizer.from_file(p)
        return lambda ids: [tok.decode([int(i)], skip_special_tokens=False) for i in ids]
    except Exception:  # noqa: BLE001
        return None


def _eval_for(cell_dir: Path) -> tuple[dict | None, dict | None]:
    """The cell's held-out eval and its experiment baseline, found in ``experiments/<exp>/results`` next
    to the runs tree (works for runs copied back from a VM, which carry their own experiments/)."""
    exp, cell = cell_dir.parent.name, cell_dir.name
    for anc in cell_dir.parents:
        res = anc / "experiments" / exp / "results"
        f = res / f"{cell}.json"
        if f.exists():
            ev = json.loads(f.read_text(encoding="utf-8")).get("eval")
            b = res / "baseline.json"
            base = json.loads(b.read_text(encoding="utf-8")).get("eval") if b.exists() else None
            return ev, base
    return None, None


def _short_eval(ev: dict | None) -> dict | None:
    return {k: ev[k] for k in EVAL_KEYS if ev and k in ev} or None


def create_app(runs_root: Path, studio: Any = None, ssh: dict | None = None) -> FastAPI:
    runs_root = runs_root.resolve()
    app = FastAPI(title="WSP Studio")

    def cell_dir(run_id: str) -> Path:
        d = (runs_root / run_id).resolve()
        if runs_root not in d.parents or not (d / "transcript.jsonl").exists():
            raise HTTPException(404, f"no run {run_id!r}")
        return d

    @app.get("/api/runs")
    def runs() -> list[dict]:
        out = []
        for tr in sorted(runs_root.rglob("transcript.jsonl")):
            d = tr.parent
            res = d / "result.json"
            r = json.loads(res.read_text(encoding="utf-8")) if res.exists() else {}
            ev, base = _eval_for(d)
            first = read_rows(tr)[:1]
            out.append({"id": d.relative_to(runs_root).as_posix(), "exp": d.parent.name, "cell": d.name,
                        "agent": first[0].get("model") if first else None, "live": not res.exists(),
                        "stop_reason": r.get("stop_reason"), "weights_changed": r.get("weights_changed"),
                        "spent_usd": r.get("spent_usd"), "eval": _short_eval(ev), "baseline": _short_eval(base),
                        "mtime": tr.stat().st_mtime})
        return out

    @app.get("/api/arch")
    def arch(model: str = DEFAULT_MODEL) -> dict:
        return load_arch(model)

    @app.get("/api/run/{run_id:path}")
    def run(run_id: str) -> dict:
        d = cell_dir(run_id)
        first = read_rows(d / "transcript.jsonl")[:1]
        target = DEFAULT_MODEL
        if first:
            target = StepParser().feed(first[0])[0].get("target") or DEFAULT_MODEL
        r = parse_run(d, token_decoder(target))
        ev, base = _eval_for(d)
        return {"id": run_id, "arch": load_arch(target), **r, "eval": ev, "baseline": base}

    @app.get("/api/live/{run_id:path}")
    async def live(run_id: str) -> StreamingResponse:
        """Server-Sent Events: every step of the run so far, then new ones as the transcript grows."""
        if ssh:
            return StreamingResponse(_ssh_tail(run_id, ssh), media_type="text/event-stream")
        d = cell_dir(run_id)
        return StreamingResponse(_file_tail(d), media_type="text/event-stream")

    async def _file_tail(d: Path):
        target = DEFAULT_MODEL
        parser: StepParser | None = None
        pos, buf = 0, ""
        idle = 0
        while True:
            f = d / "transcript.jsonl"
            with f.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(pos)
                chunk = fh.read()
                pos = fh.tell()
            buf += chunk
            lines = buf.split("\n")
            buf = lines.pop()
            for line in lines:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if parser is None:
                    parser = StepParser()
                    if row.get("kind") == "start":
                        target = StepParser().feed(row)[0].get("target") or DEFAULT_MODEL
                        parser.decode = token_decoder(target)
                        yield _sse("arch", load_arch(target))
                for s in parser.feed(row):
                    yield _sse("step", s)
                    if s["kind"] == "end":
                        diff = final_diff(d)
                        if diff:
                            yield _sse("final_diff", diff)
                        yield _sse("done", {})
                        return
            idle = 0 if chunk else idle + 1
            if idle and idle % 30 == 0:
                yield ": keepalive\n\n"
            await asyncio.sleep(0.5)

    async def _ssh_tail(run_id: str, cfg: dict):
        """Follow a transcript on a TPU VM: ``gcloud ... ssh --command 'tail -n +1 -f <file>'``."""
        path = f"{cfg['root'].rstrip('/')}/{run_id}/transcript.jsonl"
        cmd = ["gcloud", "compute", "tpus", "tpu-vm", "ssh", cfg["vm"], f"--zone={cfg['zone']}",
               f"--project={cfg['project']}", f"--command=tail -n +1 -F {shlex.quote(path)}"]
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        parser = StepParser()
        try:
            assert proc.stdout is not None
            async for raw in proc.stdout:
                try:
                    row = json.loads(raw.decode("utf-8", "replace"))
                except json.JSONDecodeError:
                    continue
                if row.get("kind") == "start":
                    target = StepParser().feed(row)[0].get("target") or DEFAULT_MODEL
                    parser.decode = token_decoder(target)
                    yield _sse("arch", load_arch(target))
                for s in parser.feed(row):
                    yield _sse("step", s)
                    if s["kind"] == "end":
                        yield _sse("done", {})
                        return
        finally:
            if proc.returncode is None:
                proc.kill()

    # -- studio (hands-on) -------------------------------------------------------------------
    @app.get("/api/studio/status")
    def studio_status() -> dict:
        if studio is None:
            return {"enabled": False}
        return {"enabled": True, **studio.status()}

    @app.post("/api/studio/{action}")
    async def studio_action(action: str, body: dict) -> dict:
        if studio is None:
            raise HTTPException(400, "start with --model to use the studio")
        return await asyncio.to_thread(studio.handle, action, body or {})

    # -- frontend ------------------------------------------------------------------------------
    if WEB_DIST.exists():
        app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

        @app.get("/{path:path}")
        def index(path: str) -> FileResponse:
            f = WEB_DIST / path
            return FileResponse(f if path and f.is_file() else WEB_DIST / "index.html")

    return app


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
