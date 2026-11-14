"""python -m wsp.studio --runs runs [--model Qwen/Qwen3.5-0.8B --device cpu] [--port 8800]

Replay needs no model. ``--model`` also loads the model for the hands-on studio (CPU load of
Qwen3.5-0.8B takes a while). ``--ssh-vm wsp-p2 --ssh-zone us-east5-b`` makes live view follow transcripts
on a TPU VM instead of local files.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="wsp.studio")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--model", default=None, help="load this model for the hands-on studio")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--ssh-vm", default=None)
    ap.add_argument("--ssh-zone", default="us-east5-b")
    ap.add_argument("--ssh-project", default="tpu-builder1")
    ap.add_argument("--ssh-root", default="/home/Tinevimbo/work/Interp-tools-wsp/runs")
    a = ap.parse_args(argv)

    import uvicorn

    from wsp.studio.server import create_app

    studio = None
    if a.model:
        import os

        from wsp.studio.arch import _config_path

        if _config_path(a.model) is not None:  # cached: load offline, so a flaky hub check can't stop it
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from wsp.studio.live import Studio

        print(f"loading {a.model} on {a.device} for the studio ...", flush=True)
        studio = Studio(a.model, device=a.device, dtype=a.dtype)
        print(f"loaded in {studio.load_s}s", flush=True)
    ssh = ({"vm": a.ssh_vm, "zone": a.ssh_zone, "project": a.ssh_project, "root": a.ssh_root}
           if a.ssh_vm else None)
    app = create_app(Path(a.runs), studio=studio, ssh=ssh)
    print(f"WSP Studio on http://{a.host}:{a.port}", flush=True)
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
