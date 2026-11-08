"""Run the editing environment in a separate process.

The harness (API keys, ledger, transcripts) lives in the parent; the model and the agent's code live
in a child process started with ``spawn``. A call that runs past its timeout kills the child, so a
runaway loop in agent code cannot hang the run. The child's environment has the API-key variables
removed before anything agent-written executes.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any

SECRET_PREFIXES = ("ANTHROPIC", "GEMINI", "GOOGLE_API", "WSP_GCP", "HF_TOKEN", "OPENAI")


class SandboxDied(RuntimeError):
    pass


def _worker(conn: Any, loader: str, loader_kwargs: dict, toolset: str, forward_budget: int,
            extras: dict) -> None:
    for k in list(os.environ):
        if k.startswith(SECRET_PREFIXES):
            del os.environ[k]
    from wsp.harness.env import Env, load_target
    from wsp.tools import TOOLSETS
    from wsp.tools.build import load_library
    from wsp.tools.sae import load_sae

    try:
        if "models" in loader_kwargs:  # several named models, e.g. {"A": {...}, "B": {...}}
            model = {k: load_target(loader, **kw) for k, kw in loader_kwargs["models"].items()}
        else:
            model = load_target(loader, **loader_kwargs)
        tools = {k: v for k, v in TOOLSETS[toolset]().items() if k not in extras.get("drop_tools", [])}
        env = Env(model, forward_budget=forward_budget, tools=tools)
        env.namespace["forbid_points"] = extras.get("forbid_points", [])
        load_library(env, extras.get("library_dir") if toolset == "c2" else None)
        env.namespace["gen_cap"] = extras.get("gen_cap", 1024)
        dev = str(env.model.hf_model.device)
        for s in extras.get("saes", []) if toolset in ("c1_sae", "c2") else []:
            env.namespace.setdefault("saes", {})[s["name"]] = load_sae(s["path"], s.get("point"), device=dev)
        for line in extras.get("setup_code", []):  # task-specific objects, e.g. a second model to compare
            out = env.run_python(line)
            if "Traceback" in out:
                raise RuntimeError(f"setup code failed: {out}")
        conn.send(("ok", [t.spec for t in env.tools.values()]))
    except Exception as e:  # noqa: BLE001
        conn.send(("error", repr(e)))
        return
    while True:
        msg = conn.recv()
        op = msg[0]
        if op == "close":
            return
        if op == "call":
            conn.send(("ok", env.call(msg[1], msg[2])))
        elif op == "status":
            st = env.status()
            st["submitted"] = env.namespace.get("__submitted__")
            st["answer"] = env.namespace.get("__answer__")
            conn.send(("ok", st))
        elif op == "eval":
            # After the agent is done: evaluation runs the network freely (not on the agent's budget)
            # and with held-out data the agent never saw. The agent cannot act after this point.
            import importlib

            env._forward_ok += 1
            mod, fn = msg[1].split(":")
            try:
                kw = dict(msg[2])
                target = env.models[kw.pop("model", None) or env.default]
                conn.send(("ok", getattr(importlib.import_module(mod), fn)(target, **kw)))
            except Exception as e:  # noqa: BLE001
                conn.send(("error", repr(e)))
        elif op == "save":
            import torch

            target = env.models[msg[2] or env.default]
            torch.save({k: v.cpu() for k, v in target.hf_model.state_dict().items()}, msg[1])
            conn.send(("ok", msg[1]))
        else:
            conn.send(("error", f"unknown op {op!r}"))


class Sandbox:
    def __init__(self, *, loader: str, loader_kwargs: dict, toolset: str, forward_budget: int,
                 extras: dict | None = None, load_timeout: float = 1800, call_timeout: float = 900):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.call_timeout = call_timeout
        self.proc = ctx.Process(target=_worker, daemon=True,
                                args=(child, loader, loader_kwargs, toolset, forward_budget, extras or {}))
        self.proc.start()
        status, payload = self._recv(load_timeout)
        if status != "ok":
            self.close()
            raise SandboxDied(f"sandbox failed to start: {payload}")
        self.tool_specs = payload

    def _recv(self, timeout: float) -> tuple[str, Any]:
        if not self.conn.poll(timeout):
            self.proc.kill()
            raise SandboxDied(f"sandbox call exceeded {timeout:.0f}s and was killed")
        try:
            return self.conn.recv()
        except EOFError as e:
            raise SandboxDied("sandbox process exited") from e

    def call(self, name: str, args: dict) -> str:
        self.conn.send(("call", name, args))
        status, payload = self._recv(self.call_timeout)
        return payload if status == "ok" else f"error: {payload}"

    def status(self) -> dict:
        self.conn.send(("status",))
        return self._recv(60)[1]

    def evaluate(self, fn: str, timeout: float = 6 * 3600, **kwargs: Any) -> dict:
        """Run ``"module:function"(model, **kwargs)`` on the current (edited) weights. Ends agent use."""
        self.conn.send(("eval", fn, kwargs))
        status, payload = self._recv(timeout)
        if status != "ok":
            raise RuntimeError(f"evaluation failed: {payload}")
        return payload

    def save_weights(self, path: str, model: str | None = None) -> str:
        self.conn.send(("save", path, model))
        return self._recv(600)[1]

    def close(self) -> None:
        if self.proc.is_alive():
            try:
                self.conn.send(("close",))
                self.proc.join(10)
            finally:
                if self.proc.is_alive():
                    self.proc.kill()

    def __enter__(self) -> Sandbox:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
