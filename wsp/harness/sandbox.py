"""Run the editing environment in a separate process.

The harness (API keys, ledger, transcripts) lives in the parent; the model and the agent's code live
in a child process started with ``spawn``. A call that runs past its timeout kills the child, so a
runaway loop in agent code cannot hang the run. The child's environment has the API-key variables
removed before anything agent-written executes, and every agent call runs inside ``Jail`` (an audit
hook refusing file access outside the workspace, processes and network; see ``wsp.harness.jail``).
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
from typing import Any

SECRET_PREFIXES = ("ANTHROPIC", "GEMINI", "GOOGLE_API", "WSP_GCP", "HF_TOKEN", "OPENAI")


class SandboxDied(RuntimeError):
    pass


# Modules agent code may reach in memory. Everything else under ``wsp`` (task registry, eval suites,
# runner, agent loop) was imported into the child only because ``spawn`` re-imports the parent's main
# module; the first p1 run read task configs that way. Re-importing them is refused by the jail.
KEEP_MODULES = ("wsp.harness.env", "wsp.harness.jail", "wsp.harness.sandbox", "wsp.harness.providers",
                "wsp.tools", "wsp.edit", "wsp.eval.generate")


def _forget_harness_modules() -> None:
    import sys
    import types

    for name in sorted(sys.modules, reverse=True):
        drop = name == "__mp_main__" or (name.startswith("wsp.") and not name.startswith(KEEP_MODULES)
                                         and not any(k.startswith(name + ".") for k in KEEP_MODULES))
        if not drop:
            continue
        del sys.modules[name]
        parent, _, child = name.rpartition(".")
        if parent in sys.modules and getattr(sys.modules[parent], child, None) is not None:
            delattr(sys.modules[parent], child)
    for pkg in ("wsp", "wsp.eval", "wsp.harness"):  # keep packages, drop their links to dropped children
        mod = sys.modules.get(pkg)
        for attr, val in list(vars(mod).items()) if mod else []:
            if isinstance(val, types.ModuleType) and val.__name__ not in sys.modules:
                delattr(mod, attr)
    sys.modules["__main__"] = types.ModuleType("__main__")
    import gc

    gc.collect()  # dropped modules live on in reference cycles; agents found them with gc.get_objects()


# Files whose code agent code must not be able to reach in memory (task configs, eval, runner).
HIDDEN_FILES = ("wsp/tasks/registry.py", "wsp/run.py", "wsp/eval/suites.py",
                "wsp/harness/agent.py", "wsp/harness/ledger.py")


def _reachable_hidden_code() -> list[str]:
    """Functions from HIDDEN_FILES still alive in this process (checked after the purge)."""
    import gc
    import types

    found = set()
    for o in gc.get_objects():
        if isinstance(o, types.FunctionType):
            fn = o.__code__.co_filename.replace("\\", "/")
            if fn.endswith(HIDDEN_FILES):
                found.add(f"{fn.rsplit('/wsp/', 1)[-1]}:{o.__name__}")
    return sorted(found)


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
        # Everything agent-triggered runs inside the jail; import lazily-loaded modules first so imports
        # inside tools never need to read outside it.
        import importlib

        from wsp.harness.jail import Jail

        for name in ("interp_engine", "safetensors.torch", "wsp.eval.generate", "wsp.tools.base",
                     "wsp.tools.interp", "wsp.tools.sae", "wsp.tools.build", "wsp.edit"):
            with contextlib.suppress(ImportError):
                importlib.import_module(name)
        _forget_harness_modules()
        leaked = _reachable_hidden_code()
        if leaked:  # p1 (Linux): agents reached the task registry this way after the purge
            raise RuntimeError(f"harness code still reachable in the sandbox: {leaked[:8]}")
        jail = Jail(library_dir=extras.get("library_dir") if toolset == "c2" else None)
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
            with jail.armed():
                out = env.call(msg[1], msg[2])
            conn.send(("ok", out))
        elif op == "delta":  # what the last python call changed, for the transcript
            conn.send(("ok", env.last_delta))
            env.last_delta = {}
        elif op == "passes":  # cheap, every turn: status() diffs every weight
            conn.send(("ok", {"forward_used": env.forward_used, "forward_budget": env.forward_budget}))
        elif op == "status":
            st = env.status()
            st["submitted"] = env.namespace.get("__submitted__")
            st["answer"] = env.namespace.get("__answer__")
            st["jail_violations"] = list(jail.violations)
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


@contextlib.contextmanager
def _main_hidden_from_spawn():
    """``spawn`` re-runs the parent's main module in the child (``python -m wsp.run`` -> the runner and
    the task registry). With no ``__spec__`` name and no ``__file__`` it imports nothing; ``_worker``
    is pickled by reference to this module, so the child needs nothing from main."""
    import sys

    main = sys.modules.get("__main__")
    saved = {k: main.__dict__[k] for k in ("__spec__", "__file__") if main is not None and k in main.__dict__}
    try:
        if main is not None:
            main.__dict__.pop("__file__", None)
            main.__spec__ = None
        yield
    finally:
        if main is not None:
            main.__dict__.update(saved)
            if "__spec__" not in saved:
                main.__dict__.pop("__spec__", None)


class Sandbox:
    def __init__(self, *, loader: str, loader_kwargs: dict, toolset: str, forward_budget: int,
                 extras: dict | None = None, load_timeout: float = 1800, call_timeout: float = 900):
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.call_timeout = call_timeout
        self.proc = ctx.Process(target=_worker, daemon=True,
                                args=(child, loader, loader_kwargs, toolset, forward_budget, extras or {}))
        with _main_hidden_from_spawn():
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

    def passes(self) -> dict:
        """``{forward_used, forward_budget}`` without the weight diff ``status()`` computes."""
        self.conn.send(("passes",))
        status, payload = self._recv(60)
        return payload if status == "ok" else {}

    def last_delta(self) -> dict:
        """``{tensor: {n, shape, rows, cols}}`` changed by the last python call (empty if none)."""
        self.conn.send(("delta",))
        status, payload = self._recv(120)
        return payload if status == "ok" else {}

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
