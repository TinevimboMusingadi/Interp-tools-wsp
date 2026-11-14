"""The editing environment: the target model, the agent's Python namespace, and the rules.

This runs **inside the sandbox worker process** (see ``sandbox.py``), never in the harness process,
so agent-written code cannot touch the harness, its API keys or its ledger.

The rules are enforced, not requested:

- **No gradients.** ``Tensor.backward``, ``torch.autograd.grad`` and ``torch.autograd.backward``
  raise. Weight-space programming means the agent writes the weights by reasoning, not by an
  optimizer. (Tools such as SAE training in condition C2 run gradients *inside a tool*, where the
  harness switches the guard off for that call only.)
- **No unbudgeted forward passes.** A forward pre-hook on every module refuses to run unless a
  budgeted tool opened a forward window, so ``model.hf_model(x)``, calling a single layer, or
  ``interp_engine.capture`` from agent code all raise instead of quietly spending compute.
- **Every edit is visible.** The weights are snapshotted at load, so ``edit_report`` can always say
  exactly what changed.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch

from wsp.edit import snapshot, weight_diff
from wsp.harness.providers import ToolSpec

MAX_OUTPUT_CHARS = 20_000


class RuleViolation(RuntimeError):
    pass


class BudgetSpent(RuntimeError):
    pass


@dataclass
class Tool:
    spec: ToolSpec
    fn: Callable[..., str]  # fn(env, **args) -> text shown to the agent


_REAL_BACKWARD = torch.Tensor.backward
_REAL_GRAD = torch.autograd.grad
_REAL_AUTOGRAD_BACKWARD = torch.autograd.backward


def _blocked(*_a: Any, **_k: Any) -> Any:
    raise RuleViolation("gradients are not allowed in weight-space programming: edit the weights directly")


def restore_gradients() -> None:
    """Undo the gradient block for this process. Only for code that shares a process with an Env and
    legitimately trains (tests, the organism builder); the sandbox worker never calls it."""
    torch.Tensor.backward = _REAL_BACKWARD  # type: ignore[method-assign]
    torch.autograd.grad = _REAL_GRAD  # type: ignore[assignment]
    torch.autograd.backward = _REAL_AUTOGRAD_BACKWARD  # type: ignore[assignment]


def load_target(loader: str, **kwargs: Any) -> Any:
    """``loader`` is ``"module:function"``; the function returns an object with ``.hf_model`` and
    ``.tokenizer`` (an interp-engine ``EagerModel`` satisfies this)."""
    mod, fn = loader.split(":")
    return getattr(importlib.import_module(mod), fn)(**kwargs)


def eager_loader(model_id: str, device: str = "cpu", dtype: str = "float32", attn: str = "eager",
                 state_dict: str | None = None) -> Any:
    """Load an HF checkpoint on interp-engine's eager backend. ``attn="eager"`` materialises attention
    probabilities (needed by the ``attention`` tool); ``state_dict`` loads edited or fine-tuned weights
    (a model organism) over the base checkpoint."""
    from interp_engine import load_model

    model = load_model(model_id, backend="eager", device=device, dtype=dtype, attn_implementation=attn)
    if state_dict:
        sd = torch.load(state_dict, map_location="cpu")
        missing, unexpected = model.hf_model.load_state_dict(sd, strict=False)
        if unexpected:
            raise ValueError(f"state_dict has keys the model lacks: {unexpected[:5]}")
    return model


class Env:
    """One editing environment. ``model`` may be a single model or a dict of named models (e.g.
    ``{"A": ..., "B": ...}`` for a comparison task); every tool then takes an optional ``model``
    argument naming which one to act on (default: the first). All models share one budget."""

    def __init__(self, model: Any, *, forward_budget: int, tools: dict[str, Tool]):
        self.models: dict[str, Any] = dict(model) if isinstance(model, dict) else {"model": model}
        self.default = next(iter(self.models))
        self.model = self.models[self.default]
        self.forward_budget = forward_budget
        self.forward_used = 0
        self.tools = tools if len(self.models) == 1 else _with_model_arg(tools, list(self.models))
        self._forward_ok = 0
        self.references = {k: snapshot(m.hf_model.named_parameters()) for k, m in self.models.items()}
        self.reference = self.references[self.default]
        self.namespace: dict[str, Any] = {"model": self.model, "models": self.models, "torch": torch,
                                          "__name__": "__agent__", "__builtins__": _guarded_builtins()}
        self._params = [(k, n, p) for k, m in self.models.items() for n, p in m.hf_model.named_parameters()]
        self._sums = self._checksums()
        self.last_delta: dict = {}
        self._install_guards()

    # -- rules -------------------------------------------------------------------------------
    def _install_guards(self) -> None:
        self._install_grad_blocks()

        def guard(_module: torch.nn.Module, _args: Any) -> None:
            if not self._forward_ok:
                raise RuleViolation(
                    "running the network (or any of its layers) costs a forward pass: use a tool"
                )

        for model in self.models.values():
            hf = model.hf_model
            for h in getattr(hf, "_wsp_guard_handles", []):  # one Env owns a model at a time
                h.remove()
            hf._wsp_guard_handles = [m.register_forward_pre_hook(guard) for m in hf.modules()]

    @contextlib.contextmanager
    def forward_window(self, passes: int = 1) -> Iterator[None]:
        """Charge ``passes`` forward passes and allow the network to run inside the block."""
        if self.forward_used + passes > self.forward_budget:
            raise BudgetSpent(f"forward-pass budget spent ({self.forward_used}/{self.forward_budget})")
        self.forward_used += passes
        self._forward_ok += 1
        try:
            yield
        finally:
            self._forward_ok -= 1

    @contextlib.contextmanager
    def uncharged(self) -> Iterator[None]:
        """Harness-internal: let a tool run a module that is part of an already-charged read (e.g. the
        final norm + unembed that decode a logit-lens row). Never exposed to agent code."""
        self._forward_ok += 1
        try:
            yield
        finally:
            self._forward_ok -= 1

    @contextlib.contextmanager
    def gradients_allowed(self) -> Iterator[None]:
        """For harness-owned tools only (e.g. training an SAE in C2); never exposed to agent code."""
        torch.Tensor.backward = _REAL_BACKWARD  # type: ignore[method-assign]
        torch.autograd.grad = _REAL_GRAD  # type: ignore[assignment]
        torch.autograd.backward = _REAL_AUTOGRAD_BACKWARD  # type: ignore[assignment]
        try:
            yield
        finally:
            self._install_grad_blocks()

    def _install_grad_blocks(self) -> None:
        torch.Tensor.backward = _blocked  # type: ignore[method-assign]
        torch.autograd.grad = _blocked  # type: ignore[assignment]
        torch.autograd.backward = _blocked  # type: ignore[assignment]

    # -- agent actions ----------------------------------------------------------------------
    def run_python(self, code: str) -> str:
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                exec(compile(code, "<agent>", "exec"), self.namespace)  # noqa: S102 - the point of a sandbox
        except Exception:  # noqa: BLE001 - every failure is shown to the agent as text
            buf.write(traceback.format_exc(limit=4))
        return _clip(buf.getvalue() or "(no output)")

    def call(self, name: str, args: dict) -> str:
        if name not in self.tools:
            return f"unknown tool {name!r}; available: {sorted(self.tools)}"
        args = dict(args)
        which = args.pop("model", None) if len(self.models) > 1 else None
        if which is not None and which not in self.models:
            return f"unknown model {which!r}; available: {sorted(self.models)}"
        self.model = self.models[which or self.default]
        try:
            return _clip(self.tools[name].fn(self, **args))
        except (RuleViolation, BudgetSpent) as e:
            return f"refused: {e}"
        except Exception:  # noqa: BLE001
            return _clip(traceback.format_exc(limit=4))
        finally:
            self.model = self.models[self.default]
            if name == "python":
                self.last_delta = self._weight_delta()

    # -- per-call edit record (for the transcript and the replay viewer) ----------------------------
    @torch.no_grad()
    def _checksums(self) -> torch.Tensor:
        """One number per tensor, computed on device and fetched in one transfer: changes when the
        tensor's values change (sum + sum of squares; a pure permutation would slip through)."""
        if not self._params:
            return torch.zeros(0)
        vals = [p.detach().float().sum() + 0.5 * p.detach().float().square().sum() for _, _, p in self._params]
        return torch.stack(vals).cpu()

    @torch.no_grad()
    def _weight_delta(self, cap: int = 4096) -> dict:
        """Tensors this call changed, each with the rows / columns that now differ from the base
        weights. Never raises: a failure here must not break the agent's run."""
        try:
            new = self._checksums()
            changed = (new != self._sums).nonzero().flatten().tolist()
            self._sums = new
            out = {}
            for i in changed:
                k, name, p = self._params[i]
                ch = p.detach().cpu() != self.references[k][name].to(p.dtype)
                d: dict[str, Any] = {"n": int(ch.sum()), "shape": list(ch.shape)}
                if ch.ndim == 2:
                    d["rows"] = ch.any(1).nonzero().flatten()[:cap].tolist()
                    d["cols"] = ch.any(0).nonzero().flatten()[:cap].tolist()
                elif ch.ndim == 1:
                    d["rows"] = ch.nonzero().flatten()[:cap].tolist()
                out[name if len(self.models) == 1 else f"{k}:{name}"] = d
            return out
        except Exception:  # noqa: BLE001
            return {}

    def _diffs(self) -> dict:
        return {k: weight_diff(self.references[k], m.hf_model.named_parameters()) for k, m in self.models.items()}

    def edit_report(self) -> str:
        diffs = self._diffs()
        if len(diffs) == 1:
            return next(iter(diffs.values())).summary()
        return "\n".join(f"[{k}] {d.summary()}" for k, d in diffs.items())

    def status(self) -> dict:
        diffs = self._diffs()
        return {"forward_used": self.forward_used, "forward_budget": self.forward_budget,
                "weights_changed": sum(d.n_changed for d in diffs.values()),
                "weights_total": sum(d.n_total for d in diffs.values()),
                "per_model_changed": {k: d.n_changed for k, d in diffs.items()}}


DENY_PATH_PARTS = (".keys", "organisms", "ledger.csv", "transcript.jsonl")


def _guarded_builtins() -> dict:
    """Builtins for agent code with ``open`` refusing answer keys, organisms and harness logs.

    A speed bump, not a jail: determined code could still reach the filesystem through ``os``. The
    real defence is that keys never exist on the machine running the sandbox in final runs, and every
    transcript is audited for file access (``wsp.report.audit``)."""
    import builtins

    real_open = builtins.open

    def guarded_open(file: Any, *a: Any, **k: Any) -> Any:
        if any(part in str(file) for part in DENY_PATH_PARTS):
            raise RuleViolation(f"reading {file!r} is not allowed")
        return real_open(file, *a, **k)

    b = dict(vars(builtins))
    b["open"] = guarded_open
    return b


def _with_model_arg(tools: dict[str, Tool], names: list[str]) -> dict[str, Tool]:
    """Add an optional ``model`` argument (one of ``names``) to every tool except submit."""
    import copy

    out = {}
    for k, t in tools.items():
        spec = copy.deepcopy(t.spec)
        if k != "submit":
            spec.parameters.setdefault("properties", {})["model"] = {
                "type": "string", "enum": names, "description": f"which model to act on (default {names[0]})"}
            if k == "python":
                spec.description += f" Named models are in `models` ({', '.join(names)}); `model` is {names[0]}."
        out[k] = Tool(spec, t.fn)
    return out


def _clip(s: str) -> str:
    if len(s) <= MAX_OUTPUT_CHARS:
        return s
    half = MAX_OUTPUT_CHARS // 2
    return f"{s[:half]}\n... [{len(s) - MAX_OUTPUT_CHARS} chars clipped] ...\n{s[-half:]}"
