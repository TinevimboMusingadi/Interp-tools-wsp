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


def load_target(loader: str, **kwargs: Any) -> Any:
    """``loader`` is ``"module:function"``; the function returns an object with ``.hf_model`` and
    ``.tokenizer`` (an interp-engine ``EagerModel`` satisfies this)."""
    mod, fn = loader.split(":")
    return getattr(importlib.import_module(mod), fn)(**kwargs)


def eager_loader(model_id: str, device: str = "cpu", dtype: str = "float32") -> Any:
    from interp_engine import load_model

    return load_model(model_id, backend="eager", device=device, dtype=dtype)


class Env:
    def __init__(self, model: Any, *, forward_budget: int, tools: dict[str, Tool]):
        self.model = model
        self.forward_budget = forward_budget
        self.forward_used = 0
        self.tools = tools
        self._forward_ok = 0
        self.reference = snapshot(model.hf_model.named_parameters())
        self.namespace: dict[str, Any] = {"model": model, "torch": torch, "__name__": "__agent__"}
        self._install_guards()

    # -- rules -------------------------------------------------------------------------------
    def _install_guards(self) -> None:
        self._install_grad_blocks()

        def guard(_module: torch.nn.Module, _args: Any) -> None:
            if not self._forward_ok:
                raise RuleViolation(
                    "running the network (or any of its layers) costs a forward pass: use a tool"
                )

        for m in self.model.hf_model.modules():
            m.register_forward_pre_hook(guard)

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
        try:
            return _clip(self.tools[name].fn(self, **args))
        except (RuleViolation, BudgetSpent) as e:
            return f"refused: {e}"
        except Exception:  # noqa: BLE001
            return _clip(traceback.format_exc(limit=4))

    def edit_report(self) -> str:
        return weight_diff(self.reference, self.model.hf_model.named_parameters()).summary()

    def status(self) -> dict:
        diff = weight_diff(self.reference, self.model.hf_model.named_parameters())
        return {"forward_used": self.forward_used, "forward_budget": self.forward_budget,
                "weights_changed": diff.n_changed, "weights_total": diff.n_total}


def _clip(s: str) -> str:
    if len(s) <= MAX_OUTPUT_CHARS:
        return s
    half = MAX_OUTPUT_CHARS // 2
    return f"{s[:half]}\n... [{len(s) - MAX_OUTPUT_CHARS} chars clipped] ...\n{s[-half:]}"
