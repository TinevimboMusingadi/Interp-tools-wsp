"""Verification of an edit: what changed in the weights, and what changed in behaviour.

``weight_diff`` needs only two state dicts. ``kl_on_text`` needs a callable that returns logits, so
it works with an interp-engine ``EagerModel`` (``model.forward``-style) or a plain HF model.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import torch


@dataclass
class DiffReport:
    n_changed: int
    n_total: int
    per_tensor: dict[str, int]

    @property
    def fraction(self) -> float:
        return self.n_changed / max(self.n_total, 1)

    def summary(self, top: int = 10) -> str:
        lines = [f"{self.n_changed:,} of {self.n_total:,} weights changed ({100 * self.fraction:.4f}%)"]
        for name, n in sorted(self.per_tensor.items(), key=lambda kv: -kv[1])[:top]:
            lines.append(f"  {name}: {n:,}")
        return "\n".join(lines)


def snapshot(params: Iterable[tuple[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """CPU copy of named parameters, the reference ``weight_diff`` compares against."""
    return {name: p.detach().to("cpu", copy=True) for name, p in params}


def weight_diff(
    before: Mapping[str, torch.Tensor], after: Iterable[tuple[str, torch.Tensor]]
) -> DiffReport:
    per: dict[str, int] = {}
    changed = total = 0
    for name, p in after:
        ref = before[name]
        n = int((p.detach().to("cpu") != ref).sum().item())
        total += ref.numel()
        if n:
            per[name] = n
            changed += n
    return DiffReport(n_changed=changed, n_total=total, per_tensor=per)


@torch.no_grad()
def kl_on_text(
    logits_fn_before: Callable[[torch.Tensor], torch.Tensor],
    logits_fn_after: Callable[[torch.Tensor], torch.Tensor],
    token_batches: Iterable[torch.Tensor],
) -> float:
    """Mean per-token KL(before || after) over held-out token batches: the side-effect cost."""
    total, count = 0.0, 0
    for toks in token_batches:
        p = torch.log_softmax(logits_fn_before(toks).float(), dim=-1)
        q = torch.log_softmax(logits_fn_after(toks).float(), dim=-1)
        kl = (p.exp() * (p - q)).sum(-1)
        total += kl.sum().item()
        count += kl.numel()
    return total / max(count, 1)
