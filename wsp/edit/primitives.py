"""Weight-edit primitives: the small, auditable vocabulary an agent edits weights with.

Every function edits a weight tensor **in place** under ``torch.no_grad()`` and returns an
:class:`EditRecord`, so a run's edits can be replayed, diffed and reported. Conventions follow
``torch.nn.Linear``: ``weight`` has shape ``[out_features, in_features]`` and computes ``x @ W.T``.

"Writes to the residual stream" means the output side of a projection whose ``out_features`` is
``d_model`` (an MLP down projection, an attention output projection). Removing a direction ``v`` from
what a module writes is ``W <- (I - v v^T) W``, the weight-space form of a directional ablation.
"""
# Docstrings in modules the sandboxed agent can reach must not hint at a task's solution or cite the
# source paper's results: agents read them (p1, Nov 2026). Background belongs in docs/ and the blog.

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class EditRecord:
    """What one edit changed. ``n_changed`` counts elements whose value actually moved."""

    op: str
    target: str
    n_changed: int
    n_total: int
    params: dict = field(default_factory=dict)

    @property
    def fraction(self) -> float:
        return self.n_changed / max(self.n_total, 1)


def _unit(v: torch.Tensor) -> torch.Tensor:
    v = v.detach().flatten().to(torch.float32)
    n = v.norm()
    if not torch.isfinite(n) or n == 0:
        raise ValueError("direction must be finite and non-zero")
    return v / n


def _commit(weight: torch.Tensor, new: torch.Tensor, op: str, target: str, params: dict) -> EditRecord:
    new = new.to(weight.dtype)
    changed = int((new != weight).sum().item())
    weight.copy_(new)
    return EditRecord(op=op, target=target, n_changed=changed, n_total=weight.numel(), params=params)


@torch.no_grad()
def orthogonalize_out(
    weight: torch.Tensor,
    direction: torch.Tensor,
    *,
    columns: torch.Tensor | list[int] | None = None,
    target: str = "",
) -> EditRecord:
    """Remove ``direction`` (length ``out_features``) from what this projection writes.

    ``columns`` restricts the edit to those input units (e.g. 64 MLP neurons), which is how a
    targeted edit stays a small fraction of the weights. ``None`` edits every column.
    """
    if direction.numel() != weight.shape[0]:
        raise ValueError(f"direction has {direction.numel()} entries, weight writes {weight.shape[0]}")
    v = _unit(direction).to(weight.device)
    w = weight.to(torch.float32)
    cols = slice(None) if columns is None else torch.as_tensor(columns, device=weight.device)
    sub = w[:, cols]
    w = w.clone()
    w[:, cols] = sub - torch.outer(v, v @ sub)
    n_cols = weight.shape[1] if columns is None else len(cols)
    return _commit(weight, w, "orthogonalize_out", target, {"n_columns": n_cols})


@torch.no_grad()
def orthogonalize_in(weight: torch.Tensor, direction: torch.Tensor, *, target: str = "") -> EditRecord:
    """Make this projection blind to ``direction`` (length ``in_features``): ``W <- W (I - v v^T)``."""
    if direction.numel() != weight.shape[1]:
        raise ValueError(f"direction has {direction.numel()} entries, weight reads {weight.shape[1]}")
    v = _unit(direction).to(weight.device)
    w = weight.to(torch.float32)
    w = w - torch.outer(w @ v, v)
    return _commit(weight, w, "orthogonalize_in", target, {})


@torch.no_grad()
def rank1(
    weight: torch.Tensor, u: torch.Tensor, v: torch.Tensor, *, scale: float = 1.0, target: str = ""
) -> EditRecord:
    """Add ``scale * u v^T``: when the input has component along ``v``, write along ``u``."""
    if u.numel() != weight.shape[0] or v.numel() != weight.shape[1]:
        raise ValueError(f"u, v must have {weight.shape[0]} and {weight.shape[1]} entries")
    w = weight.to(torch.float32)
    delta = scale * torch.outer(u.flatten().float().to(w.device), v.flatten().float().to(w.device))
    return _commit(weight, w + delta, "rank1", target, {"scale": scale})


@torch.no_grad()
def scale_units(
    weight: torch.Tensor, units: list[int], factor: float, *, axis: str = "in", target: str = ""
) -> EditRecord:
    """Multiply the given input columns (``axis='in'``) or output rows (``axis='out'``) by ``factor``.

    Scaling an MLP down projection's input columns scales those neurons' contribution to the
    residual stream without touching what they read.
    """
    w = weight.to(torch.float32).clone()
    idx = torch.as_tensor(units, device=w.device)
    if axis == "in":
        w[:, idx] *= factor
    elif axis == "out":
        w[idx, :] *= factor
    else:
        raise ValueError("axis must be 'in' or 'out'")
    return _commit(weight, w, "scale_units", target, {"units": list(units), "factor": factor, "axis": axis})


@torch.no_grad()
def feature_edit(
    weight: torch.Tensor,
    decoder_direction: torch.Tensor,
    *,
    mode: str = "remove",
    strength: float = 1.0,
    columns: torch.Tensor | list[int] | None = None,
    target: str = "",
) -> EditRecord:
    """Edit what a module writes along an SAE/transcoder feature's decoder direction.

    ``remove``: project the feature's direction out (strength 1.0 = full removal, 0.5 = halve it).
    ``amplify``: scale the existing component along it by ``1 + strength``.
    """
    if decoder_direction.numel() != weight.shape[0]:
        raise ValueError("decoder direction must live in the space this weight writes to")
    v = _unit(decoder_direction).to(weight.device)
    w = weight.to(torch.float32).clone()
    cols = slice(None) if columns is None else torch.as_tensor(columns, device=weight.device)
    sub = w[:, cols]
    comp = torch.outer(v, v @ sub)
    if mode == "remove":
        w[:, cols] = sub - strength * comp
    elif mode == "amplify":
        w[:, cols] = sub + strength * comp
    else:
        raise ValueError("mode must be 'remove' or 'amplify'")
    return _commit(weight, w, f"feature_edit:{mode}", target, {"strength": strength})
