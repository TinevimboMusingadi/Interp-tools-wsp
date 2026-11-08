"""Sparse autoencoders: load pretrained ones (Gemma Scope ``params.npz``, SAELens safetensors) and
train small TopK SAEs from captured activations (the C2 tool-building path).

Shapes: ``W_enc [d_in, d_sae]``, ``b_enc [d_sae]``, ``W_dec [d_sae, d_in]``, ``b_dec [d_in]``,
optional JumpReLU ``threshold [d_sae]``, optional ``k`` for TopK.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class SAE:
    W_enc: torch.Tensor
    b_enc: torch.Tensor
    W_dec: torch.Tensor
    b_dec: torch.Tensor
    point: str  # interp-engine address the SAE reads, e.g. "resid_post.12"
    threshold: torch.Tensor | None = None
    k: int | None = None

    @property
    def device(self) -> torch.device:
        return self.W_enc.device

    @property
    def d_sae(self) -> int:
        return self.W_enc.shape[1]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x.float() - self.b_dec) @ self.W_enc + self.b_enc
        if self.k is not None:
            top = pre.topk(self.k, dim=-1)
            return torch.zeros_like(pre).scatter_(-1, top.indices, top.values.relu())
        if self.threshold is not None:
            return pre * (pre > self.threshold)
        return pre.relu()

    @torch.no_grad()
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def decoder_direction(self, feature: int) -> torch.Tensor:
        return self.W_dec[feature]

    def save(self, path: str | Path) -> None:
        arrs = {k: getattr(self, k).cpu().numpy() for k in ("W_enc", "b_enc", "W_dec", "b_dec")}
        if self.threshold is not None:
            arrs["threshold"] = self.threshold.cpu().numpy()
        np.savez(path, point=np.array(self.point), k=np.array(-1 if self.k is None else self.k), **arrs)


def load_sae(path: str | Path, point: str | None = None, device: str = "cpu") -> SAE:
    """Load ``.npz`` (Gemma Scope / our own) or ``.safetensors`` (SAELens) SAE weights."""
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path)
        t = {k: torch.from_numpy(z[k]).float().to(device) for k in z.files if k not in ("point", "k")}
        point = point or (str(z["point"]) if "point" in z.files else None)
        k = int(z["k"]) if "k" in z.files and int(z["k"]) > 0 else None
    else:
        from safetensors.torch import load_file

        t = {k: v.float().to(device) for k, v in load_file(str(path)).items()}
        k = None
    if point is None:
        raise ValueError("SAE point (interp-engine address it reads) is required")
    return SAE(t["W_enc"], t["b_enc"], t["W_dec"], t["b_dec"], point=point, threshold=t.get("threshold"), k=k)


def train_topk_sae(acts: torch.Tensor, point: str, d_sae: int, k: int = 32, steps: int = 2000,
                   lr: float = 1e-3, batch: int = 1024, seed: int = 0) -> tuple[SAE, dict]:
    """Train a TopK SAE on ``acts [n_tokens, d_in]``. Needs gradients: call it only from a
    harness-owned tool inside ``Env.gradients_allowed()``. Returns the SAE and fit stats."""
    g = torch.Generator().manual_seed(seed)
    acts = acts.float()
    n, d = acts.shape
    b_dec = acts.mean(0).clone()
    W_dec = torch.randn(d_sae, d, generator=g)
    W_dec = W_dec / W_dec.norm(dim=1, keepdim=True)
    params = {
        "W_enc": torch.nn.Parameter(W_dec.T.clone()),
        "b_enc": torch.nn.Parameter(torch.zeros(d_sae)),
        "W_dec": torch.nn.Parameter(W_dec),
        "b_dec": torch.nn.Parameter(b_dec),
    }
    opt = torch.optim.Adam(params.values(), lr=lr)
    var = acts.var(0).sum().clamp_min(1e-8)
    for _ in range(steps):
        x = acts[torch.randint(0, n, (min(batch, n),), generator=g)]
        pre = (x - params["b_dec"]) @ params["W_enc"] + params["b_enc"]
        top = pre.topk(k, dim=-1)
        f = torch.zeros_like(pre).scatter(-1, top.indices, top.values.relu())
        loss = ((f @ params["W_dec"] + params["b_dec"] - x) ** 2).sum(-1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            params["W_dec"] /= params["W_dec"].norm(dim=1, keepdim=True).clamp_min(1e-8)
    sae = SAE(*(params[k_].detach() for k_ in ("W_enc", "b_enc", "W_dec", "b_dec")), point=point, k=k)
    with torch.no_grad():
        recon = sae.decode(sae.encode(acts))
        fvu = float(((recon - acts) ** 2).sum(-1).mean() / (var))
        alive = int((sae.encode(acts) > 0).any(0).sum())
    return sae, {"fvu": fvu, "alive_features": alive, "d_sae": d_sae, "k": k, "n_tokens": n}
