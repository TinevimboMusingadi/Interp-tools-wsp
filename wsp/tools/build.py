"""Condition C2 tools: the agent may build new interpretability tools, and they persist.

- ``train_probe``: linear probe for a labelled concept at a layer (direction usable for steering,
  ``writers_of`` and edits).
- ``train_sae``: a small TopK SAE on activations from prompts the agent chooses.
- ``save_tool`` / library: Python helper functions the agent writes are saved as source files in
  the tool library and loaded as ``lib.<name>`` in later runs. Library code can call budgeted tools
  through ``run_tool(name, **args)``, so it never gets a free forward pass.

Gradients run only *inside* ``train_probe`` / ``train_sae`` (harness-owned code under
``Env.gradients_allowed``); the edit itself must still be written by the agent. That separation is
what keeps C2 weight-space programming rather than fine-tuning.
"""

from __future__ import annotations

import math
import re
import types
from pathlib import Path

import torch

from wsp.harness.env import Env, Tool
from wsp.tools.interp import _S, _I, _SL, _capture, _left, _spec, _ws, c1_tools_with_sae
from wsp.tools.base import MAX_BATCH
from wsp.tools.sae import train_topk_sae

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,40}$")


def train_probe(env: Env, prompts: list[str], labels: list[int], layer: int, name: str,
                point: str = "resid_post", steps: int = 300) -> str:
    if len(prompts) != len(labels) or set(labels) - {0, 1}:
        return "labels must be 0/1, one per prompt"
    p = f"{point}.{layer}"
    chunks = [prompts[i:i + MAX_BATCH] for i in range(0, len(prompts), MAX_BATCH)]
    acts = []
    for c in chunks:
        cache, _ = _capture(env, c, [p])
        acts.append(cache[p][:, -1].float().cpu())
    x = torch.cat(acts)
    y = torch.tensor(labels, dtype=torch.float32)
    n = len(y)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, te = perm[: max(1, int(0.8 * n))], perm[max(1, int(0.8 * n)):]
    mu, sd = x[tr].mean(0), x[tr].std(0).clamp_min(1e-6)
    xs = (x - mu) / sd
    with env.gradients_allowed():
        w = torch.zeros(x.shape[1], requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=0.05)
        for _ in range(steps):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(xs[tr] @ w + b, y[tr]) + 1e-3 * w.pow(2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        acc = lambda idx: float((((xs[idx] @ w + b) > 0).float() == y[idx]).float().mean()) if len(idx) else float("nan")  # noqa: E731
        direction = (w.detach() / sd)
    _ws(env)[name] = direction
    return (f"{_left(env)}\nprobe ws['{name}'] at {p}: train acc {acc(tr):.2f}, held-out acc {acc(te):.2f} "
            f"(n={n}). Direction is in raw activation space.")


def train_sae(env: Env, prompts: list[str], layer: int, name: str, d_sae: int = 2048, k: int = 32,
              point: str = "resid_post", steps: int = 1500) -> str:
    p = f"{point}.{layer}"
    passes = math.ceil(len(prompts) / MAX_BATCH)
    if env.forward_used + passes > env.forward_budget:
        return f"refused: needs {passes} forward passes, {env.forward_budget - env.forward_used} left"
    acts = []
    for i in range(0, len(prompts), MAX_BATCH):
        cache, ids = _capture(env, prompts[i:i + MAX_BATCH], [p])
        a = cache[p].float().cpu()
        mask = ids.cpu() != env.model.tokenizer.pad_token_id
        acts.append(a[mask])
    x = torch.cat(acts)
    with env.gradients_allowed():
        sae, stats = train_topk_sae(x, p, d_sae=d_sae, k=k, steps=steps)
    env.namespace.setdefault("saes", {})[name] = sae
    lib = env.namespace.get("library_dir")
    if lib:
        Path(lib, "saes").mkdir(parents=True, exist_ok=True)
        sae.save(Path(lib, "saes", f"{name}.npz"))
    return f"{_left(env)}\nSAE '{name}' at {p}: {stats}. Use sae_features / sae_feature_info with sae='{name}'."


def save_tool(env: Env, name: str, description: str, code: str) -> str:
    """Save a helper function to the tool library. ``code`` must define ``def main(**kwargs)``."""
    if not NAME_RE.match(name):
        return "name must be snake_case, 2-41 chars"
    mod = types.ModuleType(f"lib.{name}")
    mod.__dict__.update({k: env.namespace[k] for k in ("model", "torch", "ws", "run_tool") if k in env.namespace})
    try:
        exec(compile(code, f"<lib.{name}>", "exec"), mod.__dict__)  # noqa: S102
    except Exception as e:  # noqa: BLE001
        return f"code failed to load: {e!r}"
    if not callable(mod.__dict__.get("main")):
        return "code must define main(**kwargs)"
    env.namespace["lib"].__dict__[name] = mod
    lib = env.namespace.get("library_dir")
    if lib:
        d = Path(lib, "tools")
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.py").write_text(f'"""{description}"""\n\n{code}\n', encoding="utf-8")
    return f"saved lib.{name}; call it in python as lib.{name}.main(...)"


def list_library(env: Env) -> str:
    lib = env.namespace["lib"].__dict__
    tools = {k: (v.__doc__ or "").strip()[:200] for k, v in lib.items() if isinstance(v, types.ModuleType)}
    saes = list(env.namespace.get("saes", {}))
    return f"tools: {tools or 'none'}\nSAEs: {saes or 'none'}"


def load_library(env: Env, library_dir: str | None) -> None:
    """Load saved tools and SAEs into the namespace at the start of a run."""
    from wsp.tools.sae import load_sae

    env.namespace["lib"] = types.SimpleNamespace()
    env.namespace["library_dir"] = library_dir
    env.namespace["run_tool"] = lambda name, **args: env.call(name, args)
    env.namespace.setdefault("ws", {})
    if not library_dir:
        return
    d = Path(library_dir)
    for f in sorted((d / "tools").glob("*.py")) if (d / "tools").exists() else []:
        src = f.read_text(encoding="utf-8")
        doc = src.split('"""')[1] if src.startswith('"""') else ""
        body = src.split('"""', 2)[2] if src.startswith('"""') else src
        save_tool(env, f.stem, doc, body)
    for f in sorted((d / "saes").glob("*.npz")) if (d / "saes").exists() else []:
        env.namespace.setdefault("saes", {})[f.stem] = load_sae(f, device=str(env.model.hf_model.device))


def c2_tools() -> dict[str, Tool]:
    t = c1_tools_with_sae()
    t.update({
        "train_probe": Tool(_spec("train_probe", "Costs ceil(n/64) forward passes. Train a linear probe for a "
                                  "0/1 concept at a layer; the direction is stored as ws[name].",
                                  {"prompts": _SL, "labels": {"type": "array", "items": _I}, "layer": _I,
                                   "name": _S, "point": _S}, ["prompts", "labels", "layer", "name"]),
                            train_probe),
        "train_sae": Tool(_spec("train_sae", "Costs ceil(n/64) forward passes. Train a TopK sparse autoencoder "
                                "on all token activations of your prompts at a layer; saved to the library.",
                                {"prompts": _SL, "layer": _I, "name": _S, "d_sae": _I, "k": _I, "point": _S},
                                ["prompts", "layer", "name"]), train_sae),
        "save_tool": Tool(_spec("save_tool", "Save a reusable Python helper to the persistent tool library. "
                                "`code` must define main(**kwargs); it can use model, torch, ws and "
                                "run_tool(name, **args) (budgeted).", {"name": _S, "description": _S, "code": _S},
                                ["name", "description", "code"]), save_tool),
        "list_library": Tool(_spec("list_library", "List saved tools and SAEs.", {}, []), list_library),
    })
    return t
