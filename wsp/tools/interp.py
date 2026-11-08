"""Condition C1 tools: the interpretability toolkit, built on interp-engine.

Two kinds:

- **Weight-only reads** (free, like reading tensors in ``python``): what a neuron or a direction
  writes in vocabulary space, which neurons write a direction, an SAE feature's decoder footprint.
- **Activation reads and interventions** (each call costs ONE forward-pass equivalent, the same
  unit as one ``generate`` batch, so C0 and C1 runs are compute-matched): capture, logit lens,
  attention, contrastive directions, SAE feature activations, steering/ablation tests.

Every tool returns a short text summary and stores the full tensors in the agent's namespace under
``ws[<key>]`` so the agent can keep analysing them in ``python`` without spending more passes.
"""

from __future__ import annotations

from typing import Any

import torch

from wsp.eval.generate import tpu_kwargs
from wsp.harness.env import Env, Tool
from wsp.harness.providers import ToolSpec
from wsp.tools.base import MAX_BATCH, MAX_NEW_TOKENS, c0_tools
from wsp.tools.sae import SAE


def _ws(env: Env) -> dict[str, Any]:
    return env.namespace.setdefault("ws", {})


def _batch(env: Env, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    if not prompts or len(prompts) > MAX_BATCH:
        raise ValueError(f"give 1..{MAX_BATCH} prompts")
    tok = env.model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # so position -1 is every prompt's last real token
    enc = tok(prompts, return_tensors="pt", padding=True)
    dev = env.model.hf_model.device
    return enc["input_ids"].to(dev), enc["attention_mask"].to(dev)


def _capture(env: Env, prompts: list[str], points: list[str]) -> tuple[Any, torch.Tensor]:
    import re

    from interp_engine import capture

    for p in points:
        if any(re.search(pat, p) for pat in env.namespace.get("forbid_points", [])):
            raise PermissionError(f"capturing {p!r} is not allowed in this task")

    ids, mask = _batch(env, prompts)
    with env.forward_window(1), torch.no_grad():
        cache = capture(env.model, ids, points, attention_mask=mask)
    return cache, ids


def mlp_out_matrix(env: Env, layer: int) -> torch.Tensor:
    """The MLP down projection as ``[d_model, d_mlp]`` (columns = what each neuron writes), whether the
    module is an ``nn.Linear`` (``[out, in]``) or GPT-2's ``Conv1D`` (``[in, out]``). For Conv1D this is
    a transposed *view*, so in-place edits through it change the real weight."""
    mod = env.model.arch.mlp_projection(layer, "down")
    w = mod.weight
    return w.T if type(mod).__name__ == "Conv1D" else w


def _unembed(env: Env) -> torch.Tensor:
    return env.model.hf_model.get_output_embeddings().weight  # [vocab, d_model]


def _top_tokens(env: Env, logits: torch.Tensor, k: int) -> list[str]:
    idx = logits.float().topk(k).indices.tolist()
    return [repr(env.model.tokenizer.decode([i])) for i in idx]


def _left(env: Env) -> str:
    return f"(forward passes left: {env.forward_budget - env.forward_used})"


def _vec(env: Env, direction: str | list[float]) -> torch.Tensor:
    """A direction given by name (``ws[name]``) or as a literal list of floats."""
    if isinstance(direction, str):
        v = _ws(env).get(direction)
        if v is None:
            raise KeyError(f"no ws[{direction!r}]; store a [d_model] tensor there first")
        return torch.as_tensor(v).flatten().float()
    return torch.tensor(direction, dtype=torch.float32)


# -- weight-only (free) ------------------------------------------------------------------------
def neuron_logits(env: Env, layer: int, neuron: int, k: int = 10) -> str:
    """What MLP neuron (layer, neuron) writes, read through the unembedding (direct effect only)."""
    col = mlp_out_matrix(env, layer)[:, neuron].float()
    logits = _unembed(env).float() @ col
    _ws(env)[f"neuron_{layer}_{neuron}"] = col.cpu()
    return (f"neuron {layer}.{neuron} writes norm {col.norm():.3f}\n"
            f"  promotes: {_top_tokens(env, logits, k)}\n  suppresses: {_top_tokens(env, -logits, k)}")


def direction_logits(env: Env, direction: str | list[float], k: int = 10) -> str:
    v = _vec(env, direction).to(_unembed(env).device)
    logits = _unembed(env).float() @ v
    return f"promotes: {_top_tokens(env, logits, k)}\nsuppresses: {_top_tokens(env, -logits, k)}"


def writers_of(env: Env, direction: str | list[float], layers: list[int] | None = None, k: int = 10) -> str:
    """Which MLP neurons write most strongly along a direction (cosine of their output column)."""
    v = _vec(env, direction)
    v = v / v.norm()
    layers = layers if layers is not None else list(range(env.model.n_layers))
    rows = []
    for L in layers:
        down = mlp_out_matrix(env, L).float()
        cos = (v.to(down.device) @ down) / down.norm(dim=0).clamp_min(1e-8)
        top = cos.abs().topk(min(k, cos.numel()))
        rows += [(float(cos[i]), L, int(i)) for i in top.indices]
    rows.sort(key=lambda r: -abs(r[0]))
    _ws(env)["writers"] = rows[:k]
    return "\n".join(f"  neuron {L}.{n}: cos {c:+.3f}" for c, L, n in rows[:k])


# -- activation reads (1 pass each) -------------------------------------------------------------
def capture_tool(env: Env, prompts: list[str], points: list[str], key: str = "cap") -> str:
    cache, ids = _capture(env, prompts, points)
    out = [_left(env)]
    for p in points:
        t = cache[p].detach().float().cpu()
        _ws(env)[f"{key}:{p}"] = t
        out.append(f"ws['{key}:{p}'] shape {tuple(t.shape)}, last-token mean norm "
                   f"{t[:, -1].reshape(len(prompts), -1).norm(dim=-1).mean():.3f}")
    _ws(env)[f"{key}:tokens"] = [env.model.tokenizer.convert_ids_to_tokens(r) for r in ids.tolist()]
    return "\n".join(out)


def logit_lens(env: Env, prompt: str, layers: list[int] | None = None, position: int = -1, k: int = 5) -> str:
    from interp_engine import decode_residuals

    layers = layers if layers is not None else list(range(env.model.n_layers))
    cache, _ = _capture(env, [prompt], [f"resid_post.{L}" for L in layers])
    rows = torch.stack([cache[f"resid_post.{L}"][0, position] for L in layers])
    with torch.no_grad(), env.uncharged():
        logits = decode_residuals(env.model, rows)
    _ws(env)["lens"] = logits.float().cpu()
    lines = [_left(env)] + [f"  L{L:>2}: {_top_tokens(env, logits[i], k)}" for i, L in enumerate(layers)]
    return "\n".join(lines)


def attention(env: Env, prompt: str, layer: int, query_position: int = -1, k: int = 5) -> str:
    cache, ids = _capture(env, [prompt], [f"attn_probs.{layer}"])
    probs = cache[f"attn_probs.{layer}"][0].float().cpu()  # [heads, q, k]
    _ws(env)[f"attn_{layer}"] = probs
    toks = env.model.tokenizer.convert_ids_to_tokens(ids[0].tolist())
    lines = [_left(env), f"layer {layer}, query token {toks[query_position]!r}:"]
    for h in range(probs.shape[0]):
        row = probs[h, query_position]
        top = row.topk(min(k, row.numel()))
        lines.append(f"  head {h}: " + ", ".join(f"{toks[i]!r}@{i}={p:.2f}"
                                                for p, i in zip(top.values.tolist(), top.indices.tolist())))
    return "\n".join(lines)


def contrast_direction(env: Env, prompts_a: list[str], prompts_b: list[str], layer: int, name: str,
                       point: str = "resid_post") -> str:
    """Mean last-token activation of A minus B at ``point.layer``; stored as ``ws[name]``."""
    p = f"{point}.{layer}"
    cache, _ = _capture(env, prompts_a + prompts_b, [p])
    acts = cache[p][:, -1].float().cpu()
    a, b = acts[: len(prompts_a)], acts[len(prompts_a):]
    d = a.mean(0) - b.mean(0)
    _ws(env)[name] = d
    sep = ((a @ d).mean() - (b @ d).mean()) / (acts @ d).std().clamp_min(1e-8)
    return f"{_left(env)}\nws['{name}'] = mean(A) - mean(B) at {p}, norm {d.norm():.3f}, separation {sep:.2f} sd"


def steer_test(env: Env, prompts: list[str], direction: str | list[float], layer: int, coeff: float = 4.0,
               method: str = "additive", point: str = "resid_post", max_new_tokens: int = 128) -> str:
    """Generate with a direction added (``additive``) or projected out (``ablate``) at every position.

    A *preview* of a weight edit: nothing in the weights changes. If an ablation fixes the behaviour,
    ``wsp.edit.orthogonalize_out`` on the writers of that direction makes it permanent.
    """
    from interp_engine import SteerMethod, SteerSpec, steer

    ids, mask = _batch(env, prompts)
    tok, hf = env.model.tokenizer, env.model.hf_model
    v = _vec(env, direction).to(hf.device, hf.dtype)
    spec = SteerSpec(vector=v, layer=layer, coeff=coeff, method=SteerMethod(method), point=point,
                     normalize=method == "additive")
    with env.forward_window(1), torch.no_grad(), steer(env.model, [spec]):
        out = hf.generate(input_ids=ids, attention_mask=mask, do_sample=False, pad_token_id=tok.pad_token_id,
                          max_new_tokens=min(max_new_tokens, env.namespace.get("gen_cap", MAX_NEW_TOKENS)),
                          **tpu_kwargs(hf))
    texts = tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)
    return _left(env) + "\n" + "\n\n".join(f"[{i}] {t}" for i, t in enumerate(texts))


def sae_features(env: Env, prompts: list[str], sae: str, k: int = 10, position: int = -1) -> str:
    """Top SAE features at ``position`` (default: last token), with each feature's decoder footprint."""
    model_sae: SAE = env.namespace["saes"][sae]
    cache, _ = _capture(env, prompts, [model_sae.point])
    x = cache[model_sae.point][:, position].float()
    acts = model_sae.encode(x.to(model_sae.device))
    _ws(env)[f"sae:{sae}"] = acts.cpu()
    lines = [_left(env)]
    for i in range(len(prompts)):
        top = acts[i].topk(k)
        feats = [(int(f), float(a)) for a, f in zip(top.values, top.indices) if a > 0]
        lines.append(f"prompt {i}: " + ", ".join(f"f{f}={a:.2f}" for f, a in feats))
    return "\n".join(lines)


def sae_feature_info(env: Env, sae: str, feature: int, k: int = 10) -> str:
    """Free: a feature's decoder direction read through the unembedding, and its strongest MLP writers.
    Stores the direction as ``ws['sae:<sae>:<feature>']`` for steer_test / writers_of / edits."""
    model_sae: SAE = env.namespace["saes"][sae]
    d = model_sae.decoder_direction(feature).float().cpu()
    key = f"sae:{sae}:{feature}"
    _ws(env)[key] = d
    return f"ws['{key}'] stored (point {model_sae.point})\n{direction_logits(env, key, k)}"


def _spec(name: str, desc: str, props: dict, required: list[str]) -> ToolSpec:
    return ToolSpec(name, desc, {"type": "object", "properties": props, "required": required})


_S, _I, _N = {"type": "string"}, {"type": "integer"}, {"type": "number"}
_SL, _IL = {"type": "array", "items": _S}, {"type": "array", "items": _I}
_DIR = {"description": "name of a [d_model] tensor in ws, or a list of floats",
        "anyOf": [_S, {"type": "array", "items": _N}]}
_FREE = "FREE (weights only, no forward pass). "
_PASS = "Costs ONE forward pass. "


def c1_tools() -> dict[str, Tool]:
    t = c0_tools()
    t.update({
        "neuron_logits": Tool(_spec("neuron_logits", _FREE + "What an MLP neuron writes, read through the "
                                    "unembedding.", {"layer": _I, "neuron": _I, "k": _I}, ["layer", "neuron"]),
                              neuron_logits),
        "direction_logits": Tool(_spec("direction_logits", _FREE + "Tokens a residual direction promotes and "
                                       "suppresses.", {"direction": _DIR, "k": _I}, ["direction"]),
                                 direction_logits),
        "writers_of": Tool(_spec("writers_of", _FREE + "MLP neurons whose output column best aligns with a "
                                 "direction: the candidates to edit.", {"direction": _DIR, "layers": _IL, "k": _I},
                                 ["direction"]), writers_of),
        "capture": Tool(_spec("capture", _PASS + "Capture activations at interp-engine points (e.g. "
                              "'resid_post.12', 'mlp_out.15', 'attn_probs.3') into ws.",
                              {"prompts": _SL, "points": _SL, "key": _S}, ["prompts", "points"]), capture_tool),
        "logit_lens": Tool(_spec("logit_lens", _PASS + "Top next-token predictions decoded from the residual "
                                 "stream at each layer.", {"prompt": _S, "layers": _IL, "position": _I, "k": _I},
                                 ["prompt"]), logit_lens),
        "attention": Tool(_spec("attention", _PASS + "Where each head of a layer attends from a query position.",
                                {"prompt": _S, "layer": _I, "query_position": _I, "k": _I},
                                ["prompt", "layer"]), attention),
        "contrast_direction": Tool(_spec("contrast_direction", _PASS + "Direction = mean activation on prompts A "
                                         "minus prompts B at a layer; stored as ws[name].",
                                         {"prompts_a": _SL, "prompts_b": _SL, "layer": _I, "name": _S, "point": _S},
                                         ["prompts_a", "prompts_b", "layer", "name"]), contrast_direction),
        "steer_test": Tool(_spec("steer_test", _PASS + "Generate while adding (method='additive') or ablating "
                                 "(method='ablate') a direction at a layer. Previews an edit without changing "
                                 "weights.", {"prompts": _SL, "direction": _DIR, "layer": _I, "coeff": _N,
                                              "method": {"type": "string", "enum": ["additive", "ablate"]},
                                              "point": _S, "max_new_tokens": _I},
                                 ["prompts", "direction", "layer"]), steer_test),
    })
    return t


def c1_tools_with_sae() -> dict[str, Tool]:
    t = c1_tools()
    t.update({
        "sae_features": Tool(_spec("sae_features", _PASS + "Top sparse-autoencoder features active on each "
                                   "prompt. SAEs available: see namespace `saes`.",
                                   {"prompts": _SL, "sae": _S, "k": _I, "position": _I}, ["prompts", "sae"]),
                             sae_features),
        "sae_feature_info": Tool(_spec("sae_feature_info", _FREE + "A feature's decoder direction: tokens it "
                                       "promotes; stored in ws for steering and editing.",
                                       {"sae": _S, "feature": _I, "k": _I}, ["sae", "feature"]),
                                 sae_feature_info),
    })
    return t
