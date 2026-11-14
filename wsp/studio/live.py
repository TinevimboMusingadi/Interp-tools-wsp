"""Hands-on studio: the model loaded in-process, the agent's own C1 tools, and weight edits with undo.

The person at the keyboard is the user on their own machine (the server binds to localhost), so there is
no sandbox, no API spend and no forward-pass budget. Every action returns the same ``touches`` the replay
view draws, plus the exact current edit state, so the network map lights up the same way for a human as
for an agent.
"""

from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from typing import Any

import torch

from wsp.studio.diff import sparse_diff
from wsp.studio.events import delta_touches, tool_touches

WRITERS = ("mlp.down_proj.weight", "self_attn.o_proj.weight", "linear_attn.out_proj.weight")


class Studio:
    def __init__(self, model_id: str, device: str = "cpu", dtype: str = "bfloat16"):
        from wsp.harness.env import Env, eager_loader
        from wsp.tools import TOOLSETS

        t = time.time()
        self.model_id = model_id
        self.model = eager_loader(model_id, device=device, dtype=dtype)
        self.env = Env(self.model, forward_budget=10**9, tools=TOOLSETS["c1"]())
        self.env.namespace["gen_cap"] = 2048
        self.env.namespace["run_tool"] = lambda name, **a: self.env.call(name, a)
        self.lock = threading.Lock()
        self.undo: list[tuple[str, dict[str, torch.Tensor]]] = []  # (label, tensors before the edit)
        self.dirty: set[str] = set()  # tensors edited this session (the only ones that can differ from base)
        self.load_s = round(time.time() - t, 1)

    # -- helpers ---------------------------------------------------------------------------------
    @property
    def params(self) -> dict[str, torch.Tensor]:
        return dict(self.model.hf_model.named_parameters())

    def ws(self) -> dict[str, Any]:
        return self.env.namespace.setdefault("ws", {})

    def state(self) -> dict:
        # only tensors ever edited in this session can differ from base: diff those, not all 750M weights
        p = self.params
        cur = {k: p[k].detach().cpu() for k in self.dirty if k in p}
        d = sparse_diff(self.env.reference, cur)
        self.dirty = set(d)  # undone / reverted tensors drop out
        touches = delta_touches(d)
        tok = self.model.tokenizer
        for t in touches:  # name the edited embedding / unembedding rows by their tokens
            if t["part"] in ("embed", "head") and t.get("index"):
                t["labels"] = [tok.decode([i]) for i in t["index"][:64]]
        return {"diff": d, "touches": touches, "n_changed": sum(x["n"] for x in d.values()),
                "undo": [label for label, _ in self.undo]}

    def status(self) -> dict:
        return {"model": self.model_id, "device": str(self.model.hf_model.device), "load_s": self.load_s,
                "ws": {k: list(v.shape) for k, v in self.ws().items() if hasattr(v, "shape")},
                "undo": [label for label, _ in self.undo]}

    def _push(self, label: str, names: list[str]) -> None:
        p = self.params
        self.undo.append((label, {n: p[n].detach().clone() for n in names}))
        self.dirty |= set(names)
        del self.undo[:-20]

    def _vec(self, direction: Any) -> torch.Tensor:
        v = self.ws()[direction] if isinstance(direction, str) else torch.tensor(direction)
        return v.detach().float().flatten()

    # -- actions ---------------------------------------------------------------------------------
    def handle(self, action: str, body: dict) -> dict:
        with self.lock:
            fn = getattr(self, f"do_{action}", None)
            if fn is None:
                return {"error": f"unknown action {action!r}"}
            t = time.time()
            try:
                out = fn(**body)
            except Exception as e:  # noqa: BLE001 - shown in the UI
                return {"error": f"{type(e).__name__}: {e}"}
            out["seconds"] = round(time.time() - t, 2)
            return out

    def do_tool(self, name: str, args: dict | None = None) -> dict:
        """Any agent tool by name (python, generate, logit_lens, attention, neuron_logits, writers_of,
        contrast_direction, steer_test, capture, direction_logits, edit_report)."""
        args = args or {}
        before = None
        if name == "python":  # python can edit anything: snapshot everything for undo (CPU copy)
            before = {k: p.detach().clone() for k, p in self.params.items()}
        out = self.env.call(name, args)
        res: dict[str, Any] = {"output": out, "touches": tool_touches(name, args, out)}
        res["output"] = re.sub(r"^\(forward passes left: \d+\)\n?", "", out)  # no budget in the studio
        if before is not None:
            changed = [k for k, p in self.params.items() if not torch.equal(before[k], p.detach())]
            self.dirty |= set(changed)
            if changed:
                self.undo.append((f"python ({len(changed)} tensors)", {k: before[k] for k in changed}))
                del self.undo[:-20]
        if name == "logit_lens":
            res["lens"] = self._lens_table(args.get("layers"), int(args.get("k", 5)))
        if name == "attention":
            res["attention"] = self._attn_grid(int(args.get("layer", 0)), args.get("prompt", ""))
        return {**res, "state": self.state() if name == "python" else None}

    def do_chat(self, prompt: str, max_new_tokens: int = 256, steer: dict | None = None) -> dict:
        """Chat with the model; with ``steer`` = {direction, layer, coeff, method} the reply is steered."""
        if steer and steer.get("direction"):
            args = {"prompts": [prompt], "max_new_tokens": max_new_tokens, **steer}
            return self.do_tool("steer_test", args)
        return self.do_tool("generate", {"prompts": [prompt], "max_new_tokens": max_new_tokens})

    def do_edit(self, op: str, tensor: str | None = None, units: list[int] | None = None,
                factor: float = 1.0, axis: str = "out", direction: Any = None, layers: list[int] | None = None,
                tokens: list[str] | None = None) -> dict:
        """Weight edits a person can make from the UI, each undoable:

        - ``scale``: multiply rows (``axis='out'``) or columns (``'in'``) of ``tensor`` by ``factor``.
          ``tokens`` (strings) select embedding rows by token, e.g. the "Wait" edit.
        - ``ablate_direction``: remove ``direction`` from what every writer in ``layers`` writes to the
          residual stream (MLP down_proj, attention o_proj / out_proj): steering baked into weights.
        """
        from wsp.edit import orthogonalize_out

        p = self.params
        if op == "scale":
            if tokens:
                tok = self.model.tokenizer
                units = sorted({i for s in tokens for i in tok.encode(s, add_special_tokens=False)[:1]})
                tensor = tensor or next(k for k in p if "embed_tokens" in k)
            if not tensor or tensor not in p or not units:
                return {"error": "scale needs a tensor name and units (or tokens)"}
            self._push(f"scale {tensor} x{factor} ({len(units)} units)", [tensor])
            # in place on just those rows / columns (scale_units would copy the whole 254M-weight
            # embedding to float32 first); same result
            w = p[tensor].data
            idx = torch.as_tensor(units, device=w.device)
            with torch.no_grad():
                if axis == "out":
                    w[idx] = (w[idx].float() * factor).to(w.dtype)
                else:
                    w[:, idx] = (w[:, idx].float() * factor).to(w.dtype)
            per = w.shape[1] if axis == "out" else w.shape[0]
            rec = {"op": "scale_units", "target": tensor, "units": units, "factor": factor, "axis": axis,
                   "n_touched": len(units) * per}
            return {"record": rec, "units": units, "state": self.state()}
        if op == "ablate_direction":
            v = self._vec(direction)
            n_layers = self.model.n_layers
            names = [k for k in p if k.endswith(WRITERS) and _layer(k) in (layers or range(n_layers))]
            self._push(f"ablate direction from {len(names)} writers", names)
            recs = [vars(orthogonalize_out(p[k].data, v.to(p[k].device), target=k)) for k in names]
            return {"records": recs, "state": self.state()}
        return {"error": f"unknown edit op {op!r}"}

    def do_undo(self) -> dict:
        if not self.undo:
            return {"error": "nothing to undo"}
        label, saved = self.undo.pop()
        p = self.params
        with torch.no_grad():
            for k, v in saved.items():
                p[k].data.copy_(v.to(p[k].device))
        return {"undone": label, "state": self.state()}

    def do_revert(self) -> dict:
        p = self.params
        with torch.no_grad():
            for k in self.dirty:
                p[k].data.copy_(self.env.reference[k].to(p[k].device, dtype=p[k].dtype))
        self.undo.clear()
        return {"state": self.state()}

    def do_state(self) -> dict:
        return {"state": self.state()}

    def do_save(self, path: str = "runs/studio/weights.pt") -> dict:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in self.model.hf_model.state_dict().items()}, out)
        return {"saved": str(out)}

    def do_tokens(self, text: str) -> dict:
        tok = self.model.tokenizer
        ids = tok.encode(text, add_special_tokens=False)
        return {"ids": ids, "tokens": [tok.decode([i]) for i in ids]}

    # -- structured views of what the tools stored in ws ---------------------------------------------
    def _lens_table(self, layers: list[int] | None, k: int) -> dict:
        logits = self.ws().get("lens")
        if logits is None:
            return {}
        layers = layers if layers is not None else list(range(self.model.n_layers))
        probs = logits.float().softmax(-1)
        top = probs.topk(k, dim=-1)
        tok = self.model.tokenizer
        return {"layers": layers, "rows": [[{"tok": tok.decode([int(i)]), "p": round(float(q), 4)}
                                            for q, i in zip(top.values[r], top.indices[r])]
                                           for r in range(len(layers))]}

    def _attn_grid(self, layer: int, prompt: str) -> dict:
        probs = self.ws().get(f"attn_{layer}")
        if probs is None:
            return {}
        tok = self.model.tokenizer
        ids = tok(prompt)["input_ids"]
        n = probs.shape[-1]
        toks = [tok.decode([i]) for i in ids][-n:]
        return {"layer": layer, "tokens": toks, "heads": probs[:, -64:, -64:].float().numpy().round(3).tolist()}


def _layer(name: str) -> int:
    import re

    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else -1
