"""Condition C0 tools: Python with the weights loaded, budgeted generation, and nothing more.

- ``python``: run code in a persistent namespace with the model's weights loaded (``model``).
  Reading and writing tensors is free; running the network is not (the forward guard refuses).
- ``generate``: one budgeted forward pass = one batch of up to ``MAX_BATCH`` prompts, text out.
- ``edit_report``: what has changed in the weights so far.
- ``submit``: end the run with an explanation of the edit.
"""

from __future__ import annotations

from typing import Any

import torch

from wsp.eval.generate import PAD_MULTIPLE, tpu_kwargs
from wsp.harness.env import Env, Tool
from wsp.harness.providers import ToolSpec

MAX_BATCH = 64
MAX_NEW_TOKENS = 1024


def _python(env: Env, code: str) -> str:
    return env.run_python(code)


@torch.no_grad()
def _generate(env: Env, prompts: list[str], max_new_tokens: int = 256, temperature: float = 0.0,
              chat: bool = True) -> str:
    if not prompts:
        return "no prompts given"
    if len(prompts) > MAX_BATCH:
        return f"at most {MAX_BATCH} prompts per forward pass"
    max_new_tokens = min(max_new_tokens, env.namespace.get("gen_cap", MAX_NEW_TOKENS))
    tok = env.model.tokenizer
    hf = env.model.hf_model
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    use_chat = chat and bool(getattr(tok, "chat_template", None))
    if use_chat:
        prompts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                           add_generation_prompt=True) for p in prompts]
    batch = tok(prompts, return_tensors="pt", padding=True, pad_to_multiple_of=PAD_MULTIPLE,
                add_special_tokens=not use_chat).to(hf.device)
    kw: dict[str, Any] = {"max_new_tokens": max_new_tokens, "pad_token_id": tok.pad_token_id, **tpu_kwargs(hf)}
    kw.update({"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False})
    with env.forward_window(1):
        out = hf.generate(**batch, **kw)
    new = out[:, batch["input_ids"].shape[1]:]
    texts = tok.batch_decode(new, skip_special_tokens=True)
    stop_ids = {tok.eos_token_id, tok.pad_token_id} - {None}
    ended = [bool(stop_ids & set(row.tolist())) for row in new]
    left = env.forward_budget - env.forward_used
    body = "\n\n".join(f"[{i}]{'' if e else ' (hit max_new_tokens)'} {t}"
                         for i, (t, e) in enumerate(zip(texts, ended)))
    return f"(forward passes left: {left})\n{body}"


def _edit_report(env: Env) -> str:
    return env.edit_report()


def _submit(env: Env, explanation: str, answer: dict | None = None) -> str:
    env.namespace["__submitted__"] = explanation
    env.namespace["__answer__"] = answer or {}
    return "submitted"


def c0_tools() -> dict[str, Tool]:
    return {
        "python": Tool(ToolSpec(
            "python",
            "Run Python in a persistent namespace. `model` is an interp-engine EagerModel; its weights are "
            "`model.hf_model` (a torch module). `torch` is imported; `import wsp.edit` has edit primitives "
            "(nn.Linear convention [out, in]; for GPT-2 Conv1D pass `weight.T`). "
            "Reading/editing weights is free; running the network here is refused (use `generate`). "
            "Gradients are disabled.",
            {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
        ), _python),
        "generate": Tool(ToolSpec(
            "generate",
            f"Spend ONE forward pass: generate continuations for up to {MAX_BATCH} prompts with the current "
            "(edited) weights. Returns the generated text.",
            {"type": "object", "properties": {
                "prompts": {"type": "array", "items": {"type": "string"}},
                "max_new_tokens": {"type": "integer", "default": 256},
                "temperature": {"type": "number", "default": 0.0},
                "chat": {"type": "boolean", "default": True,
                         "description": "wrap each prompt in the model's chat template"},
            }, "required": ["prompts"]},
        ), _generate),
        "edit_report": Tool(ToolSpec(
            "edit_report", "Report how many weights have changed so far, by tensor.",
            {"type": "object", "properties": {}},
        ), _edit_report),
        "submit": Tool(ToolSpec(
            "submit", "Finish: submit the current weights with an explanation of what you changed and why. "
            "If the task asks for a structured answer, put it in `answer`.",
            {"type": "object", "properties": {"explanation": {"type": "string"}, "answer": {"type": "object"}},
             "required": ["explanation"]},
        ), _submit),
    }


