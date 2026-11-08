"""Batched generation for evaluation (runs after the agent is done, outside its budget)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class Sampling:
    max_new_tokens: int = 4096
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    chat: bool = True  # apply the tokenizer's chat template
    enable_thinking: bool | None = None  # passed to chat templates that accept it (Qwen3.x)


@dataclass
class Gen:
    text: str
    n_tokens: int
    hit_limit: bool  # ran to max_new_tokens without an EOS: the looping signature


def render(tok: Any, prompt: str, s: Sampling) -> str:
    if not s.chat or not getattr(tok, "chat_template", None):
        return prompt
    kw = {} if s.enable_thinking is None else {"enable_thinking": s.enable_thinking}
    return tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                   add_generation_prompt=True, **kw)


@torch.no_grad()
def generate(model: Any, prompts: list[str], s: Sampling, batch_size: int = 16, seed: int = 0) -> list[Gen]:
    tok, hf = model.tokenizer, model.hf_model
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    cfg_eos = getattr(getattr(hf, "generation_config", None), "eos_token_id", None)
    eos = {tok.eos_token_id, *(cfg_eos if isinstance(cfg_eos, list) else [cfg_eos])} - {None}
    torch.manual_seed(seed)
    out: list[Gen] = []
    for i in range(0, len(prompts), batch_size):
        chunk = [render(tok, p, s) for p in prompts[i:i + batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=not s.chat).to(hf.device)
        kw: dict[str, Any] = {"max_new_tokens": s.max_new_tokens, "pad_token_id": tok.pad_token_id}
        if s.temperature > 0:
            kw.update(do_sample=True, temperature=s.temperature, top_p=s.top_p)
            if s.top_k:
                kw["top_k"] = s.top_k
        else:
            kw["do_sample"] = False
        gen = hf.generate(**enc, **kw)[:, enc["input_ids"].shape[1]:]
        for row in gen.tolist():
            ended = next((j for j, t in enumerate(row) if t in eos), None)
            n = len(row) if ended is None else ended + 1
            out.append(Gen(tok.decode(row[:n], skip_special_tokens=True), n,
                           hit_limit=ended is None and n >= s.max_new_tokens))
    return out
