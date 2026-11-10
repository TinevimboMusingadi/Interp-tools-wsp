"""Batched generation for evaluation (runs after the agent is done, outside its budget)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


PAD_MULTIPLE = 64  # prompt lengths are padded to a multiple of this so compiled shapes are reused
# A static cache holds batch x max_new_tokens positions. 32 x 4096 fits a v6e-1 (32 GB HBM) with
# Qwen3.5-0.8B; 32 x 32768 ran out of memory, so the batch shrinks as the cap grows.
CACHE_TOKENS = 32 * 4096


def tpu_kwargs(hf: Any) -> dict:
    """On TPU, a static KV cache keeps one tensor shape for the whole decode (measured on v6e-1 with
    Qwen3.5-0.8B: 2,811 vs 177 tok/s once compiled). Elsewhere, the default cache."""
    return {"cache_implementation": "static"} if hf.device.type == "tpu" else {}


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
def generate(model: Any, prompts: list[str], s: Sampling, batch_size: int | None = None,
             seed: int = 0) -> list[Gen]:
    tok, hf = model.tokenizer, model.hf_model
    if batch_size is None:
        batch_size = max(1, min(32, CACHE_TOKENS // max(1, s.max_new_tokens)))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    cfg_eos = getattr(getattr(hf, "generation_config", None), "eos_token_id", None)
    eos = {tok.eos_token_id, *(cfg_eos if isinstance(cfg_eos, list) else [cfg_eos])} - {None}
    torch.manual_seed(seed)
    out: list[Gen] = []
    for i in range(0, len(prompts), batch_size):
        chunk = [render(tok, p, s) for p in prompts[i:i + batch_size]]
        enc = tok(chunk, return_tensors="pt", padding=True, pad_to_multiple_of=PAD_MULTIPLE,
                  add_special_tokens=not s.chat).to(hf.device)
        kw: dict[str, Any] = {"max_new_tokens": s.max_new_tokens, "pad_token_id": tok.pad_token_id, **tpu_kwargs(hf)}
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
