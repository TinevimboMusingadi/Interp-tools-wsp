"""A randomly initialised 2-layer GPT-2 with a real tokenizer: a target model for harness tests that
needs no weight download. The tokenizer is Qwen2.5's (any locally cached tokenizer works; set
WSP_TEST_TOKENIZER to override)."""

import os
from types import SimpleNamespace

from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

TOKENIZER = os.environ.get("WSP_TEST_TOKENIZER", "Qwen/Qwen2.5-1.5B-Instruct")


def tiny_gpt2() -> SimpleNamespace:
    import torch

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, vocab_size=len(tok), n_positions=128)
    return SimpleNamespace(hf_model=GPT2LMHeadModel(cfg).eval(), tokenizer=tok)
