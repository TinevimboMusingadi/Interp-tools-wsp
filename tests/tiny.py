"""A randomly initialised 2-layer GPT-2 wrapped in a real interp-engine EagerModel: a target model
for harness and tool tests that needs no weight download. The tokenizer is Qwen2.5's (any locally
cached tokenizer works; set WSP_TEST_TOKENIZER to override)."""

import os

from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel

TOKENIZER = os.environ.get("WSP_TEST_TOKENIZER", "Qwen/Qwen2.5-1.5B-Instruct")


def tiny_gpt2():
    import torch
    from interp_engine import EagerModel

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    cfg = GPT2Config(n_layer=2, n_head=2, n_embd=32, n_inner=64, vocab_size=len(tok), n_positions=256)
    hf = GPT2LMHeadModel(cfg).eval()
    hf.config._attn_implementation = "eager"
    return EagerModel("tiny-gpt2", hf_model=hf, tokenizer=tok, device="cpu", attn_implementation="eager")
