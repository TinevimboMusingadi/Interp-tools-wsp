"""The network's shape for drawing it: layers, attention kind per layer, MLP width, vocabulary.

Read from the HF config in the local cache, so replay needs no model weights and no network. Works for
plain decoder configs and for multimodal ones that nest the language model under ``text_config``.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B"


def _config_path(model_id: str) -> Path | None:
    try:
        from huggingface_hub import try_to_load_from_cache

        p = try_to_load_from_cache(model_id, "config.json")
        return Path(p) if isinstance(p, str) else None
    except Exception:  # noqa: BLE001 - no hub installed or no cache: fall back to the default shape
        return None


def arch_from_config(cfg: dict, model_id: str = "") -> dict:
    tc = cfg.get("text_config", cfg)
    n = int(tc.get("num_hidden_layers", 0))
    types = list(tc.get("layer_types") or [])
    if len(types) != n:
        types = ["full_attention"] * n
    return {
        "model": model_id,
        "n_layers": n,
        "d_model": int(tc.get("hidden_size", 0)),
        "d_mlp": int(tc.get("intermediate_size", 0)),
        "vocab": int(tc.get("vocab_size", 0)),
        "layer_types": ["full" if "full" in t else "linear" if "linear" in t else t for t in types],
        "n_heads": int(tc.get("num_attention_heads", 0)),
        "n_kv_heads": int(tc.get("num_key_value_heads", 0) or 0),
        "linear_heads": int(tc.get("linear_num_value_heads", 0) or 0),
        "tied_embeddings": bool(cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings", False))),
    }


# Qwen3.5-0.8B, used when the config is not cached (e.g. replay on a machine without the model).
FALLBACK = arch_from_config({"text_config": {
    "num_hidden_layers": 24, "hidden_size": 1024, "intermediate_size": 3584, "vocab_size": 248320,
    "layer_types": (["linear_attention"] * 3 + ["full_attention"]) * 6, "num_attention_heads": 8,
    "num_key_value_heads": 2, "linear_num_value_heads": 16, "tie_word_embeddings": True}}, DEFAULT_MODEL)


@functools.lru_cache(maxsize=8)
def load_arch(model_id: str = DEFAULT_MODEL) -> dict:
    p = _config_path(model_id)
    if p is None:
        return {**FALLBACK, "model": model_id or DEFAULT_MODEL}
    return arch_from_config(json.loads(p.read_text(encoding="utf-8")), model_id)
