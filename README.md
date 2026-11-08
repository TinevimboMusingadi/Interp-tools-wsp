# Interp tools for weight-space programming

Research code and blog for one question: **do interpretability tools make LLMs better at reading and
writing neural network weights?**

[Takehi et al. (2026)](https://rikiyat.github.io/blog/weight-space-programming/) showed frontier LLMs
can edit weights directly. For example, they removed one looping direction from Qwen3.5-0.8B and
MATH-500 went from 7.9% to 48.1%. Their agent worked from raw tensors and forward passes. This project
gives the agent an interpretability toolkit, built on
[interp-engine](https://github.com/decoderesearch/interp-engine), and measures the difference in
compute-matched, pre-registered experiments. The full write-up is in [`blog/`](blog/).

## Layout

| Path | What it holds |
|---|---|
| `wsp/harness/` | Agent loop, providers (Claude on Vertex AI, Gemini), budgets and ledger, sandboxed environment |
| `wsp/tools/` | Tool sets, one per condition. `c0` is the paper baseline; `c1` (interp) and `c2` (tool-building) are next |
| `wsp/edit/` | Weight-edit primitives (`orthogonalize_out`, `rank1`, `feature_edit`, …) and verification (`weight_diff`, `kl_on_text`) |
| `experiments/` | One folder per pre-registered experiment |
| `blog/` | GitHub Pages site: a living overview plus one post per experiment |

## Rules the sandbox enforces

The agent's code runs in a separate process, with API keys removed from its environment.
- **Gradients are refused**, because weights are written by reasoning, not by an optimizer.
- **Any forward pass outside a budgeted tool is refused.** That includes running a single layer or calling `interp_engine.capture` directly.
- **Every edit is diffed** against a snapshot taken at load time.

## Setup

```bash
python -m venv --system-site-packages .venv
.venv/Scripts/python -m pip install -e D:/interp-engine      # fork branch feat/tpu-eager (TPU support)
.venv/Scripts/python -m pip install -e ".[agents,dev]"
.venv/Scripts/python -m pytest -q
```

Environment variables:
- `WSP_GCP_PROJECT` and `WSP_VERTEX_REGION`: Claude on Vertex AI.
- `GEMINI_API_KEY`: Gemini.

Set real prices in `wsp/harness/prices.yaml` before the first paid run.
