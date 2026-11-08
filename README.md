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
| `wsp/tools/` | Tool sets, one per condition: `c0` paper baseline, `c1` interp tools, `c1_sae` with SAE features, `c2` tool-building (probes, SAE training, persistent library) |
| `wsp/tasks/` | Tasks: `loop_fix`, `gemma_math`, `syco_identify`, `syco_fix`, `backdoor_audit` |
| `wsp/eval/` | Held-out evaluation: MATH-500, GSM8K, IFEval subset, sycophancy probes, backdoor trigger rates |
| `wsp/organisms.py` | Builds model organisms (sycophant/honest pairs, backdoored and clean models) with hidden labels |
| `wsp/run.py`, `wsp/rsi.py` | Experiment runner (matrix, dry-run cost, resumable cells) and the recursive-improvement rounds |
| `wsp/report.py`, `wsp/export.py` | REPORT.md tables plus a transcript audit, and SFT export of successful trajectories |
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

Set real prices and exact model IDs in `wsp/harness/prices.yaml` before the first paid run.

## Pipeline

Every step after the first runs on TPU. See `scripts/setup_tpu_vm.sh` and `experiments/README.md`.

```bash
python -m wsp.organisms sycophancy --pairs 3 --device tpu        # model organisms (Phases 2-3)
python -m wsp.organisms backdoor --prefix gemma_audit --device tpu
python -m wsp.organisms backdoor --base Qwen/Qwen3.5-0.8B --prefix qwen_audit --device tpu
python -m wsp.run experiments/p1_replicate/config.yaml --dry-run   # cells + cost estimate
python -m wsp.run experiments/p1_replicate/config.yaml             # resumable
python -m wsp.report experiments/p1_replicate                       # REPORT.md + transcript audit
python -m wsp.rsi experiments/p4_rsi/config.yaml                    # Phase 4 rounds
python -m wsp.export experiments/p2_loop_fix --metric math500_acc --min 0.3 --out data/sft.jsonl
```

Expected API spend on the current placeholder prices is about $725 for all experiments. Each
config's dry run prints its own estimate and worst case. Blog drafts live in `blog/_drafts/`, and the
paper outline in `paper/`.
