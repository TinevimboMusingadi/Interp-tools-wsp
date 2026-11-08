# Experiments

One folder per experiment: `experiments/<id>/` with `README.md` (pre-registration),
`config.yaml`, `run.py`, `results/` and a generated `REPORT.md`.

## Pre-registration template (write it before the first paid run)

```markdown
# <id>: <title>
- Claim tested: (1-4 from the overview)
- Target model / checkpoint:
- Conditions: C0 / C1 / C2 · providers: claude-vertex, gemini · seeds: 3
- Forward-pass budgets:
- Primary metric and success threshold:
- Secondary metrics (side effects): KL on held-out text, edit size, regression suite
- Held-out data (never mounted in the sandbox):
- Cost estimate (from --dry-run) and cap:
```

## Running

```bash
.venv/Scripts/python -m pytest -q          # harness + edit primitives
```

## TPU runbook

Use the cheapest spot slice that fits (v5e-1 or v6e-1). Setup is in interp-engine's `docs/TPU.md`.
Notes from the interp-engine TPU work:
- Install `torch_tpu` and then `pip install -e` with `--extra-index-url https://download.pytorch.org/whl/cpu`. Never `uv sync` on the VM.
- Call `torch.set_float32_matmul_precision("highest")` before loading an fp32 model, or results drift by about 1e-3.
- Delete the VM after every session: `gcloud compute tpus tpu-vm delete <NAME> --zone=<ZONE> --project=tpu-builder1 --quiet`.
