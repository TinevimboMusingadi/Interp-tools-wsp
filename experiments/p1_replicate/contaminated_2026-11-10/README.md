# Set aside: p1 rerun of 2026-11-10

Run on an on-demand v6e-1 (us-east5-b), evaluation capped at 8,192 tokens.

- `c0_b10_flash_s0` is excluded. Gemini Flash reached task-registry functions through `gc.get_objects()`
  after the module purge (Linux only; `spawn` had re-run `python -m wsp.run` in the child). It read the
  eval sampling settings and a docstring naming the source experiment. It changed no weights, so its
  score equals the baseline, but the run is not clean. 8 file reads were refused by the jail.
- `c0_b10_pro_s0` crashed after 21 turns on a Vertex 429 (tokens-per-minute quota), which stopped the run.
- The baseline (no agent involved) is valid and kept in `../results/baseline.json`.

Fixed in the following commit: the child no longer imports main and refuses to start if harness code is
reachable; provider 429s are retried; a failed cell no longer stops the run.
