# p1, first attempt (2026-11-08): contaminated, not reported

Gemini 3.8 Flash, condition c0, budgets 10/25/50, one seed, plus the unedited baseline. On the
100-problem MATH-500 dev subset (4,096-token cap, T=0.6) every cell is within noise of the baseline
(37%): 34%, 38%, 42%.

The cells are contaminated: the agents spent most of their tokens searching the sandbox for the task's
answer and found some of it. The 10-pass cell read the task config from modules in memory; the 25-pass
cell found the task registry via `gc.get_objects()` (a docstring citing Takehi et al. Exp. 8); the
50-pass cell read harness source whose docstring described the source paper's edit. Every route is now
closed (`wsp/harness/jail.py`, module purge + `gc.collect()` in `wsp/harness/sandbox.py`, neutral
docstrings) and covered by tests. Transcripts are in `runs/p1_replicate/` (not committed).

The baseline (37%, 24% of answers hitting the cap) is far from the paper's 7.9% because the dev eval
caps generations at 4,096 tokens rather than ~32k, so fewer answers loop.
