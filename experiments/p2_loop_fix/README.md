# p2_loop_fix: interp tools as a debugger (core claim 1)
- **Claim tested:** 1. Interp tools make weight edits better and cheaper.
- **Target:** Qwen3.5-0.8B. Paper reference: MATH-500 goes from 7.9% to 48.1% with 50 passes; 10 and 25 passes barely help.
- **Cells:** C0 vs C1 × budgets 10/25/50 × 2 agents × 3 seeds, plus baseline.
- **Primary metric:** MATH-500 (full set, 32,768-token cap, Qwen-recommended sampling).
- **Success threshold:** C1 at budget 10 beats C0 at budget 10 by at least 10 points, averaged over seeds, on both agents.
- **Secondary metrics:** GSM8K, IFEval subset, hit-limit share, % of weights changed, and cost per run.
- **Interp-tool accounting:** each activation tool costs 1 forward pass, the same as one `generate` batch.
