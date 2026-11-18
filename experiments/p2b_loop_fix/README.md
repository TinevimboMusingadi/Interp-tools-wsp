# p2b_loop_fix: does the debugger help? (pre-registration)

Written and committed before any p2b cell runs. Deviations found later are added under **Deviations**,
never edited into the sections above it.

**Claim tested:** 1. Interpretability tools make weight edits better and cheaper.

**Why a re-run:** p2_loop_fix (1 seed, 100-problem dev evals, Gemini 4 Carbon and Argon) found no C1
advantage, but it was too loose to say much:
- One C1 cell died on an empty provider reply.
- Seven of eight agents hit the token cap without submitting.
- No agent could check whether its edit generalised.

## Hypotheses
- **H1 (primary):** at a 10-pass budget, C1 beats C0 on MATH-500.
- **H2 (secondary):** C1 at 10 passes reaches C0's 25-pass score.
- **H3 (secondary):** C1 edits are smaller (fewer weights changed) at equal or better MATH-500.

## Design
- **Task:** `loop_fix` on Qwen3.5-0.8B. The prompt is unchanged from p1/p2.
- **Conditions:** C0 (python, generate, check_practice, edit_report, submit) and C1 (C0 plus the interp tools).
  - Each C1 tool description carries one real example call (`EX` in `wsp/tools/interp.py`, frozen at the commit that adds this README).
  - The examples use prompts unrelated to math or looping, and recommend no method.
- **Agents:** Gemini 4 Carbon and Gemini 4 Argon, both through Vertex AI, thought summaries logged.
- **Grid:** budgets 10 and 25, seeds 0, 1 and 2, so 24 cells plus the baseline.
- **What a seed sets:** the practice-problem order, and the Gemini sampling seed.
- **Caps per run:** 2M tokens (cached tokens count), $15, 120 turns. Every tool result shows the agent its token, turn and pass budget, with warnings at 50, 75 and 90%.
- **Empty or malformed provider replies:** the agent is re-prompted up to twice. A run the provider still ends is **invalid**: it is not reported, and it is re-run.

## Metrics
- **Primary:** MATH-500, full 500 problems, 8,192-token generation cap, Qwen sampling (temperature 0.6, top-p 0.95, top-k 20), eval seed 0. The edited weights are evaluated in a fresh process.
- **Secondary:**
  - MATH-500 share of answers hitting the limit;
  - GSM8K (300 test problems, fixed sample);
  - IFEval subset (all supported items);
  - weights changed;
  - passes used;
  - submitted (yes/no);
  - API cost.

## Test
- Seeds are pooled per problem, and every cell is scored on the same problems.
- Differences are paired bootstrap means over problems, with 95% percentile CIs (4,000 resamples), computed by `python -m wsp.report experiments/p2b_loop_fix`.
- **H1 is supported** if C1 − C0 ≥ +5 points at budget 10, with a 95% CI above 0, for **both** agents (`prereg` in `config.yaml`).
- Any other outcome is reported as not supported. The effect sizes and CIs are reported either way.

## Deviations from the original p2 plan (known before running)
- **Generation cap:** 8,192 tokens, not 32,768. A 32k cap OOMs a v6e-1, and the full set at 32k would take days per cell.
- **Budget 50:** dropped, for cost. p1 already has C0 budget-50 cells.
- **Threshold:** +5 points instead of +10. A pilot-informed change: p2 C0 cells sat 0–6 points above baseline.
- **Agents:** the same two Gemini agents as p2. A draft of this plan used Claude Opus 5.5 in place of Argon,
  but Claude is not available on Vertex for this GCP project (every endpoint returns 404), so the swap was
  dropped before any p2b cell ran. Both agents come from one lab; the claim is limited to Gemini agents.

## Deviations found while running
(none yet)
