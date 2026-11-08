# p1_replicate: reproduce the paper baseline (C0)
- **Claim tested:** none yet. This calibrates the harness against Takehi et al. Exp. 8.
- **Target:** Qwen3.5-0.8B, bf16, on TPU.
- **Cells:** C0, budgets 10/25/50, 2 dev agents (Sonnet-class and Flash-class), 1 seed. Plus the unedited baseline.
- **Primary metrics:** MATH-500 (dev subset n=100, 4,096-token cap) and the share of answers hitting the token limit.
- **Success threshold:** the baseline loops (hit-limit ≥ 50% at the dev cap), and at least one C0 run at budget 50 lowers the hit-limit share.
- **Known deviation:** dev runs cap generation at 4,096 tokens, not the paper's 32,768. Final runs (p2) use 32,768.
- **Held-out:** MATH-500, GSM8K test and the IFEval subset. None of them are in the sandbox.
