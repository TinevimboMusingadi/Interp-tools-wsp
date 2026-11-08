---
title: "A debugger for weight-space programming"
series: interp-tools-wsp
experiment: p2_loop_fix
---

*Post 2. Claim 1: interp tools make weight edits better and cheaper.*

## The question
A human editing code uses a debugger. An LLM editing weights in the original setup has raw tensors and a few forward passes. What happens when it gets logit lens, activation capture, contrastive directions and steering previews, each charged at the same price as a forward pass?

## Setup
- Same task, prompt, budget and token cap in both conditions. The only difference is the tool list (C0 vs C1).
- Budgets 10, 25 and 50; two agent models; 3 seeds each.

## Results
<!-- RESULTS: REPORT.md table + chart of MATH-500 vs forward passes, one line per condition and agent. -->

## What the agent found
<!-- From transcripts: which direction or neurons it edited, how it found them (tool-call trace), and how that compares with the paper's layer 15 and 19 MLP direction. -->

## Side effects
<!-- KL, IFEval subset, % weights changed. -->

## Limits
<!-- What didn't work, where C1 did not help. -->
