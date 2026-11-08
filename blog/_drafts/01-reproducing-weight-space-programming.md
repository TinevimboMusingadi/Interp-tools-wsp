---
title: "Reproducing weight-space programming with Claude and Gemini"
series: interp-tools-wsp
experiment: p1_replicate
---

*Post 1 of the series. [Overview](/) · pre-registration: `experiments/p1_replicate/README.md`*

## What Takehi et al. showed
<!-- Short summary: an LLM reading and writing raw weights. The Qwen3.5-0.8B loop fix (7.9% to 48.1% MATH-500 by editing 0.016% of weights) and where it struggled: 10 and 25 forward passes, the sycophant from weights alone, Gemma 270M. -->

## The harness
- The agent's code runs in a separate process. Gradients and unbudgeted forward passes are refused, not just discouraged.
- Every edit is diffed against a snapshot. Every turn is in a public transcript.
- One forward pass means one batch of up to 64 prompts.
<!-- figure: harness diagram (agent ↔ sandbox ↔ model, budget meter) -->

## Results
<!-- RESULTS: paste the table from experiments/p1_replicate/REPORT.md. Do not hand-type numbers. -->

## What it costs
<!-- From REPORT.md: $ per run, tokens per run, TPU minutes. -->

## Next
Post 2 gives the agent an interpretability toolkit and asks whether 10 forward passes can do what took 50.
