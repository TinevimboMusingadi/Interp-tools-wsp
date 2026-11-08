# Paper outline (Phase 6)

**Working title:** Interpretability Tools as the Harness for Weight-Space Programming

1. **Introduction.** Weight-space programming (Takehi et al. 2026) and its limits. Thesis: interp tools are the debugger.
2. **Harness.** Sandbox rules, forward-pass accounting, conditions C0/C1/C2, tool list, edit primitives and verification.
3. **Experiments:**
   - (a) the loop fix (p2_loop_fix);
   - (b) sycophancy, identify and repair (p2_syco_*);
   - (c) the Gemma 270M boundary (p2_gemma_math);
   - (d) backdoor audit (p3_audit);
   - (e) recursive tool improvement (p4_rsi);
   - (f) scale (p5_scale).
4. **Results.** One table per claim, all generated from `experiments/*/REPORT.md`.
5. **What the edits teach us.** The mechanistic findings.
6. **Limitations.** Planted organisms, small models, the grader, sandbox isolation, cost.
7. **Release.** Harness, tool library, transcripts and edited checkpoints.
