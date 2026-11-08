# p2_gemma_math: where tools may not help (claim 1, the boundary)
- **Claim tested:** 1 at its limit. The paper found no gain on Gemma 3 270M, which rarely loops.
- **Cells:** C0, C1, C1+SAE × budget 50 × 2 agents × 2 seeds. C1+SAE needs a verified Gemma Scope SAE path in the config.
- **Primary metric:** MATH-500 at Gemma's model-card sampling (8,192 tokens).
- **Success threshold:** C1+SAE improves MATH-500 by at least 2 points over baseline. If it doesn't, we report the negative result.
