# p5_scale: does the tool advantage grow with scale? (claim 4)
- **Claim tested:** 4. Raw weights stop fitting in context, while interp tools still work.
- **Target:** a ~4B model in bf16 on v6e-1. Verify the repo id and that it fits in memory first.
- **Cells:** C0 vs C1 × budget 25 × 2 agents × 2 seeds, plus baseline.
- **Primary metric:** the C1 − C0 gap on MATH-500, compared with the same gap at 0.8B in p2_loop_fix.
- **TPU note:** call `torch.set_float32_matmul_precision("highest")` for any fp32 analysis.
