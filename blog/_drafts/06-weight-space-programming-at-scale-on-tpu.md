---
title: "Weight-space programming at scale on TPU"
series: interp-tools-wsp
experiment: p5_scale
---

*Post 6. Claim 4: does the tool advantage grow with model size?*

## Setup
interp-engine eager backend on TorchTPU (spot v6e-1). Precision note: TPU fp32 matmuls default to bf16-rounded inputs, so we set "highest".

## Results
<!-- C1 − C0 gap at 0.8B vs ~4B. Cost per run on TPU. -->

## Credits
TorchTPU; interp-engine (TPU support: decoderesearch/interp-engine#35).
