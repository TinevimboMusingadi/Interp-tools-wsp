---
layout: post
title: "A debugger for weight-space programming? First results"
description: "Gemini 4 agents edit Qwen3.5-0.8B's weights to stop it looping, with and without interpretability tools. Replay every run on a live map of the network."
series: interp-tools-wsp
experiment: p2_loop_fix
---

<link rel="stylesheet" href="{{ '/assets/p2/p2.css' | relative_url }}">

*Post 2 of the series. Claim 1: interpretability tools make weight edits better and cheaper.*

**Short version:** not yet. We gave Gemini 4 agents a small model that loops, a few forward
passes and its raw weights, and asked them to fix it. The best edit came from an agent **without**
interp tools. It shrank eight "second-guessing" tokens like *Wait* and *Alternatively*, and MATH
went from 41% to 47%. Agents **with** the tools mostly ignored them. The one that leaned on them
hardest made the biggest edit of the experiment, 82 million weights, and didn't improve MATH.
Each cell is a single run, so the honest reading is "no evidence of a difference yet". You can
replay every run below and watch which parts of the network each agent read and rewrote.

## The task

Qwen3.5-0.8B is good at maths until it isn't. It solves the problem, writes *"Wait, let me
re-check…"*, and then loops until it hits the length limit. Without any edit, it gets **41%** of
100 held-out MATH-500 problems right, and **28%** of its answers run to the 8k-token limit.

[Takehi et al. (2026)](https://rikiyat.github.io/blog/weight-space-programming/) showed an LLM can
fix this by editing the weights directly. They removed one "looping" direction and MATH went from
7.9% to 48.1%, but it needed 50 forward passes. Our question is whether interpretability tools let
an agent find a fix like that with fewer.

## Setup

- **Agents:** Gemini 4 Carbon (most capable) and Gemini 4 Argon (cheaper).
- **Budgets:** 10 or 25 forward passes, plus a hard cap of 4M tokens per run.
- **C0, raw weights:** a Python sandbox holding the loaded model, plus `generate` (one forward pass per prompt batch). This is the paper's setup.
- **C1, interp tools:** C0 plus `logit_lens`, `capture`, `attention`, `neuron_logits`, `writers_of`, `contrast_direction`, `direction_logits` and `steer_test`.
  - Every tool call costs forward passes, so the two conditions get the same compute.
- **Same everything else:** prompt, practice problems (GSM8K and MATH train) and token cap. The held-out evals are never visible to the agent.
  - **Sandbox:** no gradients, no file or network access, and every edit is diffed against the original weights.

## Results

<div class="wsp-fig" id="p2-results">Loading the results figure…</div>

Three things stand out:

1. **The best edit used raw weights.** C0 Carbon at 10 passes got to 47% MATH, and cut answers that hit the limit from 28% to 20%. It changed 8,155 of 752 million weights (0.001%).
2. **Interp tools didn't help on this seed.** Every valid C1 cell scored 40–41%, which is the unedited model's score.
3. **More passes didn't help either.** At 25 passes, three of the four agents scored at baseline. Two of them kept no edit at all.

## Watch the agents work

This is the [WSP Studio](https://github.com/TinevimboMusingadi/Interp-tools-wsp/tree/main/wsp/studio)
replaying the real transcripts.

- **The map:** the embedding is on the left, then Qwen's 24 layers, then the tied unembedding.
  - Each layer is an attention block (**F** for full attention, **L** for linear) above an MLP block, with the residual stream between them.
- **Colours:**
  - **Blue:** what the agent reads.
  - **Orange:** what it edits.
  - **Green sweep:** a forward pass.
- **Edited tokens** appear by name on the embedding.
- **Controls:** press play, drag the scrubber, or use ← → on your keyboard. Pick any run from the menu, or press "Replay this run" in the results figure above.

<div class="wsp-studio" id="p2-studio" data-src="{{ '/studio/app/' | relative_url }}">
  <iframe title="WSP Studio replay of a p2 run" loading="lazy"></iframe>
  <div class="cap"><span class="now"></span><a class="full" target="_blank" rel="noopener">Open the full Studio, with tool calls and the agent's turns ↗</a></div>
</div>

## What the agents changed

**C0 Carbon, the winning edit.** Carbon read the embedding matrix directly. It noticed the
unembedding is tied to it, and found an "average token" direction. Then it scaled down eight token
rows and nudged them toward that average:

| Token | Kept | Pushed toward average |
|---|---|---|
| `Wait`, ` Wait` | 72% | +0.12 |
| `Alternatively`, ` Alternatively` | 72% | +0.12 |
| `Alternative`, ` Alternative` | 78% | +0.10 |
| `-check` (as in re-check) | 80% | +0.08 |
| `-read` (as in re-read) | 80% | +0.08 |

Because the head is tied, this makes the model a little less likely to *say* these words. It also
blurs what they mean when they do come up. A tiny, readable edit: you could explain it to someone
in one sentence.

**C0 Argon at 25 passes, the best diagnosis.** It projected late-layer MLP neurons onto the
vocabulary and found a cluster in layer 22:
- an "alternative" neuron;
- two "wait" neurons;
- an anti-"conclusion" neuron, whose most negative tokens are *conclusion*, *conclude* and *结论*;
- two end-of-turn neurons.

It silenced or flipped six of them, showed that four failing practice problems now finished
cleanly, and submitted. Held-out MATH barely moved (42%), and *more* answers hit the length limit
(40%). A clean mechanistic story isn't the same as a fix that generalises.

**C1 Carbon at 25 passes, the biggest edit.** It used `capture` to record the residual stream on
two groups of prompts and built a direction from their difference. Then it removed that direction
from every attention and MLP output in layers 6–22. That's 82 million weights. GSM8K jumped from
58% to **69%**, but 61% of MATH answers now run to the limit. It's a big, blunt edit, and the
opposite of what the tools were meant to encourage.

## Where the calls went

<div class="wsp-fig" id="p2-tools">Loading the tool-use figure…</div>

Even with interp tools on offer, agents wrote raw Python for most of their calls. Only C1 Argon at
10 passes really used them, with 39 neuron readouts. A tool the agent doesn't reach for can't
help it, so getting the agent to use the tools is part of the problem.

## What went wrong, and what's next

- **One seed per cell.** With 100 problems per eval, differences under about 5 points are noise. More seeds are next.
- **Almost nobody submits.** Six of the eight agents ran into the 4M-token cap mid-investigation. One submitted, and one was cut short by the bug below. The plan is a "submit by turn N" nudge, or a lower cap.
- **A harness bug.** C1 Carbon at 10 passes got an empty reply from the API at turn 8, and the harness treated that as "done". It will retry next time, and that cell gets re-run.
- **Practice vs held-out.** Agents verify their edits on a handful of practice problems, and those fixes don't always carry over. A held-out check the agent can call, at a price, may help.
- **The sandbox held.** Every run tripped the jail on the same two harness files, and one also tripped it on the Hugging Face token during a `generate` call. Both look like library code rather than the agent snooping, and the jail refused all of them.

Total cost for this round: about $61 of Gemini API, plus about a day on one v6e-1 spot TPU.

## Reproduce it

```bash
python -m wsp.run experiments/p2_loop_fix/config.yaml        # on a TPU VM
python -m wsp.studio --runs runs                              # replay locally at localhost:8800
python scripts/export_replays.py runs/p2_loop_fix             # the files behind the embed above
```

The results JSON for every cell is in
[`experiments/p2_loop_fix/results`](https://github.com/TinevimboMusingadi/Interp-tools-wsp/tree/main/experiments/p2_loop_fix/results).

<script src="{{ '/assets/p2/p2.js' | relative_url }}" defer></script>
