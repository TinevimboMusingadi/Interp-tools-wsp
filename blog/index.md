---
layout: home
title: Overview
---

# Interpretability tools are the debugger for weight-space programming

*A living overview, updated after every experiment. Each result links to its own post, its
pre-registration, and the raw agent transcripts.*

## The question

[Takehi et al. (2026)](https://rikiyat.github.io/blog/weight-space-programming/) showed frontier
LLMs can read and write neural network weights directly. Their agent saw raw tensors and a budget
of forward passes, but nothing that explained what the model was doing. This series asks what
changes when the agent gets the interpretability toolkit a human researcher would use.

## Claims under test

| # | Claim | Status | Post |
|---|-------|--------|------|
| 1 | Interp tools make weight edits better and cheaper (fewer forward passes, fewer side effects) | not yet run | – |
| 2 | Edits made through interp tools teach us about the model | not yet run | – |
| 3 | Agents that build their own interp tools get better at editing (recursive improvement) | not yet run | – |
| 4 | The loop scales to larger models and to alignment auditing | not yet run | – |

## Conditions

- **C0, paper baseline:** weights in a Python sandbox, plus N forward passes. No interp tools.
- **C1, interp harness:** C0 plus capture, logit lens, attention, SAE features, steering and ablation tests. Each tool is charged in forward-pass equivalents.
- **C2, tool-building:** C1 plus the agent may write new tools, which persist across runs.

Every cell runs on Claude and on Gemini, with 3 seeds.

## Posts

{% for post in site.posts reversed %}
- [{{ post.title }}]({{ post.url | relative_url }}), {{ post.date | date: "%b %-d, %Y" }}
{% endfor %}
