"""WSP Studio: watch agents do weight-space programming, and do it yourself.

- **Replay** a finished run (``runs/<exp>/<cell>/transcript.jsonl``): tool calls on the left, the network in
  the middle lighting up what each call reads, steers or edits, the agent's chat on the right.
- **Live**: follow a transcript that is still being written (local file, or a TPU VM over ssh).
- **Studio**: load the model and use the same interp tools by hand: chat, logit lens, attention, contrast
  directions, steering, and weight edits with undo.

    python -m wsp.studio --runs runs [--model Qwen/Qwen3.5-0.8B --device cpu]

This package is a viewer for the harness, not part of it: the sandboxed agent never imports it.
"""
