"""The three parallel SD draft-adaptation approaches.

An edge-cloud speculative-decoding deployment adapts one of three draft
forms to the fine-tuned target, and this package keeps them side by side
as equally deployment-shaped options:

- :mod:`drafts.plain` — a plain small causal-LM draft (e.g.
  Qwen3-1.7B-Base): target SFT, auxiliary-only distillation, and a
  member-data SFT boundary variant.
- :mod:`drafts.eagle3` — the published EAGLE-3 speculator head, KD-adapted
  to the fine-tuned target.
- :mod:`drafts.mtp` — the checkpoint's native MTP head, independently
  initialized for auxiliary KD and member-data MTP-CE after target SFT.

All three write run artifacts under ``experiments/results`` and replay
their data through the same frozen post-cutoff pools.
"""
