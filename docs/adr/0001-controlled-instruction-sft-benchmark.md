# Use controlled instruction SFT for the primary membership benchmark

**Status:** accepted

The primary Qwen3 benchmark uses source-stratified document chunks as assistant
responses paired with deterministic technical-writing prompts, and masks prompt
tokens from the SFT loss. This replaces the legacy causal-LM continuation
pilot as the primary training definition because it gives the requested true
instruction SFT while preserving a randomized, hash-deduplicated member versus
nonmember assignment; the result must therefore be interpreted as controlled
SFT-membership evidence rather than pretraining-membership evidence.
