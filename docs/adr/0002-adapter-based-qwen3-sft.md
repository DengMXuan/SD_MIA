# Use LoRA adapters for the Qwen3-8B target SFT

**Status:** accepted

The Qwen3-8B target and trained Qwen3-1.7B draft variants use LoRA adapters for
the controlled SFT conditions. Full AdamW training of the 8B model would not
fit reliably in the available single 80GB A100 once weights, gradients,
optimizer state, and activations are included; adapter SFT keeps the experiment
within one GPU while making the scope explicit: the result measures
adapter-based SFT membership, not full-weight fine-tuning membership.
