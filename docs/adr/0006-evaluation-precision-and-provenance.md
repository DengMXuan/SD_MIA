# Separate protocol causality, numerical drift and evaluation provenance

On 2026-09-25, four Qwen3 KD acceptance probes failed because BF16 forward
computations at different sequence lengths differed numerically. Same-length
future-token interventions left the checked predictions unchanged, and FP32
prefix comparisons passed. Formal experiments retain BF16; runtime validation
separates same-length causal interventions from prefix numerical drift, records
both, and rechecks anomalous prefix comparisons in FP32 without changing formal
inference precision. A failed or unavailable reference check blocks the condition.
Simply increasing the original tolerance would hide the distinction, while
running all formal computations in FP32 would change the experimental setting.

Live compatibility checks also exposed a severe Gemma4 prefix discrepancy in
the current memory-efficient SDPA backend, removed by the math backend, and an
EAGLE remote-head call that omitted the mask required to construct causal
attention. Evaluation now scopes Gemma4 forwards/generation to math SDPA while
retaining BF16, uses math SDPA for reference checks, and explicitly supplies the
EAGLE mask as in head training. These fixes do not change stored checkpoints or
training precision. They prevent a stricter gate from merely hiding real defects.

Quality provenance follows the conservative local import closure of its entry
point rather than every file in the shared source directory. Relevant computation,
data, model and validation changes still invalidate reuse; unrelated audit report
formatting does not. Model checkpoints, including their own remote implementation,
remain fingerprinted. The user chose to archive the original batch unchanged and
recompute all 90 tasks in a new batch, accepting the cost to avoid mixing evaluation
contracts. No legacy metrics or generation chunks are adopted into the new batch.
