# Supervise all controlled-SFT pairs through one matrix interface

Status: accepted. The complete rerun is exposed through
`retrain_unified_matrix.sh`: five model pairs, three datasets, target epochs 1
and 3, and condition seeds 1919, 1949, and 1978. This is 90 conditions and 270
condition artifacts. The existing plain-model and EAGLE-3/MTP launchers remain
the implementation behind the unified interface rather than duplicating their
training and recovery logic.

One preflight validates the four selected physical GPUs and every pinned
offline model. It prepares nine schema-v3 raw document-ID assignments under
all five training tokenizers. The child launchers receive that shared split
root and skip their narrower preflights. Thus the same dataset and seed always
select exactly the same 2,000 member, 2,000 nonmember, 2,000 draft-auxiliary,
and 600 audit-auxiliary documents across all five pairs, while
tokenizer-specific audits remain attached to the shared manifest. Audit
auxiliaries are never used to update target or draft language-model weights.

Both child matrices use the same target optimization contract: full-parameter
BF16 SFT with paged 8-bit AdamW, learning rate `2e-5`, effective batch size 16,
and the condition seed reused directly for data and training. Auxiliary
plain-draft KD and every EAGLE-3/MTP head branch use 384 optimizer updates at
learning rate `2e-5`; the plain member-data draft retains its matched
1/3-epoch SFT definition. Every KD branch uses temperature 2.0. All three
datasets use the empirically validated micro-batch 2 with accumulation 8. The
MTP member branch retains its native cross-entropy objective, for which
temperature is inapplicable.

The plain and head child matrices run sequentially because each already runs
four single-GPU workers. The supervisor still invokes the second child if the
first exits unsuccessfully, then reports aggregate status and returns nonzero
unless all 90 conditions are complete. Relaunching uses each child's existing
completion checks to preserve finished stages and retry incomplete work. The
default `unified_matrix_audit600_v2` result root is separate from every earlier
matrix, so the rigorous shared-split rerun cannot silently mix with old
artifacts.
