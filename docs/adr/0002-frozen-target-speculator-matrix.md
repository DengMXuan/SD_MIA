# Freeze targets before adapting speculative heads

Status: accepted. The speculative-head matrix contains 54 conditions: three
target–head pairs, three frozen benchmark pools, target SFT epochs 1 and 3,
and seeds 1919, 1949, and 1978. Each condition first performs full-parameter
BF16 target SFT, freezes that complete checkpoint, and then independently
initializes and trains an auxiliary-data head and a member-data head; EAGLE-3
uses KD for both branches, while native MTP uses KD for the auxiliary branch
and MTP cross-entropy for the member branch. Joint target–head training is
excluded because it would move the membership-bearing target differently
between comparison branches.

All pairs reuse one raw document-ID assignment for a benchmark and seed. The
same numeric condition seed is passed directly to split construction, target
SFT, and both head branches. Head training is fixed at 384 optimizer updates,
effective batch size 16, learning rate `2e-5`, and temperature 2.0 wherever KD
is used. Published EAGLE-3 heads and the original native MTP head are pinned to
immutable cached revisions and are reloaded independently for each branch.
MTP's trainable native layer comes from that original export, while its frozen
verifier-owned embedding and output weights come from the condition's SFT
target so proposals use the matching vocabulary projection.

Shared assignments use schema-v2 deterministic filtering. After the condition
seed orders the common length-eligible pool, every participating tokenizer
filters exact truncated-token duplicates and candidates with at least 50%
13-gram overlap against a retained record. Rejected candidates are backfilled
before the first 6,000 IDs are partitioned. Every tokenizer then reruns the
exact cross-split audit with the preregistered 80% failure threshold; training
requires the manifest hash and all-tokenizer audit attestation to match.

Four exclusive single-GPU workers use physical GPUs 3–6. Artifact completion
is marked only after an atomically promoted checkpoint and manifest exist;
relaunches skip complete stages, preserve and reuse a complete target, and
retry only missing head stages. A condition failure is isolated from other
workers, but the launcher reports all failures and exits nonzero after the
remaining runnable conditions finish.
