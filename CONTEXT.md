# Edge–Cloud SD Membership-Audit Context

This glossary fixes the language used by the controlled instruction-SFT and
edge–cloud speculative-decoding audit experiments.

## Training and audit records

**Member record**:
A source-derived response chunk intentionally included in the target model's
SFT set. Membership is defined by the controlled training assignment, not by a
guess about the model's original pretraining corpus.

**Nonmember record**:
A source-derived response chunk drawn from the same source-stratified pool as a
member record but excluded from the target SFT set.
_Avoid_: negative sample, unseen sample

**Auxiliary record**:
A source-derived response chunk kept disjoint from both audit classes and used
only to train or calibrate a draft variant.

**Controlled SFT**:
An instruction-SFT benchmark in which member status is assigned by randomized,
hash-deduplicated data construction so the causal effect of SFT exposure can be
audited.
_Avoid_: pretraining benchmark, production-training claim

**Full-parameter target SFT**:
A controlled SFT condition in which every target parameter is trainable and
the resulting artifact is a complete model checkpoint rather than an adapter.
_Avoid_: LoRA target SFT, adapter checkpoint

**Instruction SFT**:
A user-prompt to assistant-response training example where only the assistant
response is a training target; prompt tokens do not define membership loss.
_Avoid_: causal-LM continuation, completion fine-tuning

**Model pair**:
A target model and its size-matched-family draft model evaluated together under
the same controlled data condition.
_Avoid_: model family, model variant

**Draft adaptation**:
The training relationship that produces a draft for a target, distinguished by
whether it is auxiliary-distilled or trained on member records.
_Avoid_: draft checkpoint when discussing the training relationship

**EAGLE-3 adaptation variant**:
Which controlled split supplies KD records for an EAGLE-3 head. Every head
condition retains both an auxiliary-KD variant and a member-KD variant.
_Avoid_: EAGLE-3 model variant

**MTP adaptation variant**:
Whether a native MTP head is adapted by KD on auxiliary records or trained
directly on member records. Every MTP condition retains both variants.
The trainable MTP layer starts from the same pinned native export in both
branches; frozen verifier-owned embeddings and output weights are taken from
the condition's SFT target checkpoint.
_Avoid_: MTP model variant

**Target freeze boundary**:
The point immediately after a condition's member SFT completes. All EAGLE-3
and MTP head variants for that condition share that checkpoint and may update
only their own head parameters.
_Avoid_: joint target-head adaptation

**Head adaptation budget**:
The fixed post-SFT training allowance for a drafter head: 384 optimizer updates
at effective batch size 16. A condition's 1/3-epoch setting applies only to
target SFT and never changes this head budget.
_Avoid_: head epoch setting, 384 micro-batches

**Head optimization settings**:
All EAGLE-3 and MTP head branches use learning rate 2e-5. Every KD branch uses
temperature 2.0, matching the established Qwen3-8B/Qwen3-1.7B distillation
configuration. Temperature is not applicable to the native MTP member branch's
direct cross-entropy objective.
_Avoid_: model-specific head learning rates, temperature for MTP-CE

**Head branch initialization**:
The rule that a condition's auxiliary and member head branches start as
independent copies of the same published or native head. Neither branch may
inherit parameters updated by the other.
_Avoid_: sequential head adaptation

**Experiment condition**:
One fixed combination of benchmark dataset, target SFT epoch setting, and
random seed for a model pair.
_Avoid_: run, trial (when referring to the controlled matrix)

**Unified controlled-SFT matrix**:
The five-pair experiment scheduled by one supervising launcher: two plain
target–draft pairs and three frozen-target EAGLE-3/MTP pairs, crossed with
three benchmark datasets, two target epoch settings, and three condition
seeds. It contains 90 experiment conditions and 270 condition artifacts. The
supervisor performs one all-tokenizer preflight and runs its two four-GPU child
matrices sequentially so no GPU receives overlapping workers.
_Avoid_: 270 experiments, simultaneous child matrices

**Plain condition artifacts**:
The three full checkpoints produced by one plain-model experiment condition:
one SFT target, one auxiliary-data KD draft, and one member-data SFT draft. The
36-condition plain-pair matrix therefore contains 108 checkpoint directories.
_Avoid_: three conditions per seed

**Head condition artifacts**:
The three saved model artifacts produced by one head-based experiment
condition: one frozen SFT target, one auxiliary-trained head, and one
member-trained head. The 54-condition matrix therefore contains 162
checkpoint directories.
_Avoid_: 162 experiment conditions

**Stage-complete condition**:
An experiment condition whose target, auxiliary head, and member head stages
each have a loadable checkpoint, a complete manifest, and an atomic completion
marker. Relaunches reuse completed stages and run only missing stages. A failed
condition does not stop independent conditions, but the matrix launcher records
every failure and exits nonzero after all runnable conditions finish.
_Avoid_: directory-exists completion, whole-matrix fail-fast

**Condition seed**:
The root seed that jointly identifies the controlled member/nonmember/auxiliary
assignment and every stochastic training stage for one experiment condition.
For a fixed benchmark and seed, all target–head pairs share the same raw
document assignment, and every data and training stage directly reuses the
same numeric seed without stage-specific offsets.
_Avoid_: training-only seed, data-only seed

**Shared raw split**:
The common member, nonmember, and auxiliary document-ID assignment reused by
all target–head pairs for one benchmark and condition seed. Tokenization may
differ by target, but documents and class membership may not. Before the
assignment is frozen, candidates are checked in seeded order under every
participating tokenizer; exact-token and truncated 13-gram near-duplicates are
rejected and backfilled from the same frozen pool. A shared split is usable only
with its matching all-tokenizer audit attestation.
_Avoid_: tokenizer-specific split

## Edge–cloud speculative decoding

**Target model**:
The cloud-side full language model whose controlled SFT membership is audited.
_Avoid_: verifier model when discussing training membership

**Draft model**:
The client-side white-box language model that proposes candidate tokens and
provides the audit's local reference distribution.
_Avoid_: student model unless discussing distillation specifically

**Protocol feedback**:
The accept/reject, verified-token, correction-token, or equivalent semantic
information returned by the speculative-decoding verifier as part of normal
protocol operation.
_Avoid_: timing side channel, packet side channel

**Draft source**:
The relationship between a draft and the target's training exposure, such as
base, auxiliary-distilled, or member-SFT draft.

**Membership audit**:
An evaluation that ranks member records above matched nonmember records using a
specified combination of draft white-box features and protocol feedback.
_Avoid_: privacy proof, production attack rate

## Pretraining membership audits

**Pretraining member / nonmember**: labels supplied by the MIMIR benchmark
(Pile train / held-out test provenance), not the controlled-SFT random assignment.
The prepared manifest freezes these labels, source hashes and token positions.
A benchmark membership label does not additionally prove per-token exposure in
every intermediate Pythia checkpoint.

**Pretrained draft**: the unadapted Pythia 1.4B checkpoint paired with Pythia 6.9B.
Both were trained on The Pile; this role (`draft_pretrained`) must not be described
as member-blind auxiliary distillation or as member-SFT adaptation.

**Raw completion audit**: the first real text token supplies context; subsequent
text tokens are scored, with no synthetic SFT instruction or appended EOS. This
is a separate protocol from the instruction-SFT response masking above.
