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

**Instruction SFT**:
A user-prompt to assistant-response training example where only the assistant
response is a training target; prompt tokens do not define membership loss.
_Avoid_: causal-LM continuation, completion fine-tuning

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
