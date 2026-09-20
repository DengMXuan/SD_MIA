# Preserve full-parameter adaptation for the DP defense

Status: accepted for full-parameter adaptation, deployment-pair accounting,
matched target privacy budgets, and the equal target/member-draft epsilon
grid and delta allocation. The user confirmed the complete implementation plan.

The controlled-SFT DP extension retains full-parameter training for the target
and member-data draft, with document-level gradient clipping, Gaussian noise,
and privacy accounting. The auxiliary-KD draft is freshly distilled from the
DP target on the disjoint auxiliary split. The user selected full-parameter
adaptation over DP-LoRA to preserve comparability with the existing full-model
experiments, accepting the additional gradient computation and memory cost.

The existing non-DP experiment behavior and artifacts must remain compatible.
DP runs must use separate artifacts and verifiable privacy provenance; an old
non-DP checkpoint or its distilled draft cannot be relabeled as DP. The
fixed-candidate B=2 audit and nonmember-only detector fitting remain the
evaluation protocol. This decision does not authorize starting a training
matrix or claim that a DP implementation has already been validated.

Privacy budgets are reported separately for two deployment scenarios:
target plus auxiliary-KD draft, and target plus member-SFT draft. The KD pair
inherits its target's guarantee when distillation uses only fixed disjoint
auxiliaries; the member pair requires composition of target and draft
training. The primary budget is not defined over all three artifacts jointly.
These pairwise guarantees do not describe simultaneous exposure to both
deployments or to multiple training versions.

The two deployment scenarios use the same target privacy budget. In the
member-SFT scenario, target and draft receive equal epsilon budgets (1:1).
The user explicitly revised the earlier equal-total-pair-budget decision:
pair-level budgets now differ, because the member scenario additionally
composes direct member-data draft training. Sharing the same DP target would
also preserve the existing non-DP matrix's target-control structure.

Reports must distinguish target, draft, and deployment-pair guarantees and
retain requested caps alongside achieved accountant results. The confirmed
epsilon grid is {1, 4, 8} per target and equally per member-SFT draft. Basic
composition therefore gives KD-pair epsilon caps {1, 4, 8} and member-pair
caps {2, 8, 16}. These are composition bounds rather than a claim of equal
pair expenditure. The confirmed delta is 5e-6 for each target and member-SFT
draft: the KD pair inherits delta 5e-6, and the member pair has a basic
composition cap of 1e-5. Both deployments use an identically budgeted target
and may share that DP checkpoint.
No per-pair result implies a joint guarantee over the sweep or the full matrix.

The implementation is an opt-in package outside `sd_membership_sft`, because
the existing audit hashes that entire source tree for resumability. It imports
the unchanged fixed-candidate scorer and detector fitter, retains the legacy
checkpoint/config layout inside new run directories, and adds separate DP
provenance. Initial integration covers Qwen3-8B/1.7B with a shared DP target;
the original training matrix and results remain unchanged. DP training uses
fixed-step Poisson document sampling, global per-document clipping (default
1.0), and independently seeded unpublished noise/sampling streams. Completed
stages resume by verified hashes; interrupted optimization restarts that stage
from its pinned base model without reusing an unaccounted partial checkpoint.
