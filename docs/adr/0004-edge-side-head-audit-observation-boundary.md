# Run EAGLE-3 and MTP heads at the edge with an explicit detector boundary

Status: observation boundary accepted on 2026-09-19. The natural-generation
extension was removed on 2026-09-25; the edge-side head boundary below remains
in force for the main method. See the current
[Qwen matrix design](../../experiments/sd_membership_sft/docs/QWEN_AUDIT_MATRIX_DESIGN.md).

For the EAGLE-3/MTP membership-audit extension, the client runs the prediction
head and the cloud supplies the target hidden states needed to produce its
proposals. The client computes draft probabilities and difficulty features
locally and receives protocol acceptance feedback. This preserves the
edge-side white-box drafter setting while acknowledging that these heads
depend on target-provided representations.

The detector may consume draft-distribution features and acceptance feedback,
but not raw target hidden states or target probabilities. Reports must
distinguish this feature restriction from the attacker's actual access to the
hidden states; this is not a deployment with acceptance feedback as the only
cloud-to-client information. Server-side heads with exported probabilities,
and acceptance-only interfaces without local draft probabilities, are distinct
observation settings and are not the selected design.

Existing target and head checkpoints remain frozen during audit experiments.
Detector fitting, selection and calibration retain the trusted-nonmember-only
constraint. This boundary does not establish that candidate probing is
supported by a production verifier. Reports describe the implemented B=2 audit
and may not present its measured costs as production speculative-decoding speedup.
