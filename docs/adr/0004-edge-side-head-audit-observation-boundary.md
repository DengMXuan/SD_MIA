# Run EAGLE-3 and MTP heads at the edge with an explicit detector boundary

Status: accepted for the observation boundary and protocol coverage on
2026-09-19, including configurable starting prefixes for natural queries. The
user subsequently confirmed implementation of the consolidated plan in
`experiments/sd_membership_sft/docs/NATURAL_SD_DESIGN.md`.

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
constraint. Choosing this observation boundary does not establish that fixed
candidate probing is supported by a production verifier or that natural SD
trajectories retain the fixed-candidate method's membership signal.

The extension retains separately reported fixed-candidate and natural-SD
experiments for EAGLE-3/MTP and adds natural SD to the current independent-draft
main method. Fixed-candidate results describe controlled candidate probes;
natural-SD results describe draft-generated proposals followed by actual
verifier corrections and continuation. They must not share a reported
effectiveness claim or be pooled into a single protocol metric. The existing
serial pilot is a starting point, not completed support for the current
four-role audit contract or for hidden-conditioned heads.

Natural-SD records support configurable starting prefixes, including 50% and
75% of response tokens, while retaining the original suffix-64 option. This
supersedes the initially selected single-start restriction. Each selected
prefix launches a separate trajectory with fresh state; within that trajectory
generation follows actual proposals and verifier corrections without resetting
to the original response. The suffix withheld at that start is not fed to its
generator or used as a detector feature. Membership remains the original
record's training assignment, not a label attached to generated text. Token
position rules do not imply identical character boundaries across tokenizers.
Multiple starts increase query costs and must remain grouped by original
record for data partitioning, score aggregation and calibration.
