# Frozen-model direction validation, 2026-09-17

All target and auxiliary-distilled draft checkpoints stay frozen. Detectors
fit 320 trusted nonmembers, select checkpoints on 80 other nonmembers, and
calibrate on 200 disjoint nonmembers. Registered test sets contain 400 members
and 400 nonmembers per condition. No target-member labels select models,
hyperparameters, query actions or stopping boundaries.

Primary conditions: Wiki/News/ArXiv, target SFT epochs 1/3. Replay/training
seeds: 20260914, 20260915, 20260916. Report all conditions, including failures.

1. Conditional null: replicate the existing whole-document B=2 experiment
   with the two additional seeds. Keep architecture and training settings.
2. Counterfactual: same final 64 response candidates under original prefix
   versus instruction + last 32 response-context tokens. Independently sample
   both views under their recomputed p and q. Compare original B with paired
   B/2+B/2 for B=2 and B=8. Fit both predictors only on real nonmembers.
3. Active protocol design: use the same final 64 candidates and frozen cache.
   A neural latent-probability prior is trained through multi-q accept-bit
   likelihoods on nonmembers, not exact target probability regression.
   Proposal levels are lambda=0,.25,.5,.75,1 with q_lambda(y)=q0(y)^(1-lambda),
   realizable by a mixture of the draft and a point mass on the candidate.
   Positive shifts .5,1,2 define fixed hypothetical alternatives, not member
   training examples. Compare fixed-q, uniform ladder, adaptive-q, adaptive
   position, and joint JS-based design at B=2/8. Stop-rule calibration uses
   nonmember maxima over the entire registered query path.
Acceptance criteria: matched-budget paired AUC/pAUC differences, per-condition
consistency, and actual FPR alongside nominal low-FPR TPR. Bootstrap the same
record indices across seeds and checkpoints with identical test records; intervals condition on fitted
models. Prior single-seed test results have already been inspected, so this
is an exploratory follow-up, not an untouched confirmatory benchmark.
Intervals are not adjusted for multiple comparisons. The normalized pAUC@10%
integrates ROC area over FPR 0–0.10 and divides by 0.10 (random expectation
0.05), rather than the chance-corrected standardized pAUC used by some packages.

For counterfactuals, shorter context also changes positions and general
prediction difficulty. An advantage alone is not proof of causal memorization.
For active q, distinguish a simulator-valid normalized proposal from an API
that actually permits client-selected proposal distributions.
