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
4. Serial feedback: evaluate first-rejection censoring explicitly; unvisited
   candidate positions are missing, never rejected. Also collect a small
   genuine sampling-based SD experiment on saved Wiki epoch-1/3 checkpoints:
   gamma=4, 8 verifier rounds, original prefix before the final 64 tokens.
   Target correction/bonus tokens update protocol state but are not detector
   inputs. Only local q/entropy, round boundaries and reached accept bits are
   exported. Fit a causal nonmember hazard model; compare against acceptance
   rate on identical transcripts. This changes the observation experiment and
   must not be pooled with fixed-candidate results.
   This natural-SD pilot uses seed 20260914 and 1,400 selected records per
   checkpoint (400 reference nonmembers, 200 calibration nonmembers, 800 test
   records). It is a two-checkpoint mechanism test, not the six-condition,
   three-seed matrix used for the cheaper cached/paired directions.
   Report summed hazard evidence and a per-reached-decision normalization
   alongside raw acceptance rate; this controls variable transcript length
   without selecting an aggregation rule using member test outcomes.
   For sampled proposals, expected acceptance equals 1-TV(p,q), so membership
   need not increase acceptance. Before inspecting natural-SD metrics, fix
   negative and two-sided hazard alternatives too, and report all directions.
   Compare them with rejection rate and the absolute deviation from nonmember
   validation acceptance rate, respectively; do not select the best direction
   on members. All thresholds still use the separate nonmember calibration set.

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
that actually permits client-selected proposal distributions. For natural SD,
the runtime needs correction tokens for synchronization, but the detector
is restricted to acceptance feedback and local draft features.
