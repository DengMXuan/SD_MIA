# Fixed factorial combination validation

Registered 2026-09-17 before evaluating combined scores. Exploratory follow-up
on the previously inspected benchmark, not a fresh confirmatory test.

Six conditions: Wiki/News/ArXiv × saved target epochs 1/3; seeds
20260914, 20260915, 20260916. Every cell uses whole-document fixed-candidate
B=2 accept-only replay with appended EOS excluded. Use the feature-consistent
q and exactly the same accept bits, record identities and test split in all
cells. Never compare historical-q scores as if their inputs were identical.

2 × 2 × 3 = 12 configurations for each condition/seed, 216 evaluated cells:
- Inputs: q/position versus q/position + draft entropy/log-rank/top1-top2 margin.
- Scores: global versus independent sparse mixture (rho=.05/.10/.25 and
  positive count tilts .5/1/2, all nine pairs equally weighted).
- Calibration: original pooled 200; expanded pooled 1200; expanded 1200 with
  fixed binary mean-logq difficulty grouping (reference-NM median).

Reuse both saved detectors from priority_validation/features. Their fits used
320 trusted nonmembers and 80 validation nonmembers. No language-model or
small-detector training, no member-based parameter/weight/sign/threshold
selection. Regenerate both scores from each saved detector's conditional PMF,
check that global scores reproduce their saved results, then independently
calibrate each score. Expanded calibration is disjoint from train/validation
and the 400-member + 400-nonmember test set.

Report all twelve cells, not only a selected winner. Primary candidate fixed
in advance: difficulty inputs + sparse score + grouped 1200 calibration.
Primary comparisons: full candidate vs q/global/200, and dropping features,
sparsity or grouped/expanded calibration one at a time. Include pooled 1200
to separate larger calibration from grouping. Raw AUC/pAUC describe the four
scores; TPR/actual FPR at nominal 1/5/10% describe all calibrated decisions.
Report conditional FPR and per-condition results, plus actual decision costs.

Quantify feature×sparsity interaction on AUC and pAUC as
(F_sparse - Q_sparse) - (F_global - Q_global). A positive value suggests
super-additivity on that metric; it does not establish causal independence.
Paired stratified test-record bootstrap with 500 replicates reuses draws across
seeds/checkpoints sharing records. Intervals condition on fitted models and
calibration pools and are not adjusted for multiple comparisons. Repeated
seeds are not new independent samples. No exact 1% FPR claim based on nominal
threshold alone. Calibration adds reference queries but not per-test-record
queries; reuse local caches for this offline experiment.
