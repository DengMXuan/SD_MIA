# Priority-ordered nonmember-only optimization validation

Registered before examining the new test scores, 2026-09-17. This remains an
exploratory follow-up because the existing benchmark has been inspected.
Target and auxiliary-distilled draft checkpoints remain frozen. No member or
synthetic-member labels train/select detectors, policies, aggregation or cutoffs.

Conditions: Wiki/News/ArXiv, saved SFT epochs 1/3; seeds 20260914–20260916.
Whole response, appended EOS excluded. Fixed-candidate verifier replay, not
natural SD. Reference split stays 320 train + 80 validation nonmembers; test
stays 400 members + 400 nonmembers. Default calibration is the original 200.
All comparisons use paired records and the same observed bits within phase.

1. Sequence: retain the static q/position TCN, add a separate causal history
   branch with shifted one-hot accept counts and left-padded dilated convolutions.
   Predict current count from static q plus strictly previous counts. No current
   or future feedback enters the prediction. Fit by nonmember count likelihood,
   select by held-out nonmember NLL. Compare global fixed tilts .5/1/2 with the
   saved original_global model on the same B=2 bits. The autoregressive
   alternative uses the observed history under both hypotheses.
2. Features and uncertainty: a) add draft normalized entropy, log-rank and top1
   vs top2 logit margin to q/position. Existing frozen-checkpoint feature caches
   are reused only with checked identities/lengths/provenance. Their q differs
   from old replay q, so regenerate simulator bits using feature q and cached
   target p, and refit a q-only baseline under that identical protocol. Collect
   the missing News3/ArXiv3 feature caches from frozen drafts. This phase is
   separate from the historical replay. b) original replay: fit two extra
   initializations to exactly the same training bits, combine with the saved
   base model into a three-model PMF average. Also test a fixed uncertainty
   discount 1/(1+5*MI/log(3)), with MI=H(mean PMF)-mean H(PMF). No extra verifier
   training queries are consumed by reusing observations; detector compute grows.
3. Calibration: freeze original_global scores. Extend the original 200 with
   1000 unused nonmembers, excluding train/validation/test. Report nested
   sizes 200/400/800/1200. Fit two separate binary Mondrian partitions using
   reference medians of length and mean logq; calibrate each within its group
   at n=1200. Report TPR and actual FPR overall and per group. Empty calibration
   groups give p=1. No claim of distribution-free conditional guarantees beyond
   exchangeability within the chosen groups.
4. Sparse and allocation: a) original null PMF, fixed independent sparse
   alternative fractions .05/.10/.25 and count tilts .5/1/2, equal mixture over
   all nine combinations. Compare with dense global evidence. b) fixed-q pilot
   at every position, then a second query at floor(L/2) positions: posterior
   predictive entropy versus deterministic seeded-uniform positions. Same
   L+floor(L/2) budget and coupled B=2 bits. The scorer uses a coherent latent
   Bernoulli mixture with fixed logit shifts .5/1/2 and posterior updating;
   score the pilot AND second queries. Compare policies within this scorer,
   not directly against a different-budget full-query statistic.

All variants and failures are reported, no member-selected winning combination.
Bootstrap test records jointly across seeds and checkpoints that share records.
Report AUC, pAUC (ROC area on FPR 0–.1 divided by .1), actual FPR, nonmember NLL,
query costs and model fitting costs. Intervals condition on fitted models and
are not corrected for multiple comparisons. Repeated seeds are not new samples.
