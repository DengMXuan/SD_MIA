# Qwen fixed-candidate, natural-SD and baseline evaluation matrix

Status: confirmed and implemented (2026-09-19). The user approved the consolidated
design, shared 600-record auxiliary budget and both natural sparse directions.
This implementation task covers code, tests and read-only readiness checks;
the full GPU experiment matrix has not been launched.

## Requested scope

- Qwen3-8B/1.7B only: condition seeds 1919/1949/1978, Wiki/News/ArXiv,
  target epochs 1/3, and auxiliary-KD/member-SFT draft variants.
- This is 36 audit configurations over 18 target checkpoints, not the separate
  Qwen/Gemma two-family training matrix.
- Evaluate the fixed-candidate main method, natural-SD main method and existing
  baseline methods; record individual method/configuration results and metrics.
- Include AUC, pAUC up to 10% FPR, TPR at 10% and 1% FPR, and efficiency.
- Preserve the established frozen language-model and nonmember-only detector
  training/selection/calibration constraints. A member-trained draft is an
  experimental draft condition, not permission to train the detector on members.
- Preserve baseline implementations; place adapters/scheduling in the SFT
  experiment package and reuse baseline scoring functions.

## Integration boundaries

- Protocol loading now accepts an explicit `draft_auxiliary_distilled` or
  `draft_member_sft` role, with role-specific provenance/output keys. Invalid
  roles fail instead of silently falling back to the KD checkpoint.
- Baseline has 11 target-only methods: loss, min_k_prob, min_k_pp, recall,
  icp_mia, petal, sead, ws, rs, bt and samia. Target-only results do not depend
  on the draft branch; with identical target/data/settings they can be computed
  once per target condition and referenced by both draft configurations.
- Baseline's current CLI reconstructs an older split. The new matrix must
  use the frozen four-role/shared-split passport, and verify record IDs rather
  than assume that an existing baseline command uses the same audit records.
- Current main-method TPR uses thresholds from independent nonmember
  calibration. Existing baseline reporting uses thresholds derived from the
  evaluation nonmembers. These must be named separately before pooling reports.
- Matrix adapters add synchronized elapsed time and phase-specific accounting
  to both protocols and baselines. Logical generation queries remain separate
  from actual target forward calls and token workload.
- Baselines have different access assumptions (target probabilities or generated
  text) from accept-only SD. Reports must identify each observation setting.

## Confirmed implementation

- Implement a script/CLI with dry-run, run, status and summarize commands.
  Default to one worker on GPU 0; expose an explicit GPU list and allow at most
  one experiment worker per selected GPU. Fail on GPU contention rather than
  displacing existing training processes. Do not automatically launch the full
  matrix as part of implementing the script.
- Retain suffix64, a 32-round cap and one proposal per round as the natural-SD
  default. Fixed-candidate probes retain the complete response with B=2.
  These are different query budgets, so report actual costs rather than label
  them query-matched. Preserve configurable starts/budgets in every manifest.
- Use audit seed 20260914 by default, separately from the three training/data
  condition seeds. Make it configurable and apply identical settings to paired
  draft comparisons; do not implicitly multiply the matrix by replay seeds.
- Keep the existing baseline defaults: K=20%, ReCaLL 4 shots, ICP top-5/min,
  SEAD 50 samples at temperature 1, SaMIA 10 samples, prefix ratio 0.5,
  robustness perturbation rate 0.15, generation batch size 8. Use SDPA for
  supported Qwen models and record precision/backend; no tuning on test labels.
- Report 14 registered method variants per audit configuration: fixed positive
  sparse, natural positive sparse, natural two-sided sparse, and 11 baselines.
  A complete long-format summary therefore has 504 configuration/method rows.
  Target-only baseline results link to one source run per target condition;
  their replicated display rows do not become independent experiments or new
  independent samples in seed aggregation.
- Persist per-record scores/IDs, JSON metrics, long-format CSV and Markdown
  tables. Provide per-condition results and mean/std across the three condition
  seeds for each dataset/epoch/draft/method, keeping datasets and epochs apart.
- Check checkpoint completion and frozen split provenance before each stage.
  Mark unavailable conditions pending; process independent ready conditions.
  Preserve completed method outputs and checked caches on restart. A failed
  task does not discard successful independent tasks; incomplete matrices are
  explicitly labeled and do not receive a successful-completion status.
- Validate adapters, grouping, ROC/tie conventions, independent calibration,
  timing/accounting and resumability with focused tests and a matrix dry-run.
  Any bounded GPU verification is separate from the full 36-configuration run.

Efficiency definition: record preparation (including detector fit), threshold
calibration and test scoring separately; report their total amortized over
4,000 test documents, synchronized GPU elapsed time, throughput, target and
draft query/workload counters, input/generated tokens and peak allocated GPU
memory. Exclude model/data loading, warmup and serialization from the headline
method time, and document those exclusions. Preserve logical generation-query
counts separately from decoder/forward steps and do not call token counters
FLOPs. Each method carries its standalone cost even when underlying results are
reused in a paired display row; separately account for actual matrix execution
cost so reused baseline results and two scores sharing a natural trajectory
are not billed twice. Legacy records without measurements have missing values,
not fabricated zero costs.

pAUC convention: raw area under the empirical ROC on FPR [0, 0.10],
plus area divided by 0.10 for the repository's normalized pAUC. Do not silently
substitute the chance-corrected standardized pAUC used by some libraries.

## Confirmed metric reporting

The user confirmed reporting both empirical test-ROC and independent-calibration
operating points. Store ROC TPR at 10%/1% FPR separately from calibrated TPR,
realized test FPR and uncertainty. Test-derived ROC thresholds are descriptive
evaluation only; they may not select detector orientation, hyperparameters,
query starts or deployed thresholds. The same definitions apply to the main
methods and all baselines.

## Confirmed audit auxiliary budget

All methods share the same 600 audit-auxiliary records for a target condition.
The main method retains 320 detector-fit, 80 validation and 200 threshold-
calibration records. Baselines that need references or fitting may use the
same first 400 records; the same remaining 200 records are reserved exclusively
for independent threshold calibration. Record actual reference counts for
methods that consume fewer than 400. PETAL's regression fitting is method
preparation and must not use the 200 threshold-calibration records.

The 2,000 draft-training auxiliaries are not an additional baseline fitting
pool in this comparison. Already completed target/draft training is a separate
artifact-provenance cost, not part of the shared 600-record audit budget.
All methods evaluate the same complete 2,000 member and 2,000 nonmember IDs.

## Confirmed registered score variants

Keep the fixed-candidate main result as difficulty-conditioned positive sparse
evidence. The user confirmed reporting natural-SD positive sparse and two-sided
sparse evidence as two preregistered variants for every configuration. Do not
select either variant after observing test membership performance. The user's
latest suffix64/32-round query setting can remain the matrix default, while
the script exposes starts and budgets explicitly in run provenance.

## Implementation and validation

- `qwen_audit_matrix.py` and `run_qwen_audit_matrix.sh` expose planning,
  scheduling, status and reporting. The 36 configurations expand to 90 worker
  jobs and 306 unique method outputs, displayed as 504 rows. Each GPU runs one
  worker; contention is checked before dispatch, and interruption reaps children
  before releasing GPU locks.
- `matrix_main.py` uses the existing count TCN for fixed candidates and causal
  GRU for natural trajectories. Detector fitting runs on CPU after releasing
  frozen language models. Defaults: 30 detector epochs, fixed sparse priors
  0.05/0.10/0.25 and tilt strengths 0.5/1/2 (plus negative counterparts for
  the registered two-sided variant). There is no test-label direction selection.
- `matrix_baselines.py` calls the existing baseline scoring implementations.
  It reconstructs the same records and explicitly excludes appended synthetic
  EOS, matching the original-response contract. It measures each baseline
  independently, including separate model forwards where methods could have
  shared intermediate work. No baseline source files are modified.
- `matrix_metrics.py`, `matrix_costs.py` and `matrix_artifacts.py` centralize
  metric definitions, cost accounting and atomic/checksummed result recovery.
  Workers hash checkpoint contents; status/summary verify saved source hashes
  and checkpoint file inventories. Changed inputs require a fresh output root.
- Each report preserves standalone method cost. Shared natural-trajectory cost
  is counted once by execution group in the matrix total; after partial recovery
  the maximum recorded group cost is used. This is measured successful method
  work, not total retry time. Attempt logs separately retain worker wall time,
  including loading and retries; neither sum is parallel matrix elapsed time.
- Read-only readiness checks found all 18 target conditions / 90 jobs ready.
  One real Wiki/epoch1/seed1919 four-role reconstruction verified 4,600 unique
  records and the exact 320/80/200/4,000 partition sizes. This does not establish
  GPU capacity, end-to-end model execution or membership inference performance.
- Tests cover independent calibration, ties/pAUC, role selection, baseline
  reference boundaries, common IDs, shared result/cost aggregation, main-method
  CPU fitting and partial recovery, source checks, GPU preflight and cancellation.
  Full suite: 245 passed (`pytest -q tests experiments/sd_membership_sft/tests`,
  2026-09-19). See README for commands.
