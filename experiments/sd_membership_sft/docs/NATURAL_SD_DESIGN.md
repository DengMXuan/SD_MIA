# Natural SD and hidden-conditioned head audit extension

Status: confirmed by the user and implemented on 2026-09-19. The independent
Qwen draft path passed a real-model runtime smoke test. EAGLE-3/MTP adapters
have protocol/interface tests only; their new checkpoints are not ready for
real-model validation. This is not a claim of membership effectiveness.

## Accepted scope

- Preserve fixed-candidate audits and add separately reported natural SD for
  independent drafts, EAGLE-3 heads and native MTP heads.
- Run heads at the edge using cloud-supplied target hidden states. Only local
  draft-distribution features and acceptance feedback enter the detector.
- Keep existing language-model and head checkpoints frozen. Restrict changes
  to the SFT experiment implementation and its documentation/tests; preserve
  baseline implementations and existing checkpoints/results.
- Support one or several starting prefixes per record, including response-token
  fractions such as 0.50 and 0.75 and the historical response[:-64] option.
  Each trajectory starts with the original formatted prompt plus the selected
  response prefix, uses fresh generation state and its own reproducible random
  stream, and never restores the original response after a rejection.
  Record membership remains the audit label.

## First implementation

- Use stochastic rejection sampling at temperature 1 without top-k/top-p
  truncation. Apply the actual proposal distribution in acceptance and residual
  correction; handle all-accepted bonus tokens and EOS explicitly.
- Expose the round cap per starting prefix; 32 rounds per prefix is the
  configurable initial setting, not an empirically selected optimum. With two
  starts this permits 64 total rounds per record. Also record the document's
  total cap and actual costs so equal-total-budget comparisons can allocate,
  for example, 16 rounds to each of two starts versus 32 to one start. Use one
  proposal per round initially. This is a single-step protocol experiment, not a full
  EAGLE-3 tree benchmark. Preserve the independent-draft serial pilot as a
  distinct historical configuration. Expose budgets in the new interface and
  record actual rounds, proposal decisions, generated tokens, prefix work and
  transferred hidden-state volume where measurable; equal round caps do not
  establish equal computational or communication cost.
- Introduce protocol adapters for independent drafts and hidden-conditioned
  heads. Validate EAGLE draft-to-target support mapping and MTP prediction
  offsets. Use a correctness-first context reconstruction if a model's cache
  cannot safely roll back; report that execution mode and its cost.
- Keep natural observations in a separate, versioned schema. Record reached
  decisions, record and trajectory IDs, starting positions, round positions,
  termination and q-derived difficulty features;
  do not turn unobserved nodes into rejections or save target scores/raw hidden
  states in detector archives. Validate checkpoint, split and configuration
  provenance on collection, resume and evaluation.
- Retain the current four-role contract: 600 independent audit auxiliaries
  split into 320 detector-fit, 80 validation and 200 calibration records;
  2,000 members and 2,000 nonmembers remain evaluation-only. Reject incompatible
  checkpoint/split provenance instead of borrowing test nonmembers.
- Adapt conditional prediction to causal binary acceptance observations.
  Compare difficulty-conditioned global and sparse evidence against simple
  acceptance-rate and q-only baselines. Preregister positive, negative and
  two-sided alternatives; calibrate/report each without selecting a winner
  using test labels. Preserve independent document-level calibration. All
  trajectories from one record stay in the same partition. Report each fixed
  starting-position configuration separately; any multi-start score must use
  a preregistered aggregation rule and be calibrated as a document score on
  nonmembers collected under the same multi-start budget. Never treat starts
  as independent documents or choose the best start from member test results.
- For fixed-candidate EAGLE observations, represent vocabulary support
  explicitly. An out-of-support token cannot be treated as a valid naturally
  proposed token or assigned a fabricated finite q for the legacy B=2 schema.
  Exclusions and effective candidate counts must be reported.
- Resolve fraction r to floor(r * response_token_count), excluding prompt and
  appended EOS. Require a nonempty response prefix and withheld suffix; reject
  invalid or duplicate resolved positions during preflight with IDs/reasons.
  The suffix-64 option requires more than 64 response tokens. Do not silently
  change query policy for short records. Token fractions do not establish
  identical character boundaries across different tokenizers.

## Validation and reporting

Exercise rejection/continuation, all-accepted bonus handling, EOS, token
alignment, vocabulary support, observation boundaries, split isolation and
resume validation with focused tests. Implement code first. Real-model smoke
validation is limited to the old Qwen3-8B/1.7B pair when compatible artifacts
are available. Check its split provenance; a legacy-split smoke experiment
must be labeled separately and cannot establish a four-role-contract result.
EAGLE-3/MTP receive protocol/interface tests only until the new checkpoints
finish training, and must be marked as not yet validated on real models.
Smoke results verify operation; they do not establish membership effectiveness.
Report fixed-candidate and natural-SD metrics separately, including AUC,
low-FPR TPR, realized FPR, uncertainty and actual query costs. Initial scope
excludes a full tree-search engine and automatic exhaustive matrix execution.

## Implementation and validation record

- `sd_protocol.py`: frozen full-context adapters, vocabulary mapping, native
  MTP alignment, single-proposal natural sampling and fixed-candidate probes.
- `protocol_models.py`: completed checkpoint loading and four-role head split
  reconstruction with training-manifest and audit checks.
- `collect_protocol_observations.py` / `protocol_archive.py`: multi-start
  collection, observable-only archives, source fingerprints and checked resume.
- `protocol_accept_only.py`: causal binary GRU for natural SD, current count
  TCN for fixed probes, signed global/sparse scores, per-start and combined
  document-level nonmember calibration. Combined scores sum each fixed
  alternative's evidence across starts before mixing alternatives; no start
  or direction is selected from test labels.
- `tests/test_protocol_audit.py`: includes an installed-MTP-forward test that
  verifies the ignored target placeholder cannot change next-token logits.

Validation: `python -m pytest -q tests experiments/sd_membership_sft/tests`
completed with 229 passing tests. A frozen old Wiki epoch-1 Qwen3-8B/1.7B pair
ran on GPU 0 with a synthetic, membership-unlabeled text, starts 0.5/0.75 and
two rounds each. The archive is under
`experiments/results/sft_runs/natural_sd_extension_smoke_20260919/` and is
explicitly rejected by the membership evaluator. No target/head training was
started, and no EAGLE-3/MTP checkpoint was run. The full-context implementation
measures forward calls and input-token work; reported hidden-state bytes are
logical tensor payloads, not measured network traffic or deployment latency.
