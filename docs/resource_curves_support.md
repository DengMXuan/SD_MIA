# Resource-curve support: definitions and implementation

The current Qwen audit and its frozen source fingerprints remain unchanged.
This work adds callable support for future experiments, without adding a sweep
script, changing any active matrix, collecting GPU observations, or training
language models.

## Agreed experiment definitions

- Query multiplicity is 1, 2, 4, 8, or 16 fresh acceptance judgments per
  supported candidate-token position. Total queries grow with document length;
  the existing fixed-candidate method corresponds to multiplicity 2. These are
  not absolute budgets of 1K–16K judgments per document.
- Detector auxiliary resources total 200–1,600 records. Fitting and calibration
  allocations are independently configurable; fitting includes training and
  validation, split 80%/20% by default. All three allocations count toward the
  total, and remain disjoint from the fixed test set and model adaptation data.
- Extensions use unassigned records from the existing frozen pool. Preserve
  the old manifest and record extensions separately; reject insufficient
  eligible data rather than borrow test records or duplicate auxiliaries.

## Agreed auxiliary-data curves

"Model dataset" here means the detector fitting allocation (training plus
validation), not data for further target or draft language-model adaptation.
Support both independent curves:

| Curve | Fitting | Training | Validation | Calibration | Total auxiliary |
|---|---:|---:|---:|---:|---:|
| Calibration size | 400 | 320 | 80 | 200 | 600 |
| Calibration size | 400 | 320 | 80 | 600 | 1,000 |
| Calibration size | 400 | 320 | 80 | 1,200 | 1,600 |
| Fitting size | 400 | 320 | 80 | 200 | 600 |
| Fitting size | 800 | 640 | 160 | 200 | 1,000 |
| Fitting size | 1,200 | 960 | 240 | 200 | 1,400 |

The 400-fitting/200-calibration point is common to both curves. Within the
calibration curve, keep the exact fitting records and fitted detector fixed,
and grow deterministic nested calibration subsets. Within the fitting curve,
keep the exact calibration records fixed and grow nested training and
validation subsets separately, refitting the detector for each fitting size.
Do not move records between training and validation as fitting size changes.
The test records stay fixed throughout both curves.

Each configuration must have disjoint fitting, calibration, and test records.
The two curves are separate studies: their extension records may overlap
across studies, without implying a combined 1,200-fitting/1,200-calibration
configuration or a need for 2,400 simultaneously reserved auxiliary records.
Record each curve's roles explicitly so a calibration record for one study is
never silently reused by that study's fitted detector.

Default controls: auxiliary-data curves use the current query multiplicity 2;
the query curve holds fitting/calibration at 400/200. These controls isolate
one resource dimension at a time rather than create a full factorial sweep.

## Support interfaces

Implemented as an opt-in library; API usage and output conventions are in
[`experiments/resource_curves/README.md`](../experiments/resource_curves/README.md).

Place the new support in `experiments/resource_curves/`, outside both active
audit runners' source-fingerprint trees. Reuse unchanged protocol and detector
primitives through imports; do not alter their globals or default settings.

Provide configuration validation, deterministic auxiliary selection and
partitioning, multiplicity-aware observation collection, detector fitting,
evaluation, and result persistence as library interfaces. Future scripts will
compose these interfaces into the chosen sweeps. Detector output support must
match the selected count range (0 through the multiplicity), and fitting must
continue to use trusted nonmembers only.

Keep test records and scoring conventions fixed across the resource curve.
Persist the exact record IDs, model/data fingerprints, selected multiplicity,
auxiliary allocations, random seeds, and cache identities. Sampling additional
judgments requires fresh randomness; copying an old bit is never a query.
The existing count-only B=2 archive cannot establish unrecorded judgments for
other multiplicities.

Report ranking and independently calibrated metrics using the existing AUC,
pAUC@10%FPR, TPR@1%/10%FPR conventions. Record actual judgment counts separately
from model forward calls and candidate-position coverage. Measure fresh
collection and fitting/scoring durations separately. Replaying cached data may
support efficacy analysis but must not be presented as freshly measured
collection time for a different budget.

## Verification boundary

Use CPU synthetic tests for count support, independent sampling, partition
disjointness, extension exclusion, deterministic recovery, metrics and timing
provenance. Verify that existing runtime fingerprints and active experiment
outputs remain unchanged. Real model and GPU validation belongs to the future
experiments requested by the user.
