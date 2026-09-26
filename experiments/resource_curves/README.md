# Resource-curve support

This is an opt-in Python library and Qwen3 ablation runner, outside the existing
Qwen and cross-model runners' source-fingerprint trees. The new scripts schedule
their own matrices. Baseline implementations are unchanged.

**可执行消融脚本与多 GPU 用法： [QWEN_ABLATIONS.md](QWEN_ABLATIONS.md)**。
提供辅助数据量、News→Wiki/Arxiv 分布变化、B=1/2/4/8/16 三个独立入口，
均为 Qwen3、epoch 1、KD 草稿，默认三个对应 seed。

## Experiment units

`QUERY_MULTIPLICITIES = (1, 2, 4, 8, 16)` means fresh accept/reject judgments
**per supported candidate position**. A document's query count is its supported
position count times the multiplicity. Counts are distinct from actual model
forward calls and network requests. The fixed candidates, original prefixes,
draft difficulty features, positive sparse score, and metric definitions match
the current main method. At B=2 the sampling stream and detector algorithm
retain the reference behavior. Target and draft models remain frozen.

`AuxiliaryBudget(fitting, calibration, validation=None)` independently sets
detector fitting and calibration sizes. Fitting includes training and
validation (80/20 by default; validation defaults to `fitting // 5`). The total
must lie within 200–1600. No members or synthetic members train/select the
detector. The test set is never a source of auxiliary records.

| Curve | Fitting | Training | Validation | Calibration | Total |
|---|---:|---:|---:|---:|---:|
| Calibration | 400 | 320 | 80 | 200 | 600 |
| Calibration | 400 | 320 | 80 | 600 | 1000 |
| Calibration | 400 | 320 | 80 | 1200 | 1600 |
| Fitting | 400 | 320 | 80 | 200 | 600 |
| Fitting | 800 | 640 | 160 | 200 | 1000 |
| Fitting | 1200 | 960 | 240 | 200 | 1400 |

There are five distinct auxiliary settings. Use B=2 for both auxiliary curves;
use fitting/calibration 400/200 for the query curve. These are controls, not a
full factorial sweep. The API allows other valid allocations.

Within each study, roles have stable nested subsets and the test IDs stay
fixed. Calibration growth keeps the fitted detector unchanged. Fitting growth
keeps calibration unchanged and refits the detector; earlier training samples
never move into validation. Build the two studies separately: their extension
IDs may overlap, without requiring a joint 1200+1200 allocation.

## Compose an experiment through the library

1. Obtain the frozen `shared` split manifest, the original `base_prepared`
   records/tokenizer using the existing model-specific loader, the matching
   frozen `adapter`, and the model/data fingerprint dictionary `sources`.
   Recompute fingerprints before reuse; checkpoint *paths alone* do not prove
   weight identity. Loading/warmup are the future caller's responsibility.
2. `auxiliary.select_extension(pool_path, shared_manifest_path, tokenizers,
   count=1000)` verifies the frozen pool and selects unassigned nonmembers.
   Its seed defaults to the frozen shared split's seed; an explicit different
   seed is rejected. It no longer defaults to the historical seed 20260922.
   `tokenizers` maps **all exact tokenizer source names in the shared manifest**
   to locally loaded tokenizer objects. Selection excludes every original
   member/nonmember/draft-auxiliary/audit-auxiliary record, and checks raw text,
   truncated token IDs, token bands and 13-gram near-duplicates against the
   original assignments and selected extensions. Insufficient eligible data
   fails explicitly. This function does not download tokenizers or write to
   the original pool/split. `save_extension` stores a separate manifest.
3. Build the study using `build_study`; materialize extensions for the current
   tokenizer with `extension_records`; call `prepare_study` to select exactly
   the union of records needed for that study.
4. Collect observations, fit and evaluate a chosen point using the interfaces
   below. Future scripts can loop over the points; this package never starts
   those loops automatically.

```python
from experiments.resource_curves import calibration_curve, fitting_curve
from experiments.resource_curves.auxiliary import extension_records
from experiments.resource_curves.partitions import build_study, prepare_study, save_study
from experiments.resource_curves.observations import collect_observations
from experiments.resource_curves.detector import fit_detector
from experiments.resource_curves.evaluation import evaluate
from experiments.resource_curves.storage import CACHE_ROOT, DATA_ROOT, RUN_ROOT, digest

# shared, extension, base_prepared, tokenizer_source, adapter, sources and
# condition_key are supplied by the future caller. condition_key identifies
# dataset, training seed, epoch, model pair and draft role.
# Check that the model passport's config.seed and config.data_seed both equal
# shared["seed"], and that its shared-split checksum matches, before composing.
study = build_study(shared, extension, calibration_curve(), name="calibration")
save_study(DATA_ROOT / condition_key / study["name"], study)
extra = extension_records(extension, base_prepared.tokenizer, tokenizer_source)
prepared = prepare_study(base_prepared, extra, extension, study)

observations = collect_observations(
    prepared, adapter, CACHE_ROOT / condition_key / "calibration_b2",
    sources={"frozen": sources, "extension_digest": digest(extension)},
    multiplicity=2, seed=shared["seed"],
)
point = study["points"][0]
detector = fit_detector(observations, point, CACHE_ROOT / condition_key / "detectors")
report = evaluate(observations, point, detector,
                  RUN_ROOT / condition_key / "calibration_fit400_cal200")
```

For the fitting curve replace `calibration_curve()` with `fitting_curve()` and
use distinct study/collection/result directories. For the query curve build a
study containing only `AuxiliaryBudget(400, 200)` and collect each multiplicity
into its own directory. All APIs are explicit about the selected budget.

The extension, study, every point and prepared records carry the same condition
seed. `prepare_study` supplies `prepared.condition_seed`; observation collection
requires it and rejects a different seed. Detector fitting and metric bootstrap
inherit the observation seed and reject overrides. This preserves the mapping
1919→1919, 1949→1949, 1978→1978 across stages. Older extension/study artifacts
without this seed contract must be rebuilt in a new dedicated output folder.

New outputs default to dedicated roots:

- `artifacts/audits/resource_curves_v1/splits/`: extension and study manifests.
- `artifacts/audits/resource_curves_v1/intermediate/`: observation records and detectors.
- `artifacts/audits/resource_curves_v1/tasks/`: reports and scored records.

The library refuses existing unrelated output directories and old artifact
roots, including aliases through symlinks. Reuse of the same collection/result
directory with changed settings or source fingerprints fails. Completed
record observations resume with checksums; an interrupted unfinished record
is recollected. A completed detector is reused only for identical fitting
observations/roles, B and training settings. An interrupted detector fit starts
again; there are no per-epoch training checkpoints.

## Outputs and interpretation

- Observations contain draft features and individual binary judgments, not
  target probabilities, probability gaps, target hidden states or generated
  target tokens. Metadata records coverage, actual judgments, forward counters,
  measured synchronized collection times, hardware and model/data provenance.
- `FIT.json` and `detector.pt` store fitting identity, train/validation IDs,
  normalization, history, early stopping, checksums and measured duration.
  Calibration IDs, test IDs and their outcomes are excluded from fit identity.
- `REPORT.json` reports AUC, raw/normalized pAUC at 10% FPR, empirical ROC TPR
  at 1%/10% FPR, and independently calibrated deployment TPR/**actual FPR**.
  `scores.npz` retains scores, IDs, labels, partitions and test conformal p-values.
- A calibration-only curve should not change raw-score AUC/pAUC or empirical
  ROC TPR when fitting and test records are fixed. Its effect is assessed using
  calibrated TPR, actual FPR and conformal resolution. Calibration has no role
  in learning or choosing the raw-score orientation.
- Timings separate collection, fitting, calibration scoring and test scoring.
  Stored collection/fit time is the original measured cost, not a claim that
  cache replay did that work again. GPU synchronization is used when applicable.
  Loading, caller warmup, archive I/O and metric/bootstrap reporting are excluded.

`observations.view(B)` can use the first B **actually recorded** bits for
efficacy-only comparisons. It cannot increase B. A B=2 view of a B=16 archive
has no measured B=2 collection runtime: matching collection and total latency
are reported as `null`, with B=16 origin costs labeled separately. To measure
the runtime curve, collect each budget separately with the same hardware and
warmup policy. Forward calls may stay constant because full-context model
rows are batched and reused for verifier randomness; timing here is the local
protocol reference implementation, not a measurement of deployed network RTT.

The previous count-only B=2 archive cannot be imported as individual bits or
expanded into other budgets. It is left untouched. This first API uses pooled
independent calibration, matching the active fixed-main audit; it does not
silently introduce another scoring/calibration variant.

## Validation

CPU-only synthetic tests cover B=2 reference equivalence, all five count
supports, fresh/nested bits, query accounting, stable and disjoint partitions,
fitting-cache independence from calibration, source/hash rejection, interrupted
collection recovery, immutable extension selection and runtime provenance.

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m pytest \
  -q -p no:cacheprovider tests/resource_curves
```

No GPU/model experiment has been run as part of this change.
The read-only data preflight and its scope are recorded in
[DATA_PREFLIGHT.md](DATA_PREFLIGHT.md); raw unused pool size alone is not a
guarantee of post-filter availability. The library can be called independently;
the new Qwen scripts consume the existing checked extension manifests.
