# Target-only MIA baselines

This package evaluates the controlled SFT run with the saved, fine-tuned
target model only. It never loads a draft checkpoint, a reference model, or a
separately scored pre-SFT target. For a LoRA run, Transformers/PEFT must
instantiate the adapter's base weights to reconstruct the fine-tuned target;
the base is not scored as a second model and is not used as a reference.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 UV_CACHE_DIR=/tmp/sd-mia-uv-cache \
  uv run --no-sync python -m experiments.baseline.run \
  --run-dir experiments/results/sft_runs/newstection_qwen3_8b_epoch1 \
  --gpu 0 \
  --methods all \
  --output-dir experiments/results/baseline/newstection_epoch1
```

Use a comma-separated subset such as `--methods loss,min_k_prob,min_k_pp`
when generation-based methods are not needed. The output contains
`baseline_metrics.json`, `baseline_scores.npz`, and `BASELINE_RESULTS.md`.

## Implemented methods

All output scores are oriented so larger means “more likely member”.

| Method | Target-only implementation in this repository |
|---|---|
| Loss | Mean teacher-forced log-likelihood over the SFT response (prompt masked). |
| Min-K% Prob | Mean log-probability of the lowest `--k-percent` response tokens. |
| Min-K%++ | Lowest-K% standardized token log-probabilities using the target distribution's own mean and variance. |
| ReCaLL | Relative LL after prepending `--recall-shots` held-out auxiliary records; no reference model. |
| ICP-MIA | Optimization-gap proxy against the top-K auxiliary non-member probes, ranked in the fine-tuned target's own input-embedding space. |
| PETAL | PETAL's similarity-to-log-probability regression, recalibrated on auxiliary records scored by the fine-tuned target instead of a surrogate. Similarity uses target input-embedding cosine. This is a target-only adaptation, not the official surrogate-based threat model. |
| SEAD | Official surrogate-free frequency estimator: Monte Carlo next-token density from the fine-tuned target at `--sead-samples`. |
| WS / RS / BT | Label-only robustness probes. WS performs target-only lexical replacement, RS random adjacent swaps, and BT uses the target itself to generate a paraphrase before re-querying it. |
| SaMIA | Target-only sampled continuation overlap with the held-out response suffix, using the official ROUGE-1 recall statistic. |

WS/RS/BT and SaMIA are deliberately target-only adaptations for this SFT
protocol: they do not download a translator, surrogate, or original model.
Their output-channel scores should not be described as exact reproductions of
the original external-model label-only threat model.

## Source implementations read

The implementation was checked against the following first-party/public
artifacts:

- [ReCaLL official repository](https://github.com/ruoyuxie/recall):
  `src/run.py`, especially `get_ll`, `get_conditional_ll`, and the
  `base_ll - conditional_ll` score.
- [Min-K% Prob official repository](https://github.com/swj0419/detect-pretrain-code):
  `src/run.py`, including lowest-token probability aggregation and its
  reference-model baseline (the latter is intentionally omitted here).
- [Min-K%++ official repository](https://github.com/zjysteven/mink-plus-plus):
  `run.py`, including vocabulary expectation/variance normalization and
  lowest-K% aggregation.
- [ICP-MIA official repository](https://github.com/RPI-DSPlab/ICP-MIA):
  `icp_mia_attack.py` and `baseline/attacks/loss.py`; its reference-data and
  precomputed perturbation branches are replaced here with the run's
  auxiliary non-member records.
- [SaMIA official repository](https://github.com/nlp-titech/samia):
  `src/sampling.py` and `src/eval_samia.py`.
- [PETAL official repository](https://github.com/QingHuan-6/AIModelSecurityEvaluationPlatform/tree/main/PETAL)
  and [public artifact](https://zenodo.org/records/14725819): `run.py`,
  `utils.py`, and `vectors.py`. The artifact uses a surrogate regression;
  the target-only recalibration above is an explicit adaptation and must not
  be reported as exact PETAL.
- [SEAD author-associated repository](https://github.com/clearloveclearlove/SEAD):
  `run_sead.py` implements the target-only frequency estimator; equations
  (4)--(7) of the [paper](https://aclanthology.org/2026.findings-acl.337/)
  define its Monte Carlo density and optional semantic-aware extension. This
  runner uses frequency mode and does not load the optional NLI classifier.

The exact repository status, formulas, and unavailable official-code cases
are recorded in the companion research note under `research/`.
