"""Baseline scoring interface shared by matrix, DP and standalone callers.

score_methods consumes frozen records and an explicit reference_cache owned by
one audit condition. It never chooses output paths, loads checkpoints or parses
CLI arguments. WS/RS/BT share generation only within that condition.
"""
from __future__ import annotations
from typing import Any, Iterable
import numpy as np
from experiments.shared.data.data import SFTRecord, prompt_prefix_ids
from experiments.baseline.methods import icp_score, mean_log_likelihood, min_k_plus_plus_score, min_k_prob_score, petal_score, recall_score, rouge1_recall

from .types import AuditRecord
from .scorer import TargetScorer, _response_ids

def _aux_prefix(auxiliary: list[SFTRecord], tokenizer: Any, shots: int) -> list[int]:
    result: list[int] = []
    for record in auxiliary[: max(0, shots)]:
        result.extend(list(prompt_prefix_ids(record, tokenizer)))
        result.extend(_response_ids(record, tokenizer))
    return result



def _jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / max(1, len(a | b))



def _response_prefix_text(record: SFTRecord, tokenizer: Any, ratio: float) -> tuple[list[int], str, str]:
    """Split once for generation inputs, perturbation text, and reference suffix."""
    response = list(record.response_ids)
    cut = min(len(response), max(1, int(round(len(response) * ratio))))
    prefix_ids = list(prompt_prefix_ids(record, tokenizer)) + response[:cut]
    return (
        prefix_ids,
        tokenizer.decode(response[:cut], skip_special_tokens=True),
        tokenizer.decode(response[cut:], skip_special_tokens=True),
    )



def _perturb_words(text: str, kind: str, rate: float, rng: np.random.Generator) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    count = max(1, int(round(len(words) * rate)))
    if kind == "rs":
        for _ in range(count):
            left = int(rng.integers(0, len(words) - 1))
            words[left], words[left + 1] = words[left + 1], words[left]
    else:
        for index in rng.choice(len(words), size=min(count, len(words)), replace=False):
            replacement = words[(int(index) + 1) % len(words)]
            words[int(index)] = replacement
    return " ".join(words)



def score_methods(args, progress, scorer, all_records, auxiliary, tokenizer, methods,
                   *, reference_cache: dict):
    needs_aux = bool({"recall", "icp_mia", "petal"} & set(methods))
    if needs_aux and not auxiliary:
        raise RuntimeError("the requested target-only methods need non-member auxiliary records")
    icp_probe_matrix = (
        scorer.build_probe_matrix(auxiliary) if "icp_mia" in methods else None
    )
    recall_prefix = _aux_prefix(auxiliary, tokenizer, args.recall_shots) if "recall" in methods else []

    petal_slope = petal_intercept = None
    if "petal" in methods:
        calibration_x: list[float] = []
        calibration_y: list[float] = []
        for record in progress.track(auxiliary, "petal calibration"):
            stats = scorer.stats(record, need_similarity=True)
            if stats.log_similarity is not None:
                calibration_x.extend(stats.log_similarity.tolist())
                calibration_y.extend(stats.token_logp.tolist())
        _, petal_slope, petal_intercept = petal_score(
            calibration_x, calibration_y
        )

    scores = {name: [] for name in methods}
    teacher_forced_methods = {
        "loss",
        "min_k_prob",
        "min_k_pp",
        "petal",
        "sead",
        "recall",
        "icp_mia",
    } & set(methods)
    for row_index, audit_record in enumerate(progress.track(all_records if teacher_forced_methods else [], "teacher-forced scoring")):
        record = audit_record.record
        need_similarity = "petal" in methods
        need_sead = "sead" in methods
        stats = scorer.stats(record, need_similarity=need_similarity, need_sead=need_sead)
        base_ll = mean_log_likelihood(stats.token_logp)

        if "loss" in methods:
            scores["loss"].append(base_ll)
        if "min_k_prob" in methods:
            scores["min_k_prob"].append(min_k_prob_score(stats.token_logp, args.k_percent))
        if "min_k_pp" in methods:
            scores["min_k_pp"].append(
                min_k_plus_plus_score(
                    stats.token_logp,
                    stats.expected_logp,
                    stats.variance_logp,
                    args.k_percent,
                )
            )
        if "petal" in methods:
            assert stats.log_similarity is not None
            value, _, _ = petal_score(
                stats.log_similarity,
                stats.token_logp,
                slope=petal_slope,
                intercept=petal_intercept,
            )
            scores["petal"].append(value)
        if "sead" in methods:
            assert stats.sead_log_density is not None
            scores["sead"].append(stats.sead_log_density)
        if "recall" in methods:
            context = scorer.fit_context(recall_prefix, record)
            conditional_ll = scorer.mean_ll(record, context)
            scores["recall"].append(recall_score(base_ll, conditional_ll))
        if "icp_mia" in methods:
            candidates = scorer.top_probe_records(
                record, auxiliary, icp_probe_matrix, args.icp_top_k
            )
            candidate_scores = []
            for candidate in candidates:
                context = scorer.fit_context(_response_ids(candidate, tokenizer), record)
                candidate_scores.append(icp_score(base_ll, scorer.mean_ll(record, context)))
            if args.icp_aggregation == "min":
                scores["icp_mia"].append(float(np.min(candidate_scores)))
            elif args.icp_aggregation == "max":
                scores["icp_mia"].append(float(np.max(candidate_scores)))
            else:
                scores["icp_mia"].append(float(np.mean(candidate_scores)))

    generation_methods = {"ws", "rs", "bt", "samia"} & set(methods)
    if generation_methods:
        generation_inputs: list[list[int]] = []
        prefix_texts: list[str] = []
        suffixes: list[str] = []
        max_news: list[int] = []
        for audit_record in all_records:
            generation_input, prefix_text, suffix = _response_prefix_text(
                audit_record.record, tokenizer, args.prefix_ratio
            )
            generation_inputs.append(generation_input)
            prefix_texts.append(prefix_text)
            suffixes.append(suffix)
            max_news.append(
                max(1, len(tokenizer(suffix, add_special_tokens=False).input_ids))
            )

        def batches(values: list[Any], stage: str) -> Iterable[tuple[int, list[Any]]]:
            for start in progress.track(list(range(0, len(values), args.generation_batch_size)), stage, unit="batches"):
                yield start, values[start : start + max(1, args.generation_batch_size)]

        def trim(values: list[int], length: int) -> list[int]:
            return values[:length]

        if "samia" in generation_methods:
            for start, batch_inputs in batches(generation_inputs, "samia"):
                end = start + len(batch_inputs)
                generated = scorer.generate_batch(
                    batch_inputs,
                    max(max_news[start:end]),
                    args.seed + start,
                    sample=True,
                    num_return_sequences=args.samia_samples,
                )
                for offset, rows in enumerate(generated):
                    suffix = suffixes[start + offset]
                    generated_text = [
                        tokenizer.decode(
                            trim(ids, max_news[start + offset]),
                            skip_special_tokens=True,
                        )
                        for ids in rows
                    ]
                    scores["samia"].append(
                        float(
                            np.mean(
                                [rouge1_recall(text, suffix) for text in generated_text]
                            )
                        )
                    )

        if {"ws", "rs", "bt"} & generation_methods:
            if "texts" in reference_cache:
                baseline_texts = reference_cache["texts"]
                if len(baseline_texts) != len(all_records):
                    raise ValueError("robustness reference cache has the wrong record count")
            else:
                baseline_texts = [""] * len(all_records)
                for start, batch_inputs in batches(generation_inputs, "shared robustness reference generation"):
                    end = start + len(batch_inputs)
                    generated = scorer.generate_batch(
                        batch_inputs,
                        max(max_news[start:end]),
                        args.seed + 10_000 + start,
                        sample=False,
                    )
                    for offset, rows in enumerate(generated):
                        baseline_texts[start + offset] = tokenizer.decode(
                            trim(rows[0], max_news[start + offset]),
                            skip_special_tokens=True,
                        )
                reference_cache["texts"] = baseline_texts

        for kind in ("ws", "rs"):
            if kind not in generation_methods:
                continue
            perturbed_inputs: list[list[int]] = []
            for row_index, audit_record in enumerate(all_records):
                rng = np.random.default_rng(args.seed + 20_000 + row_index)
                perturbed_text = _perturb_words(
                    prefix_texts[row_index], kind, args.perturbation_rate, rng
                )
                perturbed_inputs.append(
                    list(prompt_prefix_ids(audit_record.record, tokenizer))
                    + list(tokenizer(perturbed_text, add_special_tokens=False).input_ids)
                )
            for start, batch_inputs in batches(perturbed_inputs, kind):
                end = start + len(batch_inputs)
                generated = scorer.generate_batch(
                    batch_inputs,
                    max(max_news[start:end]),
                    args.seed + 30_000 + start,
                    sample=False,
                )
                for offset, rows in enumerate(generated):
                    perturbed_text = tokenizer.decode(
                        trim(rows[0], max_news[start + offset]),
                        skip_special_tokens=True,
                    )
                    scores[kind].append(
                        rouge1_recall(perturbed_text, baseline_texts[start + offset])
                    )

        if "bt" in generation_methods:
            rewrite_inputs = [
                tokenizer(
                    "Rewrite the following passage in different words while preserving its meaning.\n"
                    f"Passage: {prefix_text}\nRewrite:",
                    return_tensors="pt",
                ).input_ids[0].tolist()
                for prefix_text in prefix_texts
            ]
            rewritten: list[list[int]] = [[] for _ in all_records]
            for start, batch_inputs in batches(rewrite_inputs, "bt rewrite"):
                end = start + len(batch_inputs)
                generated = scorer.generate_batch(
                    batch_inputs,
                    max(max_news[start:end]),
                    args.seed + 40_000 + start,
                    sample=False,
                )
                for offset, rows in enumerate(generated):
                    rewritten[start + offset] = trim(
                        rows[0], max_news[start + offset]
                    )
            bt_inputs = [
                list(prompt_prefix_ids(audit_record.record, tokenizer)) + rewritten[index]
                for index, audit_record in enumerate(all_records)
            ]
            for start, batch_inputs in batches(bt_inputs, "bt scoring"):
                end = start + len(batch_inputs)
                generated = scorer.generate_batch(
                    batch_inputs,
                    max(max_news[start:end]),
                    args.seed + 50_000 + start,
                    sample=False,
                )
                for offset, rows in enumerate(generated):
                    scores["bt"].append(
                        rouge1_recall(
                            tokenizer.decode(
                                trim(rows[0], max_news[start + offset]),
                                skip_special_tokens=True,
                            ),
                            baseline_texts[start + offset],
                        )
                    )

    return scores
