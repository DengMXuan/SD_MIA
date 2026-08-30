from __future__ import annotations

import numpy as np
import pytest

from experiments.sd_membership_sft.data import SFTRecord
from experiments.sd_membership_sft.generalization import (
    GenerationQualityScorer,
    build_eval_samples,
    paired_bootstrap_delta,
    summarize_model_scores,
)


class _StubEncoding:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class StubTokenizer:
    """Splits on spaces into stable word ids; decodes back to words."""

    eos_token_id = 0
    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False, **_: object):
        return _StubEncoding([abs(hash(word)) % 50_000 for word in text.split()])

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return " ".join(f"word{value}" for value in ids)


def _record(tokens: int, index: int = 0) -> SFTRecord:
    return SFTRecord(
        record_id=f"nart:test:{index}",
        source="example.org",
        response_ids=tuple(range(1000, 1000 + tokens)),
        response_hash=f"hash{index}",
        prompt_ids=(1, 2, 3),
        prompt_hash="p",
        prompt_text="prompt",
        topic="topic",
    )


def test_scorer_rewards_exact_match_and_penalizes_mismatch() -> None:
    scorer = GenerationQualityScorer()
    exact = scorer.score("the quick brown fox jumps", "the quick brown fox jumps")
    assert exact["bleu4"] == pytest.approx(1.0)
    assert exact["rouge1"] == pytest.approx(1.0)
    assert exact["rougeL"] == pytest.approx(1.0)

    disjoint = scorer.score("totally different words here now", "the quick brown fox jumps")
    assert disjoint["bleu4"] == pytest.approx(0.0)
    assert disjoint["rouge1"] == pytest.approx(0.0)


def test_build_eval_samples_slices_context_and_reference() -> None:
    tokenizer = StubTokenizer()
    records = [_record(tokens=500, index=index) for index in range(12)]
    samples = build_eval_samples(
        records, tokenizer, samples=8, context_tokens=256, gen_tokens=128, seed=3
    )

    assert len(samples) == 8
    sample = samples[0]
    assert len(sample["prompt_ids"]) == 3 + 256  # prompt + context
    assert sample["prompt_ids"][3:] == list(range(1000, 1256))
    assert sample["reference"] == " ".join(f"word{value}" for value in range(1256, 1384))


def test_build_eval_samples_skips_short_documents() -> None:
    tokenizer = StubTokenizer()
    records = [_record(tokens=500, index=index) for index in range(4)]
    records.append(_record(tokens=300, index=99))  # 300 < 256 + 128
    samples = build_eval_samples(
        records, tokenizer, samples=10, context_tokens=256, gen_tokens=128, seed=3
    )
    assert len(samples) == 4
    assert all(sample["record_id"] != "nart:test:99" for sample in samples)


def test_paired_bootstrap_delta_detects_shift() -> None:
    rng = np.random.default_rng(0)
    left = rng.normal(loc=0.6, scale=0.01, size=400)
    right = rng.normal(loc=0.4, scale=0.01, size=400)
    result = paired_bootstrap_delta(left, right, repeats=200, seed=1)
    assert result["delta"] == pytest.approx(0.2, abs=0.01)
    assert result["ci95_low"] > 0.15
    assert result["ci95_high"] < 0.25

    identical = paired_bootstrap_delta(left, left.copy(), repeats=50, seed=1)
    assert identical["delta"] == 0.0
    assert identical["ci95_low"] == 0.0 and identical["ci95_high"] == 0.0


def test_summarize_model_scores_gate_and_degradation() -> None:
    rng = np.random.default_rng(7)
    tuned_member = {m: rng.normal(0.8, 0.01, 50) for m in GenerationQualityScorer.METRICS}
    tuned_nonmember = {m: rng.normal(0.78, 0.01, 50) for m in GenerationQualityScorer.METRICS}
    base_member = {m: rng.normal(0.9, 0.01, 50) for m in GenerationQualityScorer.METRICS}
    base_nonmember = {m: rng.normal(0.9, 0.01, 50) for m in GenerationQualityScorer.METRICS}
    tuned = {"member": tuned_member, "nonmember": tuned_nonmember}
    base = {"member": base_member, "nonmember": base_nonmember}

    # gaps ~0.02 < 0.03 threshold -> PASS
    passing = summarize_model_scores(
        tuned, base, GenerationQualityScorer.METRICS, bootstrap_repeats=100, seed=1, gap_threshold=0.03
    )
    assert passing["nart_overfitting_gate"] == "PASS"
    for metric in GenerationQualityScorer.METRICS:
        row = passing["nart_member_minus_nonmember"][metric]
        assert row["gate"] == "PASS"
    # base was better: relative drop positive
    assert passing["base_minus_tuned"]["member/bleu4"]["relative_drop"] > 0

    # huge member-vs-nonmember gap -> FAIL
    tuned_bad = {
        "member": {m: rng.normal(0.95, 0.01, 50) for m in GenerationQualityScorer.METRICS},
        "nonmember": tuned_nonmember,
    }
    failing = summarize_model_scores(
        tuned_bad, base, GenerationQualityScorer.METRICS, bootstrap_repeats=100, seed=1, gap_threshold=0.03
    )
    assert failing["nart_overfitting_gate"] == "FAIL"
    assert failing["nart_member_minus_nonmember"]["bleu4"]["gate"] == "FAIL"
