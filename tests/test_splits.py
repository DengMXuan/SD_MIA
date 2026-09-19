from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.sd_membership_sft.data import SFTRecord, make_sft_example
from experiments.sd_membership_sft.splits import (
    BENCHMARK_TOKEN_BANDS,
    SFT_PROMPT,
    build_controlled_split,
    build_split,
)


class _StubEncoding:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class StubTokenizer:
    """Whitespace tokenizer with stable word IDs; no network or model files."""

    eos_token_id = 151643
    pad_token_id = 151643

    def __call__(self, text: str, add_special_tokens: bool = False, truncation: bool = False, max_length: int | None = None):
        words = text.split()
        ids = [int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 100_000 for word in words]
        if truncation and max_length is not None:
            ids = ids[:max_length]
        return _StubEncoding(ids)


def _write_pool(path: Path, texts: list[str], titles: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, (title, text) in enumerate(zip(titles, texts)):
            handle.write(
                json.dumps(
                    {
                        "record_id": f"fake:{index}",
                        "source": "example.org",
                        "title": title,
                        "creation_timestamp": "2026-06-01T00:00:00Z",
                        "snapshot_revision": 0,
                        "snapshot_timestamp": "2026-06-01T00:00:00Z",
                        "canonical_url": f"https://example.org/{index}",
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        "text": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    payload = path.read_bytes()
    manifest = {
        "benchmark": "wikitection",
        "jsonl_sha256": hashlib.sha256(payload).hexdigest(),
        "creation_interval_inclusive": {"start": "2026-05-01T00:00:00Z", "end": "2026-08-29T23:59:59Z"},
        "timestamp_semantics": "test",
        "license": "test",
        "provenance": "test",
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest))


def _fake_documents(count: int, words: int = 300) -> tuple[list[str], list[str]]:
    titles = [f"Document {index} about topic {index % 7}" for index in range(count)]
    texts = [
        " ".join(f"word{index}_{position}" for position in range(words))
        for index in range(count)
    ]
    return titles, texts


def test_build_split_disjoint_classes_and_ngram_gate(tmp_path: Path) -> None:
    titles, texts = _fake_documents(30)
    pool = tmp_path / "pool.jsonl"
    _write_pool(pool, texts, titles)
    tokenizer = StubTokenizer()

    members, nonmembers, auxiliary, metadata = build_split(
        "wikitection", pool, tokenizer, n_per_class=8, n_aux=8, seed=7
    )

    assert (len(members), len(nonmembers), len(auxiliary)) == (8, 8, 8)
    split_hashes = [
        {record.response_hash for record in split}
        for split in (members, nonmembers, auxiliary)
    ]
    assert split_hashes[0].isdisjoint(split_hashes[1])
    assert split_hashes[0].isdisjoint(split_hashes[2])
    assert split_hashes[1].isdisjoint(split_hashes[2])
    assert metadata["cross_split_ngram_audit"]["gate"] == "PASS"
    assert metadata["cross_split_ngram_audit"]["maximum_pair_overlap_fraction"] <= 0.80
    assert metadata["token_band"] == {
        "min_tokens": BENCHMARK_TOKEN_BANDS["wikitection"][0],
        "max_tokens": BENCHMARK_TOKEN_BANDS["wikitection"][1],
    }
    assert metadata["band_dropped_documents"] == 0  # loop stops once 24 records exist
    for record in members + nonmembers + auxiliary:
        assert record.prompt_text == SFT_PROMPT.format(topic=record.topic)
        assert record.prompt_ids is not None


def test_controlled_split_uses_independent_600_role(tmp_path: Path) -> None:
    titles, texts = _fake_documents(34)
    pool = tmp_path / "pool.jsonl"
    _write_pool(pool, texts, titles)

    split = build_controlled_split(
        "wikitection",
        pool,
        StubTokenizer(),
        n_per_class=8,
        n_draft_aux=8,
        n_audit_aux=2,
        seed=7,
    )

    roles = (
        split.members,
        split.nonmembers,
        split.draft_auxiliary,
        split.audit_auxiliary,
    )
    assert tuple(map(len, roles)) == (8, 8, 8, 2)
    role_ids = [{record.record_id for record in role} for role in roles]
    for left in range(len(role_ids)):
        for right in range(left + 1, len(role_ids)):
            assert role_ids[left].isdisjoint(role_ids[right])
    assert split.metadata["counts"] == {
        "member": 8,
        "nonmember": 8,
        "draft_auxiliary": 8,
        "audit_auxiliary": 2,
    }
    assert "audit_auxiliary_uses" not in split.metadata
    assert split.metadata["cross_split_ngram_audit"]["gate"] == "PASS"


def test_build_split_drops_documents_under_token_band(tmp_path: Path) -> None:
    short_titles, short_texts = _fake_documents(4, words=20)
    titles, texts = _fake_documents(24)
    pool = tmp_path / "pool.jsonl"
    _write_pool(pool, texts + short_texts, titles + short_titles)
    tokenizer = StubTokenizer()

    members, nonmembers, auxiliary, metadata = build_split(
        "wikitection", pool, tokenizer, n_per_class=8, n_aux=8, seed=7
    )

    assert metadata["band_dropped_documents"] == 4
    assert len(members) == len(nonmembers) == len(auxiliary) == 8
    assert all(len(record.response_ids) >= 128 for record in members)


def test_build_split_rejects_tampered_pool(tmp_path: Path) -> None:
    titles, texts = _fake_documents(24)
    pool = tmp_path / "pool.jsonl"
    _write_pool(pool, texts, titles)
    manifest = json.loads(pool.with_suffix(".manifest.json").read_text())
    manifest["jsonl_sha256"] = "0" * 64
    pool.with_suffix(".manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="hash does not match"):
        build_split("wikitection", pool, StubTokenizer(), 8, 8, seed=7)


def test_build_split_rejects_wrong_benchmark(tmp_path: Path) -> None:
    titles, texts = _fake_documents(24)
    pool = tmp_path / "pool.jsonl"
    _write_pool(pool, texts, titles)

    with pytest.raises(RuntimeError, match="benchmark"):
        build_split("arxivtection", pool, StubTokenizer(), 8, 8, seed=7)


def test_make_sft_example_masks_fixed_prompt() -> None:
    tokenizer = StubTokenizer()
    record = SFTRecord(
        record_id="sft:test:abc",
        source="example.org",
        response_ids=tuple(tokenizer("alpha beta gamma").input_ids),
        response_hash="abc",
        prompt_ids=tuple(tokenizer(SFT_PROMPT.format(topic="Topic X")).input_ids),
        prompt_hash="def",
        prompt_text=SFT_PROMPT.format(topic="Topic X"),
        topic="Topic X",
    )
    example = make_sft_example(record, tokenizer)
    prompt_length = len(record.prompt_ids)
    response_length = len(record.response_ids) + 1  # plus EOS

    assert len(example["input_ids"]) == prompt_length + response_length
    assert example["labels"][:prompt_length] == [-100] * prompt_length
    assert example["labels"][prompt_length:] == list(record.response_ids) + [
        tokenizer.eos_token_id
    ]
