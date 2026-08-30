from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.sd_membership_sft.public_data import build_public_snapshot_split


ROOT = Path(__file__).resolve().parents[1]
DATASETS = (
    ROOT / "experiments/data/sft_public/wikitext-wikitext-103-raw-v1.jsonl",
    ROOT / "experiments/data/sft_public/xsum-default.jsonl",
    ROOT / "experiments/data/sft_public/cnn-dailymail-3.0.0.jsonl",
)
ALLOWED_RECORD_FIELDS = {
    "record_id",
    "source",
    "prompt_ids",
    "response_ids",
    "source_char_count",
    "snapshot_revision",
    "creation_timestamp",
    "content_token_sha256",
}


@pytest.mark.parametrize("path", DATASETS)
def test_public_pool_matches_manifest_and_schema(path: Path) -> None:
    raw = path.read_bytes()
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    records = [json.loads(line) for line in raw.decode().splitlines() if line]

    assert hashlib.sha256(raw).hexdigest() == manifest["jsonl_sha256"]
    assert len(records) == manifest["records"] == 1800
    assert manifest["raw_text_persisted"] is False
    assert manifest["persisted_content_fields"] == ["prompt_ids", "response_ids"]
    assert all(set(record) == ALLOWED_RECORD_FIELDS for record in records)
    assert all(len(record["prompt_ids"]) == 64 for record in records)
    assert all(len(record["response_ids"]) == 128 for record in records)
    hashes = [record["content_token_sha256"] for record in records]
    assert len(set(hashes)) == len(hashes)


@pytest.mark.parametrize("path", DATASETS)
def test_public_pool_split_is_disjoint_and_passes_ngram_gate(path: Path) -> None:
    members, nonmembers, auxiliary, metadata = build_public_snapshot_split(
        path=path,
        tokenizer=None,
        response_tokens=128,
        n_per_class=16,
        n_aux=16,
        seed=20260828,
    )
    split_hashes = [
        {record.response_hash for record in split}
        for split in (members, nonmembers, auxiliary)
    ]

    assert split_hashes[0].isdisjoint(split_hashes[1])
    assert split_hashes[0].isdisjoint(split_hashes[2])
    assert split_hashes[1].isdisjoint(split_hashes[2])
    assert metadata["cross_split_ngram_audit"]["gate"] == "PASS"
    assert metadata["cross_split_ngram_audit"]["maximum_pair_overlap_fraction"] <= 0.80
