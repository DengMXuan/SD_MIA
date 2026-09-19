from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.sd_membership_sft.pools import _write_pool, merge_pool_files
from experiments.sd_membership_sft.pool_storage import read_verified_pool


def _record(record_id: str, words: list[str]) -> dict[str, object]:
    text = " ".join(words)
    return {
        "record_id": record_id,
        "source": "example.org",
        "title": record_id,
        "creation_timestamp": "2026-09-01T00:00:00Z",
        "snapshot_revision": 0,
        "canonical_url": f"https://example.org/{record_id}",
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
    }


def _unique_words(prefix: str, count: int = 30) -> list[str]:
    return [f"{prefix}_{index}" for index in range(count)]


def _pool(path: Path, records: list[dict[str, object]]) -> None:
    _write_pool(
        path,
        records,
        {
            "benchmark": "wikitection",
            "creation_interval_inclusive": {
                "start": "2026-09-01T00:00:00Z",
                "end": "2026-09-17T23:59:59Z",
            },
        },
    )


def test_merge_pool_preserves_prefix_and_filters_all_duplicate_types(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "merged.jsonl"
    original = [_record(f"base:{index}", _unique_words(f"base{index}")) for index in range(4)]
    near = _unique_words("base0")
    near[-1] = "changed_tail"
    candidates = [
        _record("base:0", _unique_words("different-id-collision")),
        _record("candidate:exact", _unique_words("base1")),
        _record("candidate:near", near),
        *[
            _record(f"candidate:{index}", _unique_words(f"candidate{index}"))
            for index in range(3)
        ],
    ]
    _pool(base, original)
    _pool(candidate, candidates)
    candidate_manifest_path = candidate.with_suffix(".manifest.json")
    candidate_manifest = json.loads(candidate_manifest_path.read_text())
    candidate_manifest["creation_interval_inclusive"] = {
        "start": "2026-09-18T00:00:00Z",
        "end": "2026-09-30T23:59:59Z",
    }
    candidate_manifest_path.write_text(json.dumps(candidate_manifest))

    manifest = merge_pool_files(
        "wikitection",
        base,
        [candidate],
        target_records=7,
        output_path=output,
        keep_backup=False,
    )

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[:4] == original
    assert [row["record_id"] for row in rows[4:]] == [
        "candidate:0",
        "candidate:1",
        "candidate:2",
    ]
    dedupe = manifest["extension"]["deduplication"]
    assert dedupe["added_records"] == 3
    assert dedupe["skipped_record_id"] == 1
    assert dedupe["skipped_exact_text"] == 1
    assert dedupe["skipped_near_duplicate"] == 1
    assert manifest["records"] == 7
    assert hashlib.sha256(output.read_bytes()).hexdigest() == manifest["jsonl_sha256"]
    assert manifest["creation_interval_inclusive"] == {
        "start": "2026-09-01T00:00:00Z",
        "end": "2026-09-30T23:59:59Z",
    }


def test_merge_pool_failure_does_not_create_partial_output(tmp_path: Path) -> None:
    base = tmp_path / "base.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "merged.jsonl"
    _pool(base, [_record("base:0", _unique_words("base"))])
    _pool(candidate, [_record("candidate:0", _unique_words("candidate"))])

    with pytest.raises(RuntimeError, match="unique records available"):
        merge_pool_files(
            "wikitection",
            base,
            [candidate],
            target_records=3,
            output_path=output,
            keep_backup=False,
        )

    assert not output.exists()
    assert not output.with_suffix(".manifest.json").exists()


def test_pool_reader_recovers_interrupted_data_manifest_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from experiments.sd_membership_sft import pool_storage

    output = tmp_path / "pool.jsonl"
    original = [_record("base:0", _unique_words("base"))]
    replacement = [
        _record("new:0", _unique_words("new0")),
        _record("new:1", _unique_words("new1")),
    ]
    _pool(output, original)

    real_replace = pool_storage.os.replace
    calls = 0

    def interrupt_second_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("simulated interruption between pair renames")
        return real_replace(source, destination)

    monkeypatch.setattr(pool_storage.os, "replace", interrupt_second_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        _pool(output, replacement)

    rows, manifest, _payload = read_verified_pool(output, "wikitection")
    assert rows == replacement
    assert manifest["records"] == 2
    assert not output.with_name(f".{output.name}.transaction.json").exists()
