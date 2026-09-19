import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

import pytest

from experiments.sd_membership_sft.drafts.common import (
    PAIR_MODELS,
    checkpoint_complete,
    save_pretrained_atomically,
    validate_training_contract,
)
from experiments.sd_membership_sft.splits import (
    CONTROLLED_SPLIT_SCHEMA_VERSION,
    build_controlled_split_from_shared_manifest,
    build_split_from_shared_manifest,
    prepare_shared_split_manifest,
)


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/sd_membership_sft/retrain_speculator_matrix.sh"


class _Tokens:
    def __init__(self, values):
        self.input_ids = values


class _FakeTokenizer:
    def __init__(self, offset: int):
        self.offset = offset

    def __call__(self, text, **kwargs):
        values = [
            self.offset + index + sum(word.encode("utf-8"))
            for index, word in enumerate(str(text).split())
        ]
        maximum = kwargs.get("max_length")
        if kwargs.get("truncation") and maximum is not None:
            values = values[:maximum]
        return _Tokens(values)


class _DuplicateTokenizer(_FakeTokenizer):
    def __call__(self, text, **kwargs):
        if str(text).startswith("You are a helpful assistant"):
            return super().__call__(text, **kwargs)
        return _Tokens([7, 8, 9])


def _write_pool(tmp_path: Path, count: int = 9) -> Path:
    pool = tmp_path / "pool.jsonl"
    rows = []
    for index in range(count):
        text = " ".join(f"document_{index}_token_{position}" for position in range(40))
        text_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        rows.append(
            {
                "record_id": f"test:{index}",
                "source": "test",
                "title": f"title {index}",
                "text": text,
                "text_sha256": text_sha,
                "creation_timestamp": "2026-01-01T00:00:00Z",
                "snapshot_revision": index,
            }
        )
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    pool.write_bytes(payload)
    pool.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "wikitection",
                "jsonl_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return pool


def _write_near_duplicate_pool(tmp_path: Path) -> Path:
    pool = tmp_path / "pool.jsonl"
    rows = []
    for index in range(7):
        words = [f"unique_{index}_{position}" for position in range(20)]
        if index in {4, 6}:
            words = [f"shared_{position}" for position in range(19)] + [
                f"different_{index}"
            ]
        text = " ".join(words)
        rows.append(
            {
                "record_id": f"test:{index}",
                "source": "test",
                "title": f"title {index}",
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "creation_timestamp": "2026-01-01T00:00:00Z",
                "snapshot_revision": index,
            }
        )
    payload = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    pool.write_bytes(payload)
    pool.with_suffix(".manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "wikitection",
                "jsonl_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ),
        encoding="utf-8",
    )
    return pool


def test_shared_manifest_keeps_raw_ids_identical_across_tokenizers(tmp_path):
    pool = _write_pool(tmp_path)
    manifest = tmp_path / "shared.json"
    sources = {"tokenizer-a@rev": _FakeTokenizer(100), "tokenizer-b@rev": _FakeTokenizer(500)}
    prepare_shared_split_manifest(
        "wikitection",
        pool,
        sources,
        n_per_class=2,
        n_aux=2,
        seed=1919,
        output_path=manifest,
        min_tokens=3,
        max_tokens=40,
    )

    split_a = build_split_from_shared_manifest(
        "wikitection", pool, sources["tokenizer-a@rev"], manifest, "tokenizer-a@rev"
    )
    split_b = build_split_from_shared_manifest(
        "wikitection", pool, sources["tokenizer-b@rev"], manifest, "tokenizer-b@rev"
    )
    ids_a = [[record.record_id for record in rows] for rows in split_a[:3]]
    ids_b = [[record.record_id for record in rows] for rows in split_b[:3]]
    assert ids_a == ids_b
    assert split_a[3]["raw_document_assignment_shared"] is True
    assert split_b[3]["split_seed"] == 1919


def test_shared_manifest_supports_four_disjoint_roles(tmp_path):
    pool = _write_pool(tmp_path, count=12)
    manifest = tmp_path / "controlled.json"
    source = "tokenizer-a@rev"
    tokenizer = _FakeTokenizer(100)

    artifact = prepare_shared_split_manifest(
        "wikitection",
        pool,
        {source: tokenizer},
        n_per_class=2,
        n_aux=2,
        n_audit_aux=2,
        seed=1919,
        output_path=manifest,
        min_tokens=3,
        max_tokens=40,
    )
    split = build_controlled_split_from_shared_manifest(
        "wikitection", pool, tokenizer, manifest, source
    )

    assert artifact["schema_version"] == CONTROLLED_SPLIT_SCHEMA_VERSION
    assert artifact["counts"] == {
        "member": 2,
        "nonmember": 2,
        "auxiliary": 2,
        "audit_auxiliary": 2,
    }
    roles = (
        split.members,
        split.nonmembers,
        split.draft_auxiliary,
        split.audit_auxiliary,
    )
    assert tuple(map(len, roles)) == (2, 2, 2, 2)
    assert len({row.record_id for role in roles for row in role}) == 8
    assert split.metadata["cross_split_ngram_audit"]["gate"] == "PASS"


def test_shared_manifest_filters_near_duplicates_and_backfills(tmp_path):
    pool = _write_near_duplicate_pool(tmp_path)
    manifest = tmp_path / "shared.json"
    sources = {
        "tokenizer-a@rev": _FakeTokenizer(100),
        "tokenizer-b@rev": _FakeTokenizer(500),
    }

    artifact = prepare_shared_split_manifest(
        "wikitection",
        pool,
        sources,
        n_per_class=2,
        n_aux=2,
        seed=1919,
        output_path=manifest,
        min_tokens=3,
        max_tokens=20,
    )

    selected_ids = {
        entry["record_id"]
        for split in artifact["splits"].values()
        for entry in split
    }
    assert len(selected_ids) == 6
    assert not {"test:4", "test:6"}.issubset(selected_ids)
    assert artifact["selection"]["rejected_near_duplicate"] == 1
    for source, tokenizer in sources.items():
        *_, metadata = build_split_from_shared_manifest(
            "wikitection", pool, tokenizer, manifest, source
        )
        assert metadata["cross_split_ngram_audit"]["gate"] == "PASS"


def test_shared_manifest_replaces_unaudited_stale_artifact(tmp_path):
    pool = _write_pool(tmp_path)
    manifest = tmp_path / "shared.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "status": "incomplete"}),
        encoding="utf-8",
    )

    artifact = prepare_shared_split_manifest(
        "wikitection",
        pool,
        {"tokenizer-a@rev": _FakeTokenizer(100)},
        n_per_class=2,
        n_aux=2,
        seed=1919,
        output_path=manifest,
        min_tokens=3,
        max_tokens=40,
    )

    assert json.loads(manifest.read_text(encoding="utf-8")) == artifact
    assert artifact["schema_version"] > 1


def test_shared_manifest_fails_instead_of_substituting_token_duplicates(tmp_path):
    pool = _write_pool(tmp_path)
    manifest = tmp_path / "shared.json"
    source = "tokenizer-a@rev"
    prepare_shared_split_manifest(
        "wikitection",
        pool,
        {source: _FakeTokenizer(100)},
        n_per_class=2,
        n_aux=2,
        seed=1919,
        output_path=manifest,
        min_tokens=3,
        max_tokens=40,
    )
    with pytest.raises(RuntimeError, match="duplicate response text"):
        build_split_from_shared_manifest(
            "wikitection", pool, _DuplicateTokenizer(0), manifest, source
        )


def test_checkpoint_is_complete_only_after_atomic_marker(tmp_path):
    destination = tmp_path / "head"

    def writer(path):
        path.mkdir(parents=True)
        (path / "config.json").write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"weights")

    save_pretrained_atomically(destination, writer, {"stage": "aux_head"})
    assert checkpoint_complete(destination)
    marker = json.loads((destination / "_COMPLETE.json").read_text(encoding="utf-8"))
    assert marker["status"] == "complete"
    assert marker["stage"] == "aux_head"
    with pytest.raises(FileExistsError):
        save_pretrained_atomically(destination, writer, {"stage": "aux_head"})


def test_training_contract_requires_same_seed_and_effective_batch_16():
    valid = SimpleNamespace(
        seed=1919,
        data_seed=1919,
        batch_size=2,
        grad_accum=8,
        head_batch_size=2,
        head_grad_accum=8,
        head_updates=384,
        kd_temperature=2.0,
    )
    validate_training_contract(valid)
    invalid_seed = SimpleNamespace(**{**vars(valid), "data_seed": 1949})
    with pytest.raises(ValueError, match="data_seed and seed"):
        validate_training_contract(invalid_seed)
    invalid_batch = SimpleNamespace(**{**vars(valid), "head_grad_accum": 7})
    with pytest.raises(ValueError, match="head batch_size"):
        validate_training_contract(invalid_batch)


def test_pinned_model_registry_matches_the_approved_revisions():
    assert PAIR_MODELS == {
        "qwen3_8b_eagle3": {
            "target": "Qwen/Qwen3-8B",
            "target_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "speculator": "RedHatAI/Qwen3-8B-speculator.eagle3",
            "speculator_revision": "08610ffa01dd9f16731fe8f627b85905b6aa51c4",
            "kind": "eagle3",
        },
        "llama31_8b_eagle3": {
            "target": "unsloth/Meta-Llama-3.1-8B-Instruct",
            "target_revision": "a2856192dd7c25b842431f39c179a6c2c2f627d1",
            "speculator": "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
            "speculator_revision": "f4fa34a8f803a0ba75d048d6b3dbc1ad5149e9ac",
            "kind": "eagle3",
        },
        "qwen35_9b_mtp": {
            "target": "Qwen/Qwen3.5-9B-Base",
            "target_revision": "68c46c4b3498877f3ef123c856ecfde50c39f404",
            "speculator": "Qwen/Qwen3.5-9B-Base",
            "speculator_revision": "68c46c4b3498877f3ef123c856ecfde50c39f404",
            "kind": "mtp",
        },
    }


def test_dry_run_has_exact_54_condition_matrix_and_162_artifacts():
    result = subprocess.run(
        [str(SCRIPT), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "PYTHON": sys.executable},
        check=True,
        capture_output=True,
        text=True,
    )
    conditions = [
        line for line in result.stdout.splitlines() if line.startswith("[condition]")
    ]
    stages = [line for line in result.stdout.splitlines() if line.startswith("[stage]")]
    assert len(conditions) == 54
    assert len(stages) == 163
    assert sum("pair=qwen3_8b_eagle3 " in line for line in conditions) == 18
    assert sum("pair=llama31_8b_eagle3 " in line for line in conditions) == 18
    assert sum("pair=qwen35_9b_mtp " in line for line in conditions) == 18
    assert sum(" stage=target " in line for line in stages) == 54
    assert sum(" stage=auxiliary " in line for line in stages) == 54
    assert sum(" stage=member " in line for line in stages) == 54
    assert sum(" stage=source " in line for line in stages) == 1
    assert all("--head-updates 384 --head-lr 2e-5" in line for line in stages)
    assert all("--kd-temperature 2.0" in line for line in stages)
    assert all(
        "--n-per-class 2000 --n-aux 2000 --n-audit-aux 600" in line
        for line in stages
    )
    assert all("--seed " in line and "--data-seed " in line for line in stages)
    for line in stages:
        seed = re.search(r"\bseed=(\d+)\b", line)
        assert seed is not None
        assert f"PYTHONHASHSEED={seed.group(1)}" in line
        assert f"--seed {seed.group(1)} --data-seed {seed.group(1)}" in line
    assert all(
        "--batch-size 2 --grad-accum 8" in line
        and "--head-batch-size 2 --head-grad-accum 8" in line
        for line in stages
    )
    assert "mtp-joint" not in result.stdout
    assert result.stdout.rstrip().endswith(
        "[plan-ok] conditions=54 checkpoints=162 workers=4"
    )
