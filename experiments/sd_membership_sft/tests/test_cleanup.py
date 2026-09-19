"""Regression checks for the SD-only cleanup and historical entry points."""
import importlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch


PACKAGE = "experiments.sd_membership_sft"
SOURCE = Path(__file__).resolve().parents[1]
ROOT = SOURCE.parents[1]


def test_mainline_imports_do_not_load_archived_experiments():
    script = """
import importlib, sys
for name in (
    'conditional_accept_only', 'difficulty_accept_only', 'combined_accept_only',
    'collect_draft_difficulty', 'collect_deployment_observations',
    'deployment_accept_only', 'serial_accept_only',
    'summarize_combination_validation', 'summarize_priority_validation',
):
    importlib.import_module('experiments.sd_membership_sft.' + name)
assert not any(name.startswith('experiments.sd_membership_sft.archive') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, capture_output=True, text=True)


def test_legacy_imports_are_the_same_module_as_the_archive():
    manifest = json.loads((SOURCE / "archive/MANIFEST.json").read_text())
    for item in manifest["modules"]:
        old = importlib.import_module(f"{PACKAGE}.{item['module']}")
        archived = importlib.import_module(f"{PACKAGE}.archive.{item['module']}")
        assert old is archived
        if hasattr(archived, "ROOT"):
            assert archived.ROOT == ROOT


def test_extracted_helpers_have_one_implementation():
    from experiments.sd_membership_sft import audit_metrics, audit_runtime, replay_cache
    from experiments.sd_membership_sft.archive import (
        adaptive_window_accept_only, full_delta_mia, neural_adaptive_accept_only, stat_delta_mia,
    )
    assert full_delta_mia.split_indices is audit_runtime.split_indices
    assert full_delta_mia.partial_auc is audit_metrics.partial_auc
    assert adaptive_window_accept_only.membership_metrics is audit_metrics.membership_metrics
    assert neural_adaptive_accept_only._paths is audit_runtime._paths
    assert stat_delta_mia.load_delta_data is replay_cache.load_delta_data


def test_lowq_baseline_matches_historical_ties_short_records_and_budgets():
    from experiments.sd_membership_sft.archive.adaptive_window_accept_only import raw_fragment_scores
    from experiments.sd_membership_sft.conditional_accept_only import Observations, lowq_score
    from experiments.sd_membership_sft.lowq_baseline import standardized_max

    rng = np.random.default_rng(121)
    lengths = np.array([1, 2, 7, 11, 31])
    q = -rng.integers(1, 5, size=lengths.sum()).astype(float)
    bits = rng.integers(0, 2, size=(len(q), 1, 4), dtype=np.uint8)
    obs = Observations(q[:, None], bits, lengths)
    reference = np.array([0, 1, 2])
    for budget in (1, 2, 4):
        accept = np.all(bits[:, 0, :budget], axis=1).astype(float)
        raw = raw_fragment_scores(accept, np.full(len(q), .5), q, lengths, budget)
        expected = standardized_max(raw, ("lowq_10", "lowq_20", "lowq_50"), reference)
        np.testing.assert_array_equal(lowq_score(obs, budget, reference), expected)


@pytest.mark.parametrize("dimensions", [2, 5])
def test_mainline_fit_preserves_historical_static_detector(dimensions):
    from experiments.sd_membership_sft import difficulty_accept_only as current
    from experiments.sd_membership_sft.archive import priority_accept_only as historical

    torch.set_num_threads(1)
    rng = np.random.default_rng(17)
    lengths = np.full(5, 8)
    features = rng.normal(size=(40, dimensions)).astype(np.float32)
    counts = rng.integers(0, 3, size=40)
    parts = {"train": np.array([0, 1]), "validation": np.array([2])}
    args = dict(seed=18, device="cpu", epochs=2)
    before = historical.fit(features, counts, lengths, parts, **args)
    after = current.fit(features, counts, lengths, parts, **args)
    assert before[3:] == after[3:]
    for name, value in before[0].state_dict().items():
        torch.testing.assert_close(value, after[0].state_dict()[name], rtol=0, atol=0)
    for index in (1, 2):
        np.testing.assert_array_equal(before[index], after[index])
    normalized = (features - before[1]) / before[2]
    np.testing.assert_array_equal(
        historical.predict(before[0], normalized, counts, lengths, "cpu"),
        current.predict(after[0], normalized, counts, lengths, "cpu"),
    )


def test_archived_cli_remains_accessible_through_old_and_new_paths():
    for name in ("priority_accept_only", "archive.priority_accept_only", "difficulty_accept_only"):
        result = subprocess.run(
            [sys.executable, "-m", f"{PACKAGE}.{name}", "--help"],
            cwd=ROOT, check=True, capture_output=True, text=True,
        )
        assert "--benchmark" in result.stdout
        if name == "difficulty_accept_only":
            assert "{sequence,features,posthoc}" not in result.stdout


def test_mainline_loads_legacy_draft_feature_manifest(tmp_path, monkeypatch):
    from experiments.sd_membership_sft import difficulty_accept_only as current
    from experiments.sd_membership_sft.replay_cache import ReplayData

    lengths = np.array([2])
    ids = np.array(["trusted-nonmember"])
    data = ReplayData(np.array([0]), ids, lengths, np.array([0, 2]),
                      np.array([-1.5, -2.5]), np.array([-2., -3.]))
    q = np.array([[-2., .5, .2, 1., 0., 1.], [-3., .6, .3, 2., 1., 1.]])
    for name, value in (("q", q), ("lengths", lengths), ("record_ids", ids)):
        np.save(tmp_path / f"{name}.npy", value)
    probability = tmp_path / "probability.npz"
    probability.write_bytes(b"frozen probability cache")
    checkpoint = tmp_path / "wikitection_qwen3_8b_epoch1/checkpoints/draft_auxiliary_distilled"
    manifest = {
        "benchmark": "wikitection", "epoch": 1,
        "role": "draft_auxiliary_distilled", "eos_included": False,
        "checkpoint_provenance": {"checkpoint_path": str(checkpoint)},
        "probability_cache": {"sha256": hashlib.sha256(probability.read_bytes()).hexdigest()},
    }
    (tmp_path / "feature_manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(current, "RESULTS", tmp_path)
    monkeypatch.setattr(current, "feature_root", lambda *args: tmp_path)
    monkeypatch.setattr(current, "_paths", lambda *args: (tmp_path / "unused.npz", probability))
    monkeypatch.setattr(current, "load_replay_data", lambda *args: data)
    observations, labels, record_ids, features, source = current.load_feature_observations("wikitection", 1, 7)
    np.testing.assert_array_equal(observations.logq, q[:, :1])
    np.testing.assert_array_equal(features, q[:, 1:4])
    np.testing.assert_array_equal(record_ids, ids)
    assert observations.bits.shape == (2, 1, 2)
    assert source["q_difference_max"] == 0


def test_deployment_partition_uses_independent_600_and_2000_per_test_class():
    from experiments.sd_membership_sft.audit_partitions import deployment_partitions

    roles = np.asarray(
        ["audit_auxiliary"] * 600 + ["member"] * 2000 + ["nonmember"] * 2000
    )
    labels = np.r_[
        np.zeros(600, dtype=np.int64),
        np.ones(2000, dtype=np.int64),
        np.zeros(2000, dtype=np.int64),
    ]
    ids = np.asarray([f"record-{index}" for index in range(4600)])
    current = deployment_partitions(labels, ids, roles)
    assert len(current["train"]) == 320
    assert len(current["validation"]) == 80
    assert len(current["calibration"]) == 200
    assert int((labels[current["test"]] == 0).sum()) == 2000
    assert int((labels[current["test"]] == 1).sum()) == 2000
    assert not np.intersect1d(
        current["test"], np.r_[current["reference"], current["calibration"]]
    ).size
    auxiliary = np.r_[
        current["train"], current["validation"], current["calibration"]
    ]
    assert np.all(roles[auxiliary] == "audit_auxiliary")
    assert len(np.unique(np.r_[auxiliary, current["test"]])) == 4600
