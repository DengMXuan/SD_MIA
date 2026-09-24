from __future__ import annotations

import json
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.sd_membership_sft.analysis.m1_extract import _build_position_metadata, extract_qh_features
from experiments.shared.methods.features import ACTIVATION_FEATURE_NAMES, Q_FEATURE_NAMES, activation_statistics, aggregate_values, q_features_from_logits
from experiments.sd_membership_sft.archive.m1_fit import load_m1_data, make_partitions, partial_auc, partition_manifest, sample_token_indices, _select_detector_candidate, validation_nonmember_indices
from experiments.shared.data.data import SFTRecord


def test_q_features_use_fp32_and_prescribed_rank_entropy() -> None:
    logits = torch.tensor([[1.0, 2.0, 0.0, -1.0]], dtype=torch.bfloat16)
    result = q_features_from_logits(
        logits,
        torch.tensor([1]),
        torch.tensor([0.0]),
        torch.tensor([np.log(1.0)]),
        row_chunk=1,
    )
    expected_logq = torch.log_softmax(logits.float(), dim=-1)[0, 1]
    assert result.dtype == torch.float32
    assert result[0, 0].item() == pytest.approx(expected_logq.item(), abs=1e-6)
    assert result[0, 2].item() == 0.0  # candidate is top-1; strict greater-than rank
    assert 0.0 < result[0, 1].item() < 1.0
    assert tuple(Q_FEATURE_NAMES) == (
        "log_q",
        "q_entropy_norm",
        "q_rank_norm",
        "q_top1_margin",
        "relative_position",
        "log_length",
    )


def test_activation_statistics_are_population_and_per_token() -> None:
    values = torch.tensor([[1.0, -1.0, 2.0]], dtype=torch.bfloat16)
    result = activation_statistics(values)
    assert result.shape == (1, len(ACTIVATION_FEATURE_NAMES) // 4)
    assert result[0, 0].item() == pytest.approx(2.0 / 3.0)
    assert result[0, 1].item() == pytest.approx(np.std([1.0, -1.0, 2.0], ddof=0))
    assert result[0, 2].item() == pytest.approx(np.sqrt(2.0))
    assert result[0, 9].item() == pytest.approx(2.0 / 3.0)


def test_aggregate_short_record_uses_one_window_mean() -> None:
    values = np.asarray([-1.0, 2.0, 3.0])
    result = aggregate_values(values)
    assert len(result) == 22
    assert result[0] == np.mean(values)
    assert result[1] == np.std(values, ddof=0)
    # Every window wider than the record has one window mean, hence q10=q90.
    assert np.all(result[[12, 14, 16, 18, 20]] == result[[13, 15, 17, 19, 21]])


def test_m1_partitions_keep_nuisance_subsets_out_of_detector_and_explicit() -> None:
    labels = np.r_[np.ones(2000, dtype=np.int64), np.zeros(2000, dtype=np.int64)]
    record_ids = np.asarray([f"record-{index}" for index in range(len(labels))])
    partitions = make_partitions(labels, record_ids)
    assert len(partitions["nuisance_fit"]) == 400
    assert len(partitions["nuisance_location"]) == 300
    assert len(partitions["nuisance_scale"]) == 100
    assert len(partitions["detector_fit"]) == 1200
    assert set(partitions["nuisance_location"]).isdisjoint(partitions["nuisance_scale"])
    assert set(partitions["nuisance_fit"]) == set(partitions["nuisance_location"]) | set(partitions["nuisance_scale"])
    assert set(partitions["nuisance_fit"]).isdisjoint(partitions["detector_fit"])
    assert np.all(labels[partitions["nuisance_fit"]] == 0)
    assert np.all(partitions.partition_by_index != "")


def test_frozen_partitions_follow_record_ids_and_reject_duplicates() -> None:
    labels = np.r_[np.ones(2000, dtype=np.int64), np.zeros(2000, dtype=np.int64)]
    record_ids = np.asarray([f"record-{index}" for index in range(len(labels))])
    original = make_partitions(labels, record_ids)
    frozen = partition_manifest(original, labels, record_ids)

    permutation = np.random.default_rng(17).permutation(len(labels))
    reordered = make_partitions(
        labels[permutation],
        record_ids[permutation],
        frozen_manifest=frozen,
    )
    original_owner = {
        str(record_ids[index]): str(original.partition_by_index[index])
        for index in range(len(record_ids))
    }
    reordered_owner = {
        str(record_ids[permutation[index]]): str(reordered.partition_by_index[index])
        for index in range(len(record_ids))
    }
    assert reordered_owner == original_owner

    duplicate_ids = record_ids.copy()
    duplicate_ids[1] = duplicate_ids[0]
    with pytest.raises(RuntimeError, match="unique"):
        make_partitions(labels, duplicate_ids)


def test_conditional_validation_is_nonmember_only() -> None:
    labels = np.r_[np.ones(2000, dtype=np.int64), np.zeros(2000, dtype=np.int64)]
    record_ids = np.asarray([f"record-{index}" for index in range(len(labels))])
    partitions = make_partitions(labels, record_ids)
    selected = validation_nonmember_indices(labels, partitions)
    assert len(selected) == 400
    assert np.all(labels[selected] == 0)


def test_detector_tie_prefers_stronger_regularization_then_smaller_model() -> None:
    selection = [
        {"validation_pauc_0_05": 0.5, "regularization": 1e-3, "parameter_count": 100},
        {"validation_pauc_0_05": 0.5, "regularization": 1e-1, "parameter_count": 100},
        {"validation_pauc_0_05": 0.5, "regularization": 1e-1, "parameter_count": 50},
    ]
    assert _select_detector_candidate(selection) == 2


def test_sample_token_indices_is_equal_document_weighted() -> None:
    lengths = np.asarray([3, 8, 2], dtype=np.int64)
    indices, docs = sample_token_indices(lengths, np.asarray([0, 1, 2]), seed=7, max_tokens=4)
    assert len(indices) == 3 + 4 + 2
    assert [int(np.sum(docs == index)) for index in range(3)] == [3, 4, 2]
    assert np.all(np.diff(indices[docs == 1]) > 0)


def test_partial_auc_interpolates_tied_threshold_groups() -> None:
    scores = np.asarray([3.0, 2.0, 1.0, 0.0])
    labels = np.asarray([1, 0, 1, 0])
    # The vertical tied-score step at fpr=.5 is kept as one threshold group;
    # the remaining boundary is linearly integrated to .75.
    value = partial_auc(scores, labels, max_fpr=0.75)
    assert value == pytest.approx(2.0 / 3.0)


def test_load_m1_data_does_not_trim_reconciled_no_eos_cache_twice(tmp_path) -> None:
    feature_dir = tmp_path / "features"
    probability_dir = tmp_path / "probability"
    source_probability_dir = tmp_path / "source_probability"
    feature_dir.mkdir()
    probability_dir.mkdir()
    source_probability_dir.mkdir()
    role = "draft_auxiliary_distilled"
    labels = np.asarray([0], dtype=np.int64)
    record_ids = np.asarray(["record-0"])
    lengths = np.asarray([2], dtype=np.int64)
    offsets = np.asarray([0, 2], dtype=np.int64)
    draft = np.asarray([-0.2, -0.3], dtype=np.float32)
    target = np.asarray([-0.1, -0.4], dtype=np.float32)
    q = np.zeros((2, len(Q_FEATURE_NAMES)), dtype=np.float32)
    q[:, 0] = draft
    h = np.zeros((2, len(ACTIVATION_FEATURE_NAMES)), dtype=np.float32)
    np.save(feature_dir / "q.npy", q)
    np.save(feature_dir / "h.npy", h)
    np.save(feature_dir / "labels.npy", labels)
    np.save(feature_dir / "record_ids.npy", record_ids)
    np.save(feature_dir / "lengths.npy", lengths)
    np.save(feature_dir / "offsets.npy", offsets)
    np.save(feature_dir / "eos_mask.npy", np.zeros(2, dtype=bool))
    source_lengths = np.asarray([3], dtype=np.int64)
    source_target = np.asarray([-0.1, -0.4, -0.5], dtype=np.float32)
    source_draft = np.asarray([-0.2, -0.3, -0.6], dtype=np.float32)
    np.savez(
        source_probability_dir / "pq_gap_token_logps.npz",
        lengths=source_lengths,
        target=source_target,
        **{role: source_draft},
    )
    np.savez(
        source_probability_dir / "pq_gap_scores.npz",
        labels=labels,
        record_ids=record_ids,
    )
    source_token_path = source_probability_dir / "pq_gap_token_logps.npz"
    source_scores_path = source_probability_dir / "pq_gap_scores.npz"
    (source_probability_dir / "pq_gap_provenance.json").write_text(
        json.dumps(
            {
                "run_dir": "/run",
                "benchmark": "test",
                "epoch": 1,
                "records": 1,
                "tokens": 3,
                "roles": {
                    role: {"run_dir": "/run", "epoch": 1, "role": role}
                },
            }
        ),
        encoding="utf-8",
    )
    np.savez(
        probability_dir / "pq_gap_scores.npz", labels=labels, record_ids=record_ids
    )
    np.savez(
        probability_dir / "pq_gap_token_logps.npz",
        lengths=lengths,
        target=target,
        **{role: draft},
    )
    token_path = probability_dir / "pq_gap_token_logps.npz"
    scores_path = probability_dir / "pq_gap_scores.npz"
    sha256 = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    (probability_dir / "pq_gap_provenance.json").write_text(
        json.dumps(
            {
                "kind": "reconciled_q_cache",
                "source_probability_dir": str(source_probability_dir),
                "source_probability_sha256": sha256(source_token_path),
                "source_scores_sha256": sha256(source_scores_path),
                "reconciled_token_logps_sha256": sha256(token_path),
                "reconciled_scores_sha256": sha256(scores_path),
                "q_role": role,
                "eos_included": False,
            }
        ),
        encoding="utf-8",
    )
    (feature_dir / "feature_manifest.json").write_text(
        json.dumps(
            {
                "benchmark": "test",
                "epoch": 1,
                "role": role,
                "run_dir": "/run",
                "records": 1,
                "total_tokens": 2,
                "record_ids_sha256": hashlib.sha256(b"record-0").hexdigest(),
                "eos_included": False,
                "probability_cache": {
                    "path": str(source_token_path),
                    "sha256": sha256(source_token_path),
                    "scores_path": str(source_scores_path),
                    "scores_sha256": sha256(source_scores_path),
                },
                "probability_cache_alignment": {
                    "reconciled_probability_dir": str(probability_dir),
                },
            }
        ),
        encoding="utf-8",
    )

    data = load_m1_data(feature_dir, probability_dir, role)
    assert np.array_equal(data.lengths, lengths)
    assert np.allclose(data.target_logp, target)
    assert np.allclose(data.draft_logq, draft)

    # The normal no-EOS path reuses the EOS-inclusive historical source,
    # without creating a reconciled q cache. It must trim exactly once too.
    historical = load_m1_data(feature_dir, source_probability_dir, role)
    assert np.array_equal(historical.lengths, lengths)
    assert np.allclose(historical.target_logp, target)
    assert np.allclose(historical.draft_logq, draft)
    source_manifest_path = source_probability_dir / "pq_gap_provenance.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    source_manifest["tokens"] += 1
    source_manifest_path.write_text(json.dumps(source_manifest))
    with pytest.raises(RuntimeError, match="token count"):
        load_m1_data(feature_dir, source_probability_dir, role)
    source_manifest["tokens"] -= 1
    source_manifest_path.write_text(json.dumps(source_manifest))

    np.savez(
        token_path,
        lengths=lengths,
        target=target + 0.01,
        **{role: draft},
    )
    with pytest.raises(RuntimeError, match="reconciled token cache content"):
        load_m1_data(feature_dir, probability_dir, role)


class _FakeDecoderLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = torch.nn.Identity()
        self.input_layernorm = torch.nn.Identity()

    def forward(self, hidden_states, **kwargs):
        return hidden_states


class _FakeQwen(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=28, hidden_size=2048)
        self.embedding = torch.nn.Embedding(8, 2048)
        self.layers = torch.nn.ModuleList([_FakeDecoderLayer() for _ in range(28)])
        self.lm_head = torch.nn.Linear(2048, 8, bias=False)

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        hidden = self.embedding(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=self.lm_head(hidden))


def test_extractor_uses_prediction_position_before_candidate_token() -> None:
    tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=7)
    record = SFTRecord(
        record_id="record",
        source="source",
        response_ids=(1, 2, 3),
        response_hash="hash",
        prompt_ids=(4, 5),
    )
    examples, lengths, offsets, token_ids, prediction_positions, input_positions, eos_mask = _build_position_metadata(
        [record], tokenizer, include_eos=True
    )
    assert examples[0]["input_ids"] == [4, 5, 1, 2, 3, 7]
    assert token_ids.tolist() == [1, 2, 3, 7]
    assert input_positions.tolist() == [2, 3, 4, 5]
    assert prediction_positions.tolist() == [1, 2, 3, 4]
    assert eos_mask.tolist() == [False, False, False, True]

    torch.manual_seed(3)
    model = _FakeQwen()
    q = np.empty((4, 6), dtype=np.float32)
    h = np.empty((4, 40), dtype=np.float32)
    extract_qh_features(
        model,
        examples,
        lengths,
        offsets,
        tokenizer,
        torch.device("cpu"),
        batch_size=1,
        q_output=q,
        h_output=h,
    )
    with torch.inference_mode():
        logits = model(torch.tensor([examples[0]["input_ids"]])).logits[0, 1:5]
        expected = torch.log_softmax(logits.float(), dim=-1).gather(
            -1, torch.tensor(token_ids).unsqueeze(-1)
        ).squeeze(-1)
    assert np.allclose(q[:, 0], expected.numpy(), atol=1e-6)
    assert np.allclose(q[:, 4], [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    assert np.allclose(q[:, 5], np.log(4.0))
    assert h.shape == (4, 40)
