import dataclasses
import itertools
import json
import shutil

import numpy as np
import pytest
import torch

from experiments.sd_membership_sft import conditional_accept_only as conditional
from experiments.sd_membership_sft.analyze_conditional_accept_only import analyze, attach_legacy, ranking
from experiments.sd_membership_sft.full_delta_mia import partial_auc, rank_auc
from experiments.sd_membership_sft.collect_counterfactual_accept_only import paired_records, simulate_paired_bits
from experiments.sd_membership_sft.data import SFTRecord, make_sft_example


@pytest.fixture(autouse=True)
def small_cpu_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observations(views=2):
    rng = np.random.default_rng(13)
    return conditional.Observations(
        -rng.uniform(0.01, 8, (96, views)),
        rng.integers(0, 2, (96, views, 4), dtype=np.uint8),
        np.full(12, 8, dtype=np.int64),
        ("original", "truncated_context")[:views],
    )


@pytest.mark.parametrize("k", [1, 2, 8])
def test_count_distribution_is_normalized_with_finite_boundary_gradients(k):
    model = conditional.ConditionalCountTCN(2, k, 8)
    mask = torch.ones(2, 7, dtype=torch.bool)
    output = model(torch.randn(2, 7, 2), mask)
    torch.testing.assert_close(output.exp().sum(-1), torch.ones(2, 7), atol=1e-6, rtol=1e-6)
    counts = torch.tensor([[0] * 7, [k] * 7])
    conditional.count_nll(output, counts, mask).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())


def test_padding_length_and_other_batch_records_do_not_change_predictions():
    torch.manual_seed(9)
    model = conditional.ConditionalCountTCN(2, 2, 8).eval()
    x = torch.randn(1, 9, 2)
    alone = model(x, torch.ones(1, 9, dtype=torch.bool))
    batch = torch.randn(2, 21, 2) * 1000
    batch[0, :9] = x[0]
    mask = torch.ones(2, 21, dtype=torch.bool)
    mask[0, 9:] = False
    together = model(batch, mask)[0, :9]
    torch.testing.assert_close(alone[0], together, atol=2e-6, rtol=2e-6)


def test_paired_inputs_and_fusion_base_cannot_read_unbudgeted_bits():
    obs = observations()
    changed = obs.bits.copy()
    changed[:, :, 1:] = 1 - changed[:, :, 1:]
    modified = dataclasses.replace(obs, bits=changed)
    for before, after in zip(conditional.observable_inputs(obs, 2, True), conditional.observable_inputs(modified, 2, True)):
        np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(conditional.lowq_score(obs, 1, np.arange(4)), conditional.lowq_score(modified, 1, np.arange(4)))
    original_features = conditional.observable_inputs(obs, 2, False)[0]
    original_changed = conditional.observable_inputs(dataclasses.replace(obs, bits=1 - obs.bits), 2, False)[0]
    np.testing.assert_array_equal(original_features, original_changed)
    with pytest.raises(ValueError, match="even budget"):
        conditional.observable_inputs(obs, 3, True)


def test_fit_is_invariant_to_all_heldout_bits_and_q_values():
    obs = observations()
    bits, logq = obs.bits.copy(), obs.logq.copy()
    bits[64:] = 1 - bits[64:]
    logq[64:] *= 9
    modified = dataclasses.replace(obs, bits=bits, logq=logq)
    kwargs = dict(budget=2, paired=True, seed=4, device=torch.device("cpu"), epochs=2, channels=4)
    first = conditional.fit_null(obs, np.arange(6), np.arange(6, 8), **kwargs)
    second = conditional.fit_null(modified, np.arange(6), np.arange(6, 8), **kwargs)
    assert first.history == second.history
    for key, weight in first.model.state_dict().items():
        torch.testing.assert_close(weight, second.model.state_dict()[key], equal_nan=True)


def test_split_contract_rejects_members_and_cross_partition_reuse():
    ids = np.asarray([f"r{i}" for i in range(8)])
    labels = np.asarray([0, 0, 0, 0, 0, 0, 1, 1])
    parts = {"train": np.array([0, 1]), "validation": np.array([2]), "calibration": np.array([3]), "test": np.array([4, 6])}
    conditional.assert_partition_contract(labels, ids, parts)
    with pytest.raises(ValueError, match="member labels"):
        conditional.assert_partition_contract(labels, ids, {**parts, "train": np.array([6])})
    with pytest.raises(ValueError, match="overlap"):
        conditional.assert_partition_contract(labels, ids, {**parts, "test": np.array([3, 6])})


def test_span_score_matches_exhaustive_latent_path_marginalization():
    pmf = np.asarray([[0.4, 0.6], [0.7, 0.3], [0.2, 0.8]])
    counts = np.asarray([1, 0, 1])
    _, actual = conditional.directional_scores(np.log(pmf), counts)
    enter, leave = 1 / 64, 1 / 8
    prior = enter / (enter + leave)
    transition = np.asarray([[1 - enter, enter], [leave, 1 - leave]])
    evidence = []
    for tilt in (0.5, 1.0, 2.0):
        ratios = np.exp(tilt * counts) / (pmf[:, 0] + pmf[:, 1] * np.exp(tilt))
        marginal = 0.0
        for path in itertools.product((0, 1), repeat=3):
            weight = prior if path[0] else 1 - prior
            for i in range(1, 3):
                weight *= transition[path[i - 1], path[i]]
            marginal += weight * np.prod([ratios[i] if state else 1 for i, state in enumerate(path)])
        evidence.append(marginal)
    assert actual == pytest.approx(np.log(np.mean(evidence)))
    rejected = conditional.directional_scores(np.log(pmf), np.zeros(3, dtype=int))
    accepted = conditional.directional_scores(np.log(pmf), np.ones(3, dtype=int))
    assert all(a > r for a, r in zip(accepted, rejected))


def test_archive_forbids_target_probabilities(tmp_path):
    obs = observations()
    path = tmp_path / "observations.npz"
    conditional.save_observations(path, obs, np.zeros(12), np.asarray([f"r{i}" for i in range(12)]))
    recovered, _, _ = conditional.load_observations(path)
    np.testing.assert_array_equal(obs.bits, recovered.bits)
    with np.load(path) as source:
        values = dict(source)
    np.savez(path, **values, logp=np.ones(96))
    with pytest.raises(ValueError, match="forbidden"):
        conditional.load_observations(path)


def test_legacy_comparison_rejects_different_records_or_query_observations(tmp_path):
    archive = {"labels": np.array([0, 1]), "record_ids": np.array(["a", "b"]), "lowq": np.array([0.1, 0.9])}
    path = tmp_path / "legacy.npz"
    common = dict(labels=archive["labels"], record_ids=archive["record_ids"], lowq_plus_neural_k2=np.array([0.2, 0.8]))
    np.savez(path, **common, lowq_k2=archive["lowq"])
    attach_legacy(archive, path)
    np.testing.assert_array_equal(archive["legacy_neural_fusion"], [0.2, 0.8])
    np.savez(path, **common, lowq_k2=np.array([0.1, 0.8]))
    with pytest.raises(ValueError, match="baseline differs"):
        attach_legacy(archive, path)
    np.savez(path, **{**common, "record_ids": np.array(["b", "a"])}, lowq_k2=archive["lowq"])
    with pytest.raises(ValueError, match="alignment failed"):
        attach_legacy(archive, path)


def test_vectorized_bootstrap_ranking_matches_registered_tie_and_tail_rules():
    rng = np.random.default_rng(57)
    for count in (2, 17, 100):
        member, nonmember = np.arange(count), np.arange(count, 3 * count)
        labels = np.r_[np.ones(count), np.zeros(2 * count)]
        for scores in (rng.normal(size=3 * count), rng.integers(0, 5, size=3 * count), np.ones(3 * count)):
            expected = [rank_auc(scores[member], scores[nonmember]), partial_auc(scores, labels)]
            np.testing.assert_allclose(ranking(scores, member, nonmember), expected, atol=1e-12, rtol=1e-12)


def test_counterfactual_changes_only_prefix_and_keeps_candidates_without_eos():
    record = SFTRecord("r", "s", tuple(range(20)), "hash", prompt_ids=(100, 101))
    tokenizer = type("Tokenizer", (), {"eos_token_id": 999})()
    original, truncated = paired_records([record], tokenizer, candidate_tokens=4, keep_context=3)
    assert original[0].response_ids == truncated[0].response_ids == (16, 17, 18, 19)
    assert original[0].prompt_ids == (100, 101, *range(16))
    assert truncated[0].prompt_ids == (100, 101, 13, 14, 15)
    for view in (original, truncated):
        example = make_sft_example(view[0], tokenizer)
        assert [x for x in example["labels"] if x != -100] == [16, 17, 18, 19]
        assert 999 not in example["input_ids"]
    with pytest.raises(ValueError, match="insufficient"):
        paired_records([record], tokenizer, candidate_tokens=18, keep_context=3)


def test_simulator_is_independent_across_views_and_has_budget_stable_prefixes():
    q = np.full((200, 2), -1.0)
    p = q + np.log(0.5)
    lengths = np.array([100, 100])
    short = simulate_paired_bits(p, q, lengths, repeats=2, seed=5)
    long = simulate_paired_bits(p, q, lengths, repeats=4, seed=5)
    np.testing.assert_array_equal(short.bits, long.bits[:, :, :2])
    assert 0.35 < np.mean(short.bits[:, 0] != short.bits[:, 1]) < 0.65
    saturated = simulate_paired_bits(q + 0.5, q, lengths, repeats=2, seed=5)
    assert saturated.bits.all()


def test_paired_pipeline_writes_models_and_equal_budget_report(tmp_path, monkeypatch):
    obs = observations()
    labels = np.array([0] * 10 + [1] * 2)
    ids = np.asarray([f"r{i}" for i in range(12)])
    parts = {"train": np.arange(4), "validation": np.arange(4, 6), "reference": np.arange(6),
             "calibration": np.arange(6, 8), "test": np.arange(8, 12)}
    monkeypatch.setattr(conditional, "record_partitions", lambda *_: parts)
    output = tmp_path / "paired" / "b2_seed3"
    report = conditional.evaluate(obs, labels, ids, budget=2, output=output, device=torch.device("cpu"),
                                  seed=3, epochs=2, patience=1, channels=4)
    assert report["training_member_count"] == report["synthetic_member_count"] == 0
    for costs in report["training"].values():
        assert costs["original_queries"] + costs["counterfactual_queries"] == 2
    assert set(report["metrics"]) >= {"original_span", "paired_span", "paired_fusion", "lowq"}
    assert (output / "REPORT.md").exists()
    checkpoint = torch.load(output / "paired.pt", weights_only=True)
    restored = conditional.ConditionalCountTCN(checkpoint["input_dim"], checkpoint["k"], checkpoint["channels"])
    restored.load_state_dict(checkpoint["state_dict"])
    source = {"observations": "fixture.npz", "sha256": "first_bits", "candidate_scope_sha256": "same_candidates"}
    (output / "SOURCE.json").write_text(json.dumps(source))
    single = analyze(tmp_path, 2, None, repeats=5, scope="paired")
    assert single["condition_count"] == single["run_count"] == 1
    with pytest.raises(FileNotFoundError, match="no cached"):
        analyze(tmp_path, 2, None, repeats=5, scope="cached")
    duplicate = output.with_name("b2_seed4")
    shutil.copytree(output, duplicate)
    (duplicate / "SOURCE.json").write_text(json.dumps({**source, "sha256": "second_bits"}))
    repeated = analyze(tmp_path, 2, None, repeats=5, scope="paired")
    assert repeated["condition_count"] == 1 and repeated["run_count"] == 2
    # Duplicating an identical fit cannot narrow a paired record interval.
    assert single["comparisons"] == repeated["comparisons"]
    other = tmp_path / "other_checkpoint" / "b2_seed3"
    shutil.copytree(output, other)
    (other / "SOURCE.json").write_text(json.dumps({**source, "candidate_scope_sha256": "different_checkpoint_same_records"}))
    checkpoints = analyze(tmp_path, 2, None, repeats=5, scope="paired")
    assert checkpoints["condition_count"] == 2
    assert single["comparisons"] == checkpoints["comparisons"]
