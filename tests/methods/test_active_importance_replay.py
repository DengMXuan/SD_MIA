from pathlib import Path

import numpy as np

from experiments.sd_membership_sft.analysis.active_importance_replay import LAMBDAS, ReplayData, acceptance_probabilities, estimate_corrected_delta0, estimate_uncorrected_log_alpha, load_replay_data, oracle_action_levels, oracle_positions, oracle_schedule, q_lambda_from_logq, run_replay, sample_schedule, selected_ladder_schedule, uniform_schedule
from experiments.sd_membership_sft.analysis.token_signal_anatomy import Roles


def test_q_lambda_matches_registered_action_and_q0_one_is_no_action() -> None:
    logq = np.log(np.asarray([0.01, 0.6, 1.0]))
    assert np.allclose(q_lambda_from_logq(logq, 0.0), np.exp(logq))
    assert np.allclose(q_lambda_from_logq(logq, 0.5), np.sqrt(np.exp(logq)))
    assert np.allclose(q_lambda_from_logq(logq, 1.0), 1.0)
    for level in LAMBDAS:
        assert q_lambda_from_logq(np.asarray([0.0]), level)[0] == 1.0


def test_acceptance_probabilities_use_changed_q() -> None:
    logp = np.log(np.asarray([0.8, 0.3]))
    logq = np.log(np.asarray([0.6, 0.6]))
    alpha = acceptance_probabilities(logp, logq, (0.0, 1.0))
    assert np.allclose(alpha[:, 0], [1.0, 0.5])
    assert np.allclose(alpha[:, 1], [0.8, 0.3])


def test_uniform_and_oracle_schedules_have_exact_fair_budget_and_prefixes() -> None:
    length = 10
    for budget in (1, 2, 4, 8):
        fixed = uniform_schedule(length, budget, active=False)
        active = uniform_schedule(length, budget, active=True)
        assert np.sum(fixed >= 0) == np.sum(active >= 0) == budget * length
        if budget > 1:
            previous = uniform_schedule(length, budget - 1, active=True)
            assert np.array_equal(active[:, : budget - 1], previous)

    selected = np.asarray([1, 4, 8])
    levels = np.full(length, 4, dtype=np.int8)
    previous_counts = np.zeros(length, dtype=np.int64)
    for budget in (1, 2, 4, 8):
        schedule = oracle_schedule(length, budget, selected, levels)
        counts = np.sum(schedule >= 0, axis=1)
        assert np.sum(counts) == budget * length
        assert np.all(counts >= previous_counts)
        assert np.all(counts[np.setdiff1d(np.arange(length), selected)] == 1)
        previous_counts = counts

    previous_counts = np.zeros(length, dtype=np.int64)
    for budget in (1, 2, 4, 8):
        schedule = selected_ladder_schedule(length, budget, selected)
        counts = np.sum(schedule >= 0, axis=1)
        assert np.sum(counts) == budget * length
        assert np.all(counts >= previous_counts)
        assert np.all(counts[np.setdiff1d(np.arange(length), selected)] == 1)
        previous_counts = counts


def test_sampling_reuses_uniforms_and_higher_q_accepts_are_a_subset() -> None:
    logp = np.log(np.asarray([0.4, 0.8]))
    logq0 = np.log(np.asarray([0.3, 0.6]))
    alpha = acceptance_probabilities(logp, logq0)
    uniforms = np.asarray([[0.2, 0.5], [0.7, 0.9]])
    normal = np.zeros((2, 2), dtype=np.int8)
    raised = np.full((2, 2), 4, dtype=np.int8)
    normal_accepts, _ = sample_schedule(alpha, normal, uniforms)
    raised_accepts, _ = sample_schedule(alpha, raised, uniforms)
    assert np.all(raised_accepts[:, 4] <= normal_accepts[:, 0])


def test_vectorized_joint_mle_recovers_common_delta_for_mixed_counts() -> None:
    q0 = np.asarray([0.6, 0.2])
    p = np.asarray([0.3, 0.1])
    logq0 = np.log(q0)
    # Exact proportional counts at q0 and q=1 make p the joint MLE.
    trials = np.asarray([[200, 0, 0, 0, 200], [200, 0, 0, 0, 200]])
    accepts = np.asarray([[100, 0, 0, 0, 60], [100, 0, 0, 0, 20]])
    result = estimate_corrected_delta0(logq0, accepts, trials)
    assert np.all(result.censoring == 0)
    assert np.allclose(result.delta0, np.log(p / q0), atol=2e-10)


def test_uncorrected_negative_control_discards_q_coordinate() -> None:
    trials = np.asarray([[0, 0, 0, 0, 90]])
    accepts = np.asarray([[0, 0, 0, 0, 80]])
    wrong = estimate_uncorrected_log_alpha(accepts, trials)[0]
    correct = estimate_corrected_delta0(np.asarray([np.log(0.6)]), accepts, trials).delta0[0]
    assert np.isclose(wrong, np.log(80.5 / 91.0))
    assert not np.isclose(wrong, correct)
    assert np.isclose(correct, np.log((80 / 90) / 0.6), atol=2e-10)


def test_boundary_observations_stay_finite_but_are_marked_censored() -> None:
    logq0 = np.log(np.asarray([0.6, 0.6, 1.0]))
    trials = np.asarray([[16, 0, 0, 0, 0], [16, 0, 0, 0, 0], [16, 0, 0, 0, 0]])
    accepts = np.asarray([[16, 0, 0, 0, 0], [0, 0, 0, 0, 0], [16, 0, 0, 0, 0]])
    result = estimate_corrected_delta0(logq0, accepts, trials)
    assert np.all(np.isfinite(result.delta0))
    assert result.censoring.tolist() == [1, -1, 1]


def test_oracle_position_and_action_choices_are_deterministic() -> None:
    delta = np.asarray([-3.0, 2.0, 1.0, -0.5])
    assert oracle_positions(delta, 0.5, "absolute").tolist() == [0, 1]
    assert oracle_positions(delta, 0.5, "positive").tolist() == [1, 2]
    logp = np.log(np.asarray([0.8, 0.3]))
    logq = np.log(np.asarray([0.6, 0.6]))
    levels = oracle_action_levels(logp, logq, target_acceptance=0.5)
    alpha = acceptance_probabilities(logp, logq)
    assert np.array_equal(levels, np.argmin(np.abs(alpha - 0.5), axis=1))


def test_window_boundary_oracle_targets_tokens_in_near_zero_windows() -> None:
    # The first width-2 window sums to zero; the other windows are far from
    # the sign boundary.  Its two tokens should therefore be selected first.
    delta = np.asarray([1.0, -1.0, 5.0, 5.0])
    selected = oracle_positions(delta, 0.5, "window_boundary", window_width=2)
    assert set(selected.tolist()) == {0, 1}


def test_loader_drops_final_token_from_every_record(tmp_path: Path) -> None:
    labels = np.asarray([1, 0])
    record_ids = np.asarray(["m", "n"])
    lengths = np.asarray([3, 2])
    offsets = np.asarray([0, 3, 5])
    draft = np.log(np.asarray([0.6, 0.4, 1.0, 0.5, 1.0])).astype(np.float32)
    target = np.log(np.asarray([0.7, 0.3, 1.0, 0.4, 1.0])).astype(np.float32)
    full = tmp_path / "full.npz"
    pq = tmp_path / "pq.npz"
    np.savez(full, labels=labels, record_ids=record_ids, lengths=lengths, offsets=offsets, delta=target - draft)
    np.savez(pq, lengths=lengths, target=target, draft_auxiliary_distilled=draft)
    loaded = load_replay_data(full, pq)
    assert loaded.lengths.tolist() == [2, 1]
    assert loaded.offsets.tolist() == [0, 2, 3]
    assert np.allclose(loaded.logp, target[[0, 1, 3]])
    assert np.allclose(loaded.logq0, draft[[0, 1, 3]])


def test_small_end_to_end_replay_is_deterministic_and_budget_fair() -> None:
    lengths = np.full(8, 3, dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    q0 = np.tile(np.asarray([0.2, 0.4, 0.6]), 8)
    # T members (records 4/5) have larger absolute deltas than T nonmembers.
    delta = np.concatenate([
        np.full(3, value) for value in (0.1, -0.1, 0.0, 0.05, 0.8, 0.7, 0.0, 0.1)
    ])
    data = ReplayData(
        labels=np.asarray([0, 0, 0, 0, 1, 1, 0, 0]),
        record_ids=np.asarray([f"r{i}" for i in range(8)]),
        lengths=lengths,
        offsets=offsets,
        logp=np.log(q0) + delta,
        logq0=np.log(q0),
    )
    roles = Roles(
        n_ref=np.asarray([0, 1]),
        m_diag=np.asarray([], dtype=np.int64),
        n_cal=np.asarray([2, 3]),
        t_member=np.asarray([4, 5]),
        t_nonmember=np.asarray([6, 7]),
        reserved_member=np.asarray([], dtype=np.int64),
    )
    first, first_scores = run_replay(data, roles, budgets=(1, 2), seed=17)
    second, second_scores = run_replay(data, roles, budgets=(1, 2), seed=17)
    assert first == second
    assert all(np.array_equal(first_scores[key], second_scores[key], equal_nan=True) for key in first_scores)
    selected_tokens = 8 * 3
    for method in first["methods"].values():
        assert method["1"]["cost"]["verifier_decisions"] == selected_tokens
        assert method["2"]["cost"]["verifier_decisions"] == 2 * selected_tokens
