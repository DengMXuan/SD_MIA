"""E3 offline replay for q-corrected accept-only probing.

The replay consumes the frozen exact ``log p``/``log q0`` cache, removes the
producer-appended final token from every record, and samples position-locked
verifier bits.  Every method at a given ``K_eq`` receives exactly ``K_eq * L``
decisions for a record of length ``L``.  Methods are coupled with the same
per-token uniform-random-number prefixes, so their differences are not driven
by unrelated Monte Carlo noise.

This program is deliberately an offline Gate-2 instrument.  In particular,
``oracle_importance_active_q`` uses exact delta to allocate probes and exact p
to choose a q action.  It is an upper bound, not a deployable attack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.shared.core.audit_metrics import conformal_tail_pvalues
from experiments.shared.core.audit_metrics import partial_auc, rank_auc
from experiments.shared.core.replay_cache import load_delta_data, sliding_means
from experiments.sd_membership_sft.analysis.token_signal_anatomy import Roles, build_roles
from experiments.shared.core.replay_cache import drop_final_cached_token, load_paired_logps

from experiments.shared.core.replay_cache import ReplayData, load_replay_data


from experiments.paths import ROOT
BENCHMARKS = ("wikitection", "newstection", "arxivtection")
EPOCHS = (1, 3)
BUDGETS = (1, 2, 4, 8, 16, 32)
LAMBDAS = (0.0, 0.25, 0.5, 0.75, 1.0)
METHODS = (
    "fixed_q",
    "active_q_uncorrected",
    "active_q_corrected_uniform",
    "fixed_q_lowq_score",
    "active_q_corrected_uniform_lowq_score",
    "fixed_q_lowq_multiscale",
    "active_q_corrected_uniform_lowq_multiscale",
    "q_low_importance_active_q",
    "oracle_importance_active_q",
)
RATES = (0.01, 0.05, 0.10)
LOWQ_SCORE_FRACTIONS = (0.10, 0.20, 0.50)




@dataclass(frozen=True)
class ReplayEstimate:
    delta0: np.ndarray
    censoring: np.ndarray


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_jsonable(value), handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()




def q_lambda_from_logq(logq0: np.ndarray, lambda_value: float) -> np.ndarray:
    """Return q_lambda(y)=q0(y)**(1-lambda), including q0=1 no-action."""
    if not 0.0 <= float(lambda_value) <= 1.0:
        raise ValueError("lambda must lie in [0, 1]")
    values = np.asarray(logq0, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values > 1e-9):
        raise ValueError("logq0 must be finite and no greater than zero")
    return np.exp((1.0 - float(lambda_value)) * values)


def acceptance_probabilities(
    logp: np.ndarray, logq0: np.ndarray, lambdas: Iterable[float] = LAMBDAS
) -> np.ndarray:
    """Return token-by-action min(1,p/q_lambda) without exponentiating p/q."""
    logp = np.asarray(logp, dtype=np.float64)
    logq0 = np.asarray(logq0, dtype=np.float64)
    if logp.shape != logq0.shape or logp.ndim != 1:
        raise ValueError("logp and logq0 must be aligned vectors")
    levels = np.asarray(tuple(lambdas), dtype=np.float64)
    if levels.ndim != 1 or len(levels) == 0 or np.any((levels < 0.0) | (levels > 1.0)):
        raise ValueError("lambdas must be a nonempty vector in [0, 1]")
    logq = logq0[:, None] * (1.0 - levels[None, :])
    return np.exp(np.minimum(0.0, logp[:, None] - logq))


def uniform_schedule(length: int, budget: int, active: bool) -> np.ndarray:
    """Build an exact-budget all-position schedule; every budget is a prefix."""
    if length <= 0 or budget <= 0:
        raise ValueError("length and budget must be positive")
    if not active:
        levels = np.zeros(budget, dtype=np.int8)
    else:
        # One normal pilot, followed by the registered nonzero q ladder.
        cycle = np.asarray([1, 2, 3, 4], dtype=np.int8)
        levels = np.r_[np.int8(0), np.resize(cycle, max(0, budget - 1))]
    return np.broadcast_to(levels, (length, budget)).copy()


def oracle_positions(
    delta0: np.ndarray,
    fraction: float,
    signal: str,
    window_width: int = 8,
) -> np.ndarray:
    """Return exact-delta top positions for the explicitly nondeployable oracle."""
    values = np.asarray(delta0, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("delta0 must be a nonempty vector")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    if signal == "positive":
        evidence = np.maximum(values, 0.0)
    elif signal == "negative":
        evidence = np.maximum(-values, 0.0)
    elif signal == "absolute":
        evidence = np.abs(values)
    elif signal == "window_boundary":
        if window_width <= 0:
            raise ValueError("window_width must be positive")
        width = min(window_width, len(values))
        window_sums = np.convolve(values, np.ones(width, dtype=np.float64), mode="valid")
        nonzero = np.abs(window_sums[np.abs(window_sums) > 0.0])
        scale = float(np.median(nonzero)) if len(nonzero) else 1.0
        scale = max(scale, np.finfo(np.float64).eps)
        # A token matters to window-sign scoring when it belongs to windows
        # whose exact sum is close to the zero decision boundary.  Summing the
        # smooth boundary weights over all containing windows avoids choosing
        # a single accidental extremum and is deterministic on ties.
        window_weight = np.exp(-np.abs(window_sums) / scale)
        difference = np.zeros(len(values) + 1, dtype=np.float64)
        starts = np.arange(len(window_weight))
        np.add.at(difference, starts, window_weight)
        np.add.at(difference, starts + width, -window_weight)
        evidence = np.cumsum(difference[:-1])
    else:
        raise ValueError(f"unknown oracle signal {signal!r}")
    count = min(len(values), max(1, int(math.ceil(fraction * len(values)))))
    return np.argsort(-evidence, kind="stable")[:count]


def oracle_action_levels(
    logp: np.ndarray,
    logq0: np.ndarray,
    target_acceptance: float = 0.5,
    lambdas: Iterable[float] = LAMBDAS,
) -> np.ndarray:
    """Choose the legal q whose exact acceptance is closest to a mixed regime."""
    if not 0.0 < target_acceptance < 1.0:
        raise ValueError("target_acceptance must lie in (0, 1)")
    alpha = acceptance_probabilities(logp, logq0, lambdas)
    # Stable argmin deliberately favors the smaller q modification on ties.
    return np.argmin(np.abs(alpha - target_acceptance), axis=1).astype(np.int8)


def oracle_schedule(
    length: int,
    budget: int,
    selected: np.ndarray,
    action_levels: np.ndarray,
) -> np.ndarray:
    """Allocate exactly budget*length decisions after a one-per-token pilot."""
    if length <= 0 or budget <= 0:
        raise ValueError("length and budget must be positive")
    selected = np.asarray(selected, dtype=np.int64)
    action_levels = np.asarray(action_levels, dtype=np.int8)
    if action_levels.shape != (length,):
        raise ValueError("action_levels must have one entry per token")
    if len(selected) == 0 or len(np.unique(selected)) != len(selected):
        raise ValueError("selected positions must be nonempty and unique")
    if np.any((selected < 0) | (selected >= length)):
        raise ValueError("selected position is out of range")
    extra = (budget - 1) * length
    base, remainder = divmod(extra, len(selected))
    extra_counts = np.full(len(selected), base, dtype=np.int64)
    extra_counts[:remainder] += 1
    maximum = 1 + int(np.max(extra_counts, initial=0))
    schedule = np.full((length, maximum), -1, dtype=np.int8)
    schedule[:, 0] = 0
    for rank, position in enumerate(selected):
        schedule[position, 1 : 1 + extra_counts[rank]] = action_levels[position]
    if int(np.sum(schedule >= 0)) != budget * length:
        raise RuntimeError("oracle allocation violated the exact decision budget")
    return schedule


def selected_ladder_schedule(length: int, budget: int, selected: np.ndarray) -> np.ndarray:
    """Allocate extras to q-only selected positions while cycling legal q levels."""
    selected = np.asarray(selected, dtype=np.int64)
    if length <= 0 or budget <= 0 or len(selected) == 0:
        raise ValueError("length, budget, and selected positions must be positive/nonempty")
    if len(np.unique(selected)) != len(selected) or np.any((selected < 0) | (selected >= length)):
        raise ValueError("selected positions must be unique and in range")
    extra = (budget - 1) * length
    base, remainder = divmod(extra, len(selected))
    extra_counts = np.full(len(selected), base, dtype=np.int64)
    extra_counts[:remainder] += 1
    maximum = 1 + int(np.max(extra_counts, initial=0))
    schedule = np.full((length, maximum), -1, dtype=np.int8)
    schedule[:, 0] = 0
    ladder = np.asarray([1, 2, 3, 4], dtype=np.int8)
    for rank, position in enumerate(selected):
        schedule[position, 1 : 1 + extra_counts[rank]] = np.resize(ladder, extra_counts[rank])
    if int(np.sum(schedule >= 0)) != budget * length:
        raise RuntimeError("selected ladder allocation violated the exact decision budget")
    return schedule


def sample_schedule(
    alpha: np.ndarray, schedule: np.ndarray, uniforms: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate scheduled Bernoulli draws into token-by-q accept/trial counts."""
    alpha = np.asarray(alpha, dtype=np.float64)
    schedule = np.asarray(schedule)
    uniforms = np.asarray(uniforms, dtype=np.float64)
    if alpha.ndim != 2 or schedule.ndim != 2:
        raise ValueError("alpha and schedule must be matrices")
    if schedule.shape[0] != alpha.shape[0] or uniforms.shape[0] != alpha.shape[0]:
        raise ValueError("token dimensions are not aligned")
    if uniforms.shape[1] < schedule.shape[1]:
        raise ValueError("uniform prefix is shorter than the schedule")
    if np.any((schedule < -1) | (schedule >= alpha.shape[1])):
        raise ValueError("schedule contains an invalid action level")
    trials = np.zeros_like(alpha, dtype=np.int64)
    accepts = np.zeros_like(alpha, dtype=np.int64)
    for level in range(alpha.shape[1]):
        chosen = schedule == level
        trials[:, level] = np.sum(chosen, axis=1)
        accepts[:, level] = np.sum(
            chosen & (uniforms[:, : schedule.shape[1]] < alpha[:, level, None]), axis=1
        )
    return accepts, trials


def estimate_corrected_delta0(
    logq0: np.ndarray,
    accepts: np.ndarray,
    trials: np.ndarray,
    lambdas: Iterable[float] = LAMBDAS,
    bisection_steps: int = 42,
) -> ReplayEstimate:
    """Vectorized joint-likelihood estimate in the common delta0 coordinate.

    Interior MLEs solve the exact multi-q score equation.  Boundary likelihoods
    remain explicitly marked as censored.  A finite representative is used for
    RMSE/scoring only: the geometric midpoint of an all-accept plateau, and a
    local-likelihood half-count scale for all-reject observations.  These
    representatives never change the reported censoring state.
    """
    logq0 = np.asarray(logq0, dtype=np.float64)
    accepts = np.asarray(accepts, dtype=np.int64)
    trials = np.asarray(trials, dtype=np.int64)
    levels = np.asarray(tuple(lambdas), dtype=np.float64)
    if accepts.shape != trials.shape or accepts.shape != (len(logq0), len(levels)):
        raise ValueError("accepts/trials must be token-by-lambda matrices")
    if np.any(trials < 0) or np.any(accepts < 0) or np.any(accepts > trials):
        raise ValueError("invalid accept/trial counts")
    total_trials = np.sum(trials, axis=1)
    total_accepts = np.sum(accepts, axis=1)
    if np.any(total_trials <= 0):
        raise ValueError("every token must have at least one observation")

    logq = logq0[:, None] * (1.0 - levels[None, :])
    q = np.exp(logq)
    observed = trials > 0
    rejects = trials - accepts
    all_reject = total_accepts == 0
    all_accept = total_accepts == total_trials
    interior = ~(all_reject | all_accept)
    p_hat = np.empty(len(logq0), dtype=np.float64)
    censoring = np.zeros(len(logq0), dtype=np.int8)  # -1 upper, 0 none, +1 lower

    if np.any(all_reject):
        # Near p=0, log L(p) = -p*sum(K_j/q_j)+O(p^2).  Half a unit on this
        # likelihood scale supplies a finite summary without claiming recovery.
        information = np.sum(np.where(observed, trials / q, 0.0), axis=1)
        p_hat[all_reject] = 0.5 / information[all_reject]
        censoring[all_reject] = -1

    if np.any(all_accept):
        q_max = np.max(np.where(observed, q, 0.0), axis=1)
        # If max(q)<1, the likelihood plateau is [max(q_j),1] and we expose
        # its log midpoint.  At q=1 the MLE set is the singleton {1}, but the
        # finite Bernoulli sample still supplies only a lower confidence bound,
        # so it remains explicitly marked as censored.
        p_hat[all_accept] = np.sqrt(q_max[all_accept])
        censoring[all_accept] = 1

    if np.any(interior):
        rows = np.flatnonzero(interior)
        a = accepts[rows].astype(np.float64)
        r = rejects[rows].astype(np.float64)
        n = trials[rows]
        qr = q[rows]
        rejected_q = np.where(r > 0.0, qr, np.inf)
        hi = np.min(rejected_q, axis=1) * (1.0 - 1e-13)
        lo = np.zeros(len(rows), dtype=np.float64)
        for _ in range(bisection_steps):
            mid = (lo + hi) / 2.0
            unsaturated = mid[:, None] < qr
            with np.errstate(divide="ignore", invalid="ignore"):
                derivative = np.sum(
                    np.where(
                        (n > 0) & unsaturated,
                        a / mid[:, None] - r / (qr - mid[:, None]),
                        0.0,
                    ),
                    axis=1,
                )
            move_right = derivative > 0.0
            lo = np.where(move_right, mid, lo)
            hi = np.where(move_right, hi, mid)
        p_hat[rows] = (lo + hi) / 2.0

    p_hat = np.clip(p_hat, np.finfo(np.float64).tiny, 1.0)
    return ReplayEstimate(delta0=np.log(p_hat) - logq0, censoring=censoring)


def estimate_uncorrected_log_alpha(accepts: np.ndarray, trials: np.ndarray) -> np.ndarray:
    """Old-B2 negative control: pool active-q bits and omit every q correction."""
    accepts = np.asarray(accepts, dtype=np.float64)
    trials = np.asarray(trials, dtype=np.float64)
    if accepts.shape != trials.shape or accepts.ndim != 2:
        raise ValueError("accepts and trials must be aligned matrices")
    total = np.sum(trials, axis=1)
    if np.any(total <= 0):
        raise ValueError("every token must have at least one observation")
    return np.log((np.sum(accepts, axis=1) + 0.5) / (total + 1.0))


def fragment_score(delta: np.ndarray, kind: str, top_fraction: float = 0.10) -> float:
    values = np.asarray(delta, dtype=np.float64)
    if len(values) == 0 or np.any(~np.isfinite(values)):
        raise ValueError("fragment delta must be finite and nonempty")
    if kind == "mean_abs":
        return float(np.mean(np.abs(values)))
    if kind == "mean_positive":
        return float(np.mean(np.maximum(values, 0.0)))
    if kind == "mean_signed":
        return float(np.mean(values))
    if kind == "positive_fraction":
        return float(np.mean(values > 0.0))
    if kind == "window_sign_8":
        return float(np.mean(sliding_means(values, 8) > 0.0))
    if kind in ("top_positive", "top_absolute"):
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError("top_fraction must lie in (0, 1]")
        evidence = np.maximum(values, 0.0) if kind == "top_positive" else np.abs(values)
        count = min(len(values), max(1, int(math.ceil(top_fraction * len(values)))))
        chosen = np.partition(evidence, len(evidence) - count)[-count:]
        return float(np.mean(chosen))
    raise ValueError(f"unknown fragment score {kind!r}")


def _record_uniforms(seed: int, record_index: int, length: int, width: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([seed, record_index]))
    return rng.random((length, width), dtype=np.float64)


def _record_indices(roles: Roles) -> np.ndarray:
    # M_diag and reserved C members are intentionally absent in E3.
    values = np.concatenate((roles.n_ref, roles.n_cal, roles.t_member, roles.t_nonmember))
    if len(np.unique(values)) != len(values):
        raise ValueError("E3 roles overlap")
    return np.sort(values)


def _role_token_mask(data: ReplayData, indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(data.logp), dtype=bool)
    for index in indices:
        mask[int(data.offsets[index]) : int(data.offsets[index + 1])] = True
    return mask


def _membership_metrics(scores: np.ndarray, data: ReplayData, roles: Roles) -> dict[str, Any]:
    calibration = scores[roles.n_cal]
    member = scores[roles.t_member]
    nonmember = scores[roles.t_nonmember]
    if np.any(~np.isfinite(np.r_[calibration, member, nonmember])):
        raise ValueError("required role has a missing/nonfinite fragment score")
    labels = np.r_[np.ones(len(member), dtype=np.int64), np.zeros(len(nonmember), dtype=np.int64)]
    result: dict[str, Any] = {
        "auc": rank_auc(member, nonmember),
        "pauc_0_10": partial_auc(np.r_[member, nonmember], labels, max_fpr=0.10),
        "calibration_nonmember_count": len(calibration),
        "test_member_count": len(member),
        "test_nonmember_count": len(nonmember),
        "tpr_at_fpr": {},
    }
    for rate in RATES:
        member_p = conformal_tail_pvalues(member, calibration)
        nonmember_p = conformal_tail_pvalues(nonmember, calibration)
        result["tpr_at_fpr"][f"{int(rate * 100)}%"] = {
            "tpr": float(np.mean(member_p <= rate)),
            "actual_fpr": float(np.mean(nonmember_p <= rate)),
        }
    return result


def run_replay(
    data: ReplayData,
    roles: Roles,
    *,
    budgets: tuple[int, ...] = BUDGETS,
    seed: int = 20260914,
    score_kind: str = "window_sign_8",
    top_fraction: float = 0.10,
    oracle_fraction: float = 0.10,
    oracle_signal: str = "window_boundary",
    importance_fraction: float = 0.20,
    target_acceptance: float = 0.5,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Replay one condition with exact roles, budgets, and common randomness."""
    if not budgets or any(int(value) <= 0 for value in budgets):
        raise ValueError("budgets must be positive")
    if not 0.0 < importance_fraction <= 1.0:
        raise ValueError("importance_fraction must lie in (0, 1]")
    budgets = tuple(sorted(set(map(int, budgets))))
    selected_records = _record_indices(roles)
    scores = {
        method: {budget: np.full(len(data.labels), np.nan) for budget in budgets}
        for method in METHODS
    }
    multiscale_raw = {
        method: {budget: np.full((len(data.labels), len(LOWQ_SCORE_FRACTIONS)), np.nan) for budget in budgets}
        for method in ("fixed_q_lowq_multiscale", "active_q_corrected_uniform_lowq_multiscale")
    }
    sse = {method: {budget: 0.0 for budget in budgets} for method in METHODS}
    token_count = {method: {budget: 0 for budget in budgets} for method in METHODS}
    censored = {method: {budget: 0 for budget in budgets} for method in METHODS}
    modified_decisions = {method: {budget: 0 for budget in budgets} for method in METHODS}
    total_decisions = {method: {budget: 0 for budget in budgets} for method in METHODS}

    for record_index in selected_records:
        start, end = int(data.offsets[record_index]), int(data.offsets[record_index + 1])
        logp = data.logp[start:end]
        logq0 = data.logq0[start:end]
        truth = logp - logq0
        length = len(logp)
        alpha = acceptance_probabilities(logp, logq0)
        important = oracle_positions(truth, oracle_fraction, oracle_signal)
        q_low = np.argsort(logq0, kind="stable")[
            : min(length, max(1, int(math.ceil(importance_fraction * length))))
        ]
        q_order = np.argsort(logq0, kind="stable")
        oracle_levels = oracle_action_levels(logp, logq0, target_acceptance)
        max_extra = (max(budgets) - 1) * length
        max_oracle_width = 1 + int(math.ceil(max_extra / len(important)))
        uniforms = _record_uniforms(seed, int(record_index), length, max(max(budgets), max_oracle_width))

        for budget in budgets:
            schedules = {
                "fixed_q": uniform_schedule(length, budget, active=False),
                "active_q_uncorrected": uniform_schedule(length, budget, active=True),
                "active_q_corrected_uniform": uniform_schedule(length, budget, active=True),
                "fixed_q_lowq_score": uniform_schedule(length, budget, active=False),
                "active_q_corrected_uniform_lowq_score": uniform_schedule(length, budget, active=True),
                "fixed_q_lowq_multiscale": uniform_schedule(length, budget, active=False),
                "active_q_corrected_uniform_lowq_multiscale": uniform_schedule(length, budget, active=True),
                "q_low_importance_active_q": selected_ladder_schedule(length, budget, q_low),
                "oracle_importance_active_q": oracle_schedule(
                    length, budget, important, oracle_levels
                ),
            }
            observation_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
            corrected_cache: dict[bytes, ReplayEstimate] = {}
            for method in METHODS:
                schedule = schedules[method]
                key = schedule.tobytes()
                if key not in observation_cache:
                    observation_cache[key] = sample_schedule(alpha, schedule, uniforms)
                accepts, trials = observation_cache[key]
                if method == "active_q_uncorrected":
                    estimate = estimate_uncorrected_log_alpha(accepts, trials)
                    censor = np.zeros(length, dtype=np.int8)
                    # Old B2 scores log alpha directly, not as if it were delta0.
                    score = float(np.mean(estimate))
                else:
                    if key not in corrected_cache:
                        corrected_cache[key] = estimate_corrected_delta0(logq0, accepts, trials)
                    recovered = corrected_cache[key]
                    estimate, censor = recovered.delta0, recovered.censoring
                    if method in {
                        "fixed_q_lowq_score",
                        "active_q_corrected_uniform_lowq_score",
                        "q_low_importance_active_q",
                    }:
                        score = fragment_score(estimate[q_low], "positive_fraction")
                    elif method in multiscale_raw:
                        raw = np.asarray([
                            fragment_score(
                                estimate[q_order[:max(1, int(math.ceil(fraction * length)))]],
                                "positive_fraction",
                            )
                            for fraction in LOWQ_SCORE_FRACTIONS
                        ])
                        multiscale_raw[method][budget][record_index] = raw
                        score = float("nan")
                    else:
                        score = fragment_score(estimate, score_kind, top_fraction)
                scores[method][budget][record_index] = score
                sse[method][budget] += float(np.sum((estimate - truth) ** 2))
                token_count[method][budget] += length
                censored[method][budget] += int(np.sum(censor != 0))
                q_actions = np.asarray(
                    [q_lambda_from_logq(logq0, level) for level in LAMBDAS]
                ).T
                active_decision = np.zeros_like(schedule, dtype=bool)
                for level in range(len(LAMBDAS)):
                    active_decision |= (schedule == level) & (
                        q_actions[:, level, None] > np.exp(logq0)[:, None] * (1.0 + 1e-12)
                    )
                modified_decisions[method][budget] += int(np.sum(active_decision))
                total_decisions[method][budget] += int(np.sum(schedule >= 0))
                if int(np.sum(schedule >= 0)) != budget * length:
                    raise RuntimeError("method violated exact K_eq budget")

    # The unknown-sparsity score chooses no fraction using member labels.
    # Each fixed candidate is standardized on N_ref, then their maximum is
    # calibrated as a whole on N_cal by _membership_metrics.
    for method, by_budget in multiscale_raw.items():
        for budget, raw in by_budget.items():
            fit = raw[roles.n_ref]
            center = np.mean(fit, axis=0)
            scale = np.std(fit, axis=0)
            scale = np.where(scale < 1e-8, 1.0, scale)
            scores[method][budget] = np.max((raw - center) / scale, axis=1)

    report_methods: dict[str, Any] = {}
    flat_scores: dict[str, np.ndarray] = {}
    for method in METHODS:
        report_methods[method] = {}
        for budget in budgets:
            method_scores = scores[method][budget]
            key = f"{method}_keq{budget}"
            flat_scores[key] = method_scores.astype(np.float32)
            count = token_count[method][budget]
            decisions = total_decisions[method][budget]
            expected_decisions = budget * int(np.sum(data.lengths[selected_records]))
            if decisions != expected_decisions:
                raise RuntimeError("aggregate decision count is not fair")
            report_methods[method][str(budget)] = {
                "membership": _membership_metrics(method_scores, data, roles),
                "measurement": {
                    "delta0_rmse": math.sqrt(sse[method][budget] / count),
                    "censored_token_fraction": censored[method][budget] / count,
                },
                "cost": {
                    "K_eq": budget,
                    "verifier_decisions": decisions,
                    "modified_q_decisions": modified_decisions[method][budget],
                    "modified_q_decision_fraction": modified_decisions[method][budget] / decisions,
                },
            }
    report = {
        "experiment": "E3 offline q-corrected active-importance replay",
        "protocol": {
            "budgets": list(budgets),
            "lambdas": list(LAMBDAS),
            "seed": seed,
            "randomness": "same per-record/per-token uniform prefixes across methods and budgets",
            "eos_policy": "drop producer-appended final token from every record",
            "score_kind": score_kind,
            "top_fraction": top_fraction,
            "oracle_fraction": oracle_fraction,
            "oracle_signal": oracle_signal,
            "importance_fraction": importance_fraction,
            "importance_policy": "lowest q0 positions; q-only and label-free after fraction is frozen",
            "lowq_multiscale_score": {
                "fractions": list(LOWQ_SCORE_FRACTIONS),
                "selection": "N_ref-standardized maximum; no member-labelled fraction selection",
                "calibration": "complete maximum score calibrated on N_cal",
            },
            "oracle_target_acceptance": target_acceptance,
            "oracle_status": "nondeployable exact-delta/exact-p upper bound",
            "point_estimate_boundary_policy": "explicit censor flag plus finite likelihood-scale representative",
            "role_policy": {
                "N_ref": "nonmember reference summaries only; no member labels",
                "N_cal": "nonmember operating-point calibration only",
                "T_member_T_nonmember": "exploratory membership evaluation only",
                "M_diag_and_reserved_C_member": "not replayed and never used",
            },
        },
        "role_counts": {
            "N_ref": len(roles.n_ref),
            "N_cal": len(roles.n_cal),
            "T_member": len(roles.t_member),
            "T_nonmember": len(roles.t_nonmember),
        },
        "methods": report_methods,
    }
    return report, flat_scores


def run_condition(
    benchmark: str,
    epoch: int,
    full_delta_path: Path,
    pq_path: Path,
    output_dir: Path,
    *,
    budgets: tuple[int, ...] = BUDGETS,
    split_seed: int = 20260824,
    replay_seed: int = 20260914,
    score_kind: str = "window_sign_8",
    top_fraction: float = 0.10,
    oracle_fraction: float = 0.10,
    oracle_signal: str = "window_boundary",
    importance_fraction: float = 0.20,
) -> dict[str, Any]:
    data = load_replay_data(full_delta_path, pq_path)
    roles = build_roles(data.labels, split_seed)
    report, scores = run_replay(
        data,
        roles,
        budgets=budgets,
        seed=replay_seed,
        score_kind=score_kind,
        top_fraction=top_fraction,
        oracle_fraction=oracle_fraction,
        oracle_signal=oracle_signal,
        importance_fraction=importance_fraction,
    )
    report["protocol"].update({
        "condition": f"{benchmark}_epoch{epoch}",
        "benchmark": benchmark,
        "epoch": epoch,
        "full_delta_input": str(full_delta_path.resolve()),
        "full_delta_sha256": _sha256(full_delta_path),
        "paired_pq_input": str(pq_path.resolve()),
        "paired_pq_sha256": _sha256(pq_path),
        "split_seed": split_seed,
        "records": len(data.labels),
        "tokens_after_eos_drop": len(data.logp),
        "length_range_after_eos_drop": [int(np.min(data.lengths)), int(np.max(data.lengths))],
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "E3_OFFLINE_REPORT.json", report)
    np.savez_compressed(
        output_dir / "scores.npz",
        labels=data.labels,
        record_ids=data.record_ids,
        lengths=data.lengths,
        **scores,
    )
    return report


def _default_full_delta(benchmark: str, epoch: int) -> Path:
    return ROOT / "experiments/results/sft_runs/full_delta" / f"{benchmark}_epoch{epoch}" / "draft_auxiliary_distilled/full_delta.npz"


def _default_pq(benchmark: str, epoch: int) -> Path:
    return ROOT / "experiments/results/sft_runs/pq_directional" / f"{benchmark}_epoch{epoch}" / "pq_gap_token_logps.npz"


def _default_output(benchmark: str, epoch: int, replay_seed: int) -> Path:
    return (
        ROOT
        / "experiments/results/sft_runs/accept_only_active_v2/e3_offline"
        / f"seed_{replay_seed}"
        / f"{benchmark}_epoch{epoch}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=BENCHMARKS, required=True)
    parser.add_argument("--epoch", choices=EPOCHS, type=int, required=True)
    parser.add_argument("--full-delta", type=Path)
    parser.add_argument("--pq-input", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--budgets", nargs="+", type=int, default=list(BUDGETS))
    parser.add_argument("--split-seed", type=int, default=20260824)
    parser.add_argument("--replay-seed", type=int, default=20260914)
    parser.add_argument(
        "--score-kind",
        choices=("mean_abs", "mean_positive", "mean_signed", "positive_fraction", "window_sign_8", "top_positive", "top_absolute"),
        default="window_sign_8",
    )
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--oracle-fraction", type=float, default=0.10)
    parser.add_argument("--importance-fraction", type=float, default=0.20)
    parser.add_argument(
        "--oracle-signal",
        choices=("positive", "negative", "absolute", "window_boundary"),
        default="window_boundary",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    full_delta = (args.full_delta or _default_full_delta(args.benchmark, args.epoch)).resolve()
    pq_path = (args.pq_input or _default_pq(args.benchmark, args.epoch)).resolve()
    output_dir = (
        args.output_dir
        or _default_output(args.benchmark, args.epoch, args.replay_seed)
    ).resolve()
    report = run_condition(
        args.benchmark,
        args.epoch,
        full_delta,
        pq_path,
        output_dir,
        budgets=tuple(args.budgets),
        split_seed=args.split_seed,
        replay_seed=args.replay_seed,
        score_kind=args.score_kind,
        top_fraction=args.top_fraction,
        oracle_fraction=args.oracle_fraction,
        oracle_signal=args.oracle_signal,
        importance_fraction=args.importance_fraction,
    )
    print(json.dumps({
        "condition": report["protocol"]["condition"],
        "output": str(output_dir),
        "budgets": report["protocol"]["budgets"],
    }, indent=2))


if __name__ == "__main__":
    main()
