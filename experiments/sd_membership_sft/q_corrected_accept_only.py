"""Q-corrected inference for position-locked accept-only observations.

For one fixed ``(prefix, candidate token)`` the target probability ``p`` is
unchanged across probes, while probe ``j`` exposes a known proposal
probability ``q_j``.  The verifier bit has probability

    alpha_j(p) = min(1, p / q_j).

This module keeps every binomial rejection term in the joint likelihood.  It
also reports likelihood plateaus explicitly: all-accept observations identify
a lower bound, not an exact probability.  A finite Bayesian point summary is
provided for downstream scoring, but is deliberately kept separate from the
MLE set and its censoring status.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import betaln, gammaln, logsumexp
from scipy.stats import chi2


@dataclass(frozen=True)
class QBinomialObservation:
    """Accept/reject counts collected under one known proposal probability."""

    q: float
    accepts: int
    trials: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.q) or not 0.0 < self.q <= 1.0:
            raise ValueError("q must be finite and in (0, 1]")
        if isinstance(self.accepts, bool) or not isinstance(
            self.accepts, (int, np.integer)
        ):
            raise TypeError("accepts must be an integer")
        if isinstance(self.trials, bool) or not isinstance(
            self.trials, (int, np.integer)
        ):
            raise TypeError("trials must be an integer")
        if self.trials <= 0:
            raise ValueError("trials must be positive")
        if not 0 <= self.accepts <= self.trials:
            raise ValueError("accepts must be between zero and trials")


@dataclass(frozen=True)
class ExactAlphaRecovery:
    """Identification interval produced by a noise-free acceptance rate."""

    p_lower: float
    p_upper: float
    delta0_lower: float
    delta0_upper: float
    point_identified: bool


@dataclass(frozen=True)
class JointEstimate:
    """Likelihood and finite posterior summaries for one candidate token.

    ``mle_p_lower`` and ``mle_p_upper`` describe the entire MLE set.  They are
    unequal for an all-accept likelihood plateau.  ``censoring`` prevents the
    finite posterior point from being mistaken for an exactly recovered p.
    """

    q0: float
    mle_p_lower: float
    mle_p_upper: float
    profile_p_lower: float
    profile_p_upper: float
    profile_delta0_lower: float
    profile_delta0_upper: float
    posterior_p_median: float
    posterior_delta0_median: float
    posterior_p_lower: float
    posterior_p_upper: float
    max_log_likelihood: float
    confidence: float
    censoring: Literal["none", "lower", "upper"]
    total_accepts: int
    total_trials: int


def _validate_probability(value: float, name: str, *, allow_zero: bool) -> float:
    value = float(value)
    lower_ok = value >= 0.0 if allow_zero else value > 0.0
    if not math.isfinite(value) or not lower_ok or value > 1.0:
        bracket = "[0, 1]" if allow_zero else "(0, 1]"
        raise ValueError(f"{name} must be finite and in {bracket}")
    return value


def acceptance_probability(p: float | np.ndarray, q: float) -> float | np.ndarray:
    """Return ``min(1, p/q)`` after validating its probability domain."""

    q = _validate_probability(q, "q", allow_zero=False)
    values = np.asarray(p, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("p must be finite and in [0, 1]")
    result = np.minimum(1.0, values / q)
    if values.ndim == 0:
        return float(result)
    return result


def recover_from_exact_alpha(q0: float, q: float, alpha: float) -> ExactAlphaRecovery:
    """Map an exact acceptance rate back to the common ``delta0`` coordinate.

    An ``alpha`` of one is saturated and therefore maps to ``p in [q, 1]``.
    Any smaller exact alpha point-identifies ``p=q*alpha``.  A zero p has
    ``delta0=-inf`` mathematically; finite-sample scoring should instead use
    :func:`estimate_joint`'s finite posterior summary.
    """

    q0 = _validate_probability(q0, "q0", allow_zero=False)
    q = _validate_probability(q, "q", allow_zero=False)
    alpha = _validate_probability(alpha, "alpha", allow_zero=True)
    if alpha == 1.0:
        p_lower, p_upper = q, 1.0
        point_identified = q == 1.0
    else:
        p_lower = p_upper = q * alpha
        point_identified = True
    delta_lower = -math.inf if p_lower == 0.0 else math.log(p_lower / q0)
    delta_upper = math.log(p_upper / q0)
    return ExactAlphaRecovery(
        p_lower=p_lower,
        p_upper=p_upper,
        delta0_lower=delta_lower,
        delta0_upper=delta_upper,
        point_identified=point_identified,
    )


def corrected_delta0(q0: float, q: float, alpha: float) -> float:
    """Return ``log(q*alpha)-log(q0)`` for an unsaturated exact observation.

    Saturated observations are deliberately rejected because they identify an
    interval rather than a point.  Use :func:`recover_from_exact_alpha` when
    interval output is desired.
    """

    recovery = recover_from_exact_alpha(q0, q, alpha)
    if not recovery.point_identified or recovery.p_lower != recovery.p_upper:
        raise ValueError("alpha=1 is saturated and does not point-identify delta0")
    return recovery.delta0_lower


def _as_observations(
    observations: Iterable[QBinomialObservation],
) -> tuple[QBinomialObservation, ...]:
    output = tuple(observations)
    if not output:
        raise ValueError("at least one observation is required")
    if not all(isinstance(item, QBinomialObservation) for item in output):
        raise TypeError("observations must contain QBinomialObservation values")
    return output


def joint_binomial_log_likelihood(
    p: float | np.ndarray,
    observations: Iterable[QBinomialObservation],
    *,
    include_binomial_constants: bool = False,
) -> float | np.ndarray:
    """Evaluate the multi-q joint binomial log likelihood.

    Impossible events correctly have log likelihood ``-inf``.  In particular,
    any rejection under a saturated proposal rules out ``p >= q``.  No
    pseudo-counts are added.
    """

    obs = _as_observations(observations)
    values = np.asarray(p, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("p must be finite and in [0, 1]")
    flat = values.reshape(-1)
    total = np.zeros(flat.shape, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        for item in obs:
            alpha = np.minimum(1.0, flat / item.q)
            if item.accepts:
                total += item.accepts * np.log(alpha)
            rejects = item.trials - item.accepts
            if rejects:
                total += rejects * np.log1p(-alpha)
            if include_binomial_constants:
                total += (
                    gammaln(item.trials + 1)
                    - gammaln(item.accepts + 1)
                    - gammaln(rejects + 1)
                )
    result = total.reshape(values.shape)
    if values.ndim == 0:
        return float(result)
    return result


def _mle_set(
    observations: tuple[QBinomialObservation, ...],
) -> tuple[float, float, float]:
    total_accepts = sum(item.accepts for item in observations)
    total_trials = sum(item.trials for item in observations)
    if total_accepts == 0:
        return 0.0, 0.0, float(joint_binomial_log_likelihood(0.0, observations))
    if total_accepts == total_trials:
        lower = max(item.q for item in observations)
        return lower, 1.0, float(joint_binomial_log_likelihood(lower, observations))

    rejected_q = [item.q for item in observations if item.accepts < item.trials]
    upper = min(rejected_q) if rejected_q else 1.0
    normalized_upper = float(np.nextafter(1.0, 0.0)) if upper < 1.0 else 1.0

    def objective(normalized: float) -> float:
        result = joint_binomial_log_likelihood(normalized * upper, observations)
        return math.inf if not math.isfinite(result) else -float(result)

    result = minimize_scalar(
        objective,
        bounds=(float(np.nextafter(0.0, 1.0)), normalized_upper),
        method="bounded",
        options={"xatol": 1e-13, "maxiter": 1000},
    )
    candidates = [float(result.x * upper)]
    candidates.extend(
        item.q
        for item in observations
        if item.accepts == item.trials and item.q < upper
    )
    likelihoods = np.asarray(
        [joint_binomial_log_likelihood(value, observations) for value in candidates]
    )
    best = int(np.argmax(likelihoods))
    p_hat = float(candidates[best])
    return p_hat, p_hat, float(likelihoods[best])


def likelihood_ratio_interval(
    observations: Iterable[QBinomialObservation],
    *,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute a one-parameter profile-likelihood confidence interval for p."""

    obs = _as_observations(observations)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    mle_lower, mle_upper, maximum = _mle_set(obs)
    threshold = maximum - 0.5 * float(chi2.ppf(confidence, df=1))

    def root_function(value: float) -> float:
        return float(joint_binomial_log_likelihood(value, obs)) - threshold

    if mle_lower == 0.0 or root_function(0.0) >= 0.0:
        lower = 0.0
    else:
        normalized_root = brentq(
            lambda value: root_function(value * mle_lower),
            0.0,
            1.0,
            xtol=1e-14,
            rtol=1e-14,
        )
        lower = float(normalized_root * mle_lower)

    if mle_upper == 1.0 or root_function(1.0) >= 0.0:
        upper = 1.0
    else:
        rejected_q = [item.q for item in obs if item.accepts < item.trials]
        domain_upper = min(rejected_q) if rejected_q else 1.0
        search_upper = (
            float(np.nextafter(domain_upper, 0.0)) if domain_upper < 1.0 else 1.0
        )
        if root_function(search_upper) >= 0.0:
            upper = domain_upper
        else:
            width = search_upper - mle_upper
            normalized_root = brentq(
                lambda value: root_function(mle_upper + value * width),
                0.0,
                1.0,
                xtol=1e-14,
                rtol=1e-14,
            )
            upper = float(mle_upper + normalized_root * width)
    return lower, upper


def _posterior_summary(
    observations: tuple[QBinomialObservation, ...],
    confidence: float,
    prior: tuple[float, float],
    quadrature_order: int,
) -> tuple[float, float, float]:
    """Numerically integrate a Beta-prior posterior, split at every q knot."""

    prior_a, prior_b = map(float, prior)
    if not prior_a > 0.0 or not prior_b > 0.0:
        raise ValueError("Beta prior parameters must be positive")
    if quadrature_order < 16:
        raise ValueError("quadrature_order must be at least 16")
    base_nodes, base_weights = np.polynomial.legendre.leggauss(quadrature_order)
    knots = np.unique(np.r_[0.0, [item.q for item in observations], 1.0])
    all_nodes: list[np.ndarray] = []
    all_log_weights: list[np.ndarray] = []
    for left, right in zip(knots[:-1], knots[1:]):
        if right <= left:
            continue
        nodes = left + (base_nodes + 1.0) * (right - left) / 2.0
        weights = base_weights * (right - left) / 2.0
        log_prior = (
            (prior_a - 1.0) * np.log(nodes)
            + (prior_b - 1.0) * np.log1p(-nodes)
            - betaln(prior_a, prior_b)
        )
        log_likelihood = np.asarray(joint_binomial_log_likelihood(nodes, observations))
        all_nodes.append(nodes)
        all_log_weights.append(np.log(weights) + log_prior + log_likelihood)
    nodes = np.concatenate(all_nodes)
    log_weights = np.concatenate(all_log_weights)
    order = np.argsort(nodes)
    nodes = nodes[order]
    probabilities = np.exp(log_weights[order] - logsumexp(log_weights[order]))
    cumulative = np.cumsum(probabilities)

    def quantile(probability: float) -> float:
        index = min(int(np.searchsorted(cumulative, probability, side="left")), len(nodes) - 1)
        return float(nodes[index])

    tail = (1.0 - confidence) / 2.0
    return quantile(0.5), quantile(tail), quantile(1.0 - tail)


def estimate_joint(
    q0: float,
    observations: Iterable[QBinomialObservation],
    *,
    confidence: float = 0.95,
    prior: tuple[float, float] = (0.5, 0.5),
    quadrature_order: int = 96,
) -> JointEstimate:
    """Fit p jointly and express every delta relative to the baseline q0.

    The default Jeffreys prior is used only for the explicitly named finite
    posterior summary.  It never alters the joint likelihood, MLE plateau, or
    profile interval, so all-accept/all-reject censoring remains visible.
    """

    q0 = _validate_probability(q0, "q0", allow_zero=False)
    obs = _as_observations(observations)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    mle_lower, mle_upper, maximum = _mle_set(obs)
    profile_lower, profile_upper = likelihood_ratio_interval(obs, confidence=confidence)
    posterior_median, posterior_lower, posterior_upper = _posterior_summary(
        obs, confidence, prior, quadrature_order
    )
    total_accepts = sum(item.accepts for item in obs)
    total_trials = sum(item.trials for item in obs)
    if total_accepts == total_trials:
        censoring: Literal["none", "lower", "upper"] = "lower"
    elif total_accepts == 0:
        censoring = "upper"
    else:
        censoring = "none"
    return JointEstimate(
        q0=q0,
        mle_p_lower=mle_lower,
        mle_p_upper=mle_upper,
        profile_p_lower=profile_lower,
        profile_p_upper=profile_upper,
        profile_delta0_lower=(
            -math.inf if profile_lower == 0.0 else math.log(profile_lower / q0)
        ),
        profile_delta0_upper=math.log(profile_upper / q0),
        posterior_p_median=posterior_median,
        posterior_delta0_median=math.log(posterior_median / q0),
        posterior_p_lower=posterior_lower,
        posterior_p_upper=posterior_upper,
        max_log_likelihood=maximum,
        confidence=confidence,
        censoring=censoring,
        total_accepts=total_accepts,
        total_trials=total_trials,
    )


def synthetic_invariance_report() -> dict[str, object]:
    """Return the preregistered noise-free E0 q-coordinate checks."""

    cases: list[dict[str, object]] = []
    for q0, p, active_qs in (
        (0.6, 0.8, (0.85, 0.9, 0.95)),
        (0.6, 0.3, (0.6, 0.75, 0.9)),
    ):
        expected = math.log(p / q0)
        recovered = [corrected_delta0(q0, q, p / q) for q in active_qs]
        cases.append(
            {
                "q0": q0,
                "p": p,
                "active_q": list(active_qs),
                "expected_delta0": expected,
                "recovered_delta0": recovered,
                "max_abs_error": max(abs(value - expected) for value in recovered),
            }
        )
    return {
        "experiment": "E0 synthetic q-correction invariance",
        "status": "pass"
        if all(float(case["max_abs_error"]) < 1e-12 for case in cases)
        else "fail",
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = synthetic_invariance_report()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
