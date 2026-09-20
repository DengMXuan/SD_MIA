"""Poisson-sampled Gaussian mechanism, using the maintained Opacus RDP accountant."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.metadata import version
import math


@dataclass(frozen=True)
class PrivacyPlan:
    epsilon: float
    delta: float
    max_grad_norm: float
    population: int
    expected_batch_size: int
    steps: int
    sample_rate: float
    noise_multiplier: float
    accounted_epsilon: float
    accountant_version: str

    def as_dict(self):
        return {**asdict(self), "accountant": "opacus_rdp", "adjacency": "add_remove_document",
                "sampling": "independent_poisson", "normalization": "fixed_expected_batch_size"}


def epsilon_for(noise_multiplier: float, sample_rate: float, steps: int, delta: float) -> float:
    if (not math.isfinite(noise_multiplier) or noise_multiplier <= 0
            or not 0 < sample_rate <= 1 or not 0 < delta < 1
            or not isinstance(steps, int) or steps < 0):
        raise ValueError("invalid Gaussian mechanism accounting parameters")
    if not steps:
        return 0.0
    from opacus.accountants import RDPAccountant
    accountant = RDPAccountant()
    accountant.history = [(noise_multiplier, sample_rate, steps)]
    return float(accountant.get_epsilon(delta=delta))


def make_plan(*, epsilon: float, delta: float = 5e-6, max_grad_norm: float = 1.,
              population: int = 2000, expected_batch_size: int = 16, epochs: int = 1) -> PrivacyPlan:
    for value in (population, expected_batch_size, epochs):
        if type(value) is not int or value <= 0:
            raise ValueError("population, expected batch size and epochs must be positive integers")
    if (not math.isfinite(epsilon) or epsilon <= 0 or not 0 < delta < 1
            or not math.isfinite(max_grad_norm) or max_grad_norm <= 0
            or expected_batch_size > population):
        raise ValueError("invalid privacy budget, clipping bound or expected batch size")
    from opacus.accountants.utils import get_noise_multiplier
    q = expected_batch_size / population
    # Public reference population fixes q, step count and normalization even
    # across add/remove neighbors. Epochs denote expected passes, not shuffling.
    steps = math.ceil(population / expected_batch_size) * epochs
    sigma = float(get_noise_multiplier(
        target_epsilon=epsilon, target_delta=delta, sample_rate=q, steps=steps,
        accountant="rdp", epsilon_tolerance=min(.001, epsilon * .001),
    ))
    actual = epsilon_for(sigma, q, steps, delta)
    if actual > epsilon:
        raise ValueError("noise calibration exceeded the requested epsilon")
    return PrivacyPlan(epsilon, delta, max_grad_norm, population, expected_batch_size,
                       steps, q, sigma, actual, version("opacus"))


def pair_budgets(target: dict, member: dict) -> dict:
    """Basic composition; distillation does not independently spend member budget."""
    for stage in (target, member):
        if stage["accounted_epsilon"] > stage["epsilon"] + 1e-10:
            raise ValueError("a stage exceeded its privacy cap")
    if target["epsilon"] != member["epsilon"] or target["delta"] != member["delta"]:
        raise ValueError("this experiment requires matching target/member budgets")
    def bound(epsilon, delta, cap, mechanism):
        return dict(epsilon=epsilon, delta=delta, epsilon_cap=cap, composition=mechanism)
    return {
        "draft_auxiliary_distilled": bound(target["accounted_epsilon"], target["delta"],
                                            target["epsilon"], "target_postprocessing"),
        "draft_member_sft": bound(target["accounted_epsilon"] + member["accounted_epsilon"],
                                   target["delta"] + member["delta"],
                                   target["epsilon"] + member["epsilon"], "basic_sequential"),
    }
