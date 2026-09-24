"""One fixed-probe and baseline execution contract for all audit entry points."""
BASELINE_DEFAULTS = dict(k_percent=20., recall_shots=4, icp_top_k=5, icp_aggregation="min",
                         sead_samples=50, sead_temperature=1., samia_samples=10,
                         prefix_ratio=.5, perturbation_rate=.15, generation_batch_size=8)


def audit_settings(*, audit_seed=20260914, detector_epochs=30,
                   starts=None, rounds_per_start=32):
    if type(audit_seed) is not int or audit_seed < 0 or detector_epochs < 1 or rounds_per_start < 1:
        raise ValueError('nonnegative audit seed and positive detector/round budgets required')
    return dict(starts=list(starts) if starts is not None else ['suffix64'],
                rounds_per_start=rounds_per_start, audit_seed=audit_seed,
                detector_epochs=detector_epochs, baseline=dict(BASELINE_DEFAULTS),
                baseline_execution='shared_robustness_reference_v1')
