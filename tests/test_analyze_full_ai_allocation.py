import numpy as np

from experiments.sd_membership_sft.analyze_full_ai_allocation import _summary


def test_summary_respects_lower_is_better_metrics() -> None:
    bootstrap = np.asarray([-0.4, -0.2, 0.1], dtype=np.float64)
    conditions = np.asarray([-0.3, 0.2, -0.1], dtype=np.float64)
    result = _summary(
        float(np.mean(conditions)),
        bootstrap,
        conditions,
        improvement_sign=-1.0,
    )
    assert result["condition_seed_wins"] == 2
    assert result["bootstrap_probability_improvement"] == 2 / 3
    assert result["point"] < 0.0
