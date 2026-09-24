import numpy as np

from experiments.sd_membership_sft.analysis.directional_mia import conformal_tail_pvalues, order_statistic_threshold, threshold_metrics


def test_order_statistic_calibration_does_not_linearly_interpolate() -> None:
    calibration = np.arange(10, dtype=np.float64)
    assert order_statistic_threshold(calibration, 0.10) == 8.0
    result = threshold_metrics(
        np.array([8.0, 9.5]),
        np.array([8.0, 9.5]),
        0.10,
        calibration_nonmember=calibration,
    )
    # The tied score 8.0 has p=(1+2)/11 and is not a hit; 9.5 has p=1/11.
    assert result["member_hits"] == 1
    assert result["nonmember_hits"] == 1


def test_conformal_tail_pvalues_count_calibration_ties_inclusively() -> None:
    calibration = np.array([0.0, 1.0, 1.0, 2.0])
    pvalues = conformal_tail_pvalues(np.array([2.0, 1.0, 0.5]), calibration)
    assert np.allclose(pvalues, [2 / 5, 4 / 5, 4 / 5])
