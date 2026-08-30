import numpy as np
import pytest

from experiments.sd_membership_sft.public_budget_sweep import _nested_acceptance


def test_nested_acceptance_rejects_nonfinite_selected_logp() -> None:
    target_logp = np.asarray([[np.nan, -1.0]], dtype=np.float32)
    draft_logp = np.asarray([[-2.0, -1.5]], dtype=np.float32)
    selected = np.asarray([[0]], dtype=np.int64)

    with pytest.raises(ValueError, match="non-finite selected token logp"):
        _nested_acceptance(target_logp, draft_logp, selected, [4], seed=7)
