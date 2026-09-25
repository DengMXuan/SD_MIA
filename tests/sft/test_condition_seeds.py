import numpy as np
import pytest

from experiments.shared.audit.config import audit_settings, condition_settings
from experiments.shared.core.audit_partitions import deployment_partitions
from experiments.shared.models.registry import MODEL_PAIRS
from experiments.cross_model_audit.engine import make_tasks
from tests.sft.test_qwen_audit_matrix import prepared_records


def test_all_pairs_bind_main_and_baselines_without_mutating_other_conditions(tmp_path):
    unresolved = audit_settings()
    for pair in MODEL_PAIRS:
        tasks = make_tasks(tmp_path, tmp_path / 'out', ['wikitection'], [1], [1919, 1949, 1978], unresolved, pair)
        for task in tasks:
            assert task['settings']['audit_seed'] == task['condition']['condition_seed']
            assert task['settings']['seed_policy'] == 'condition_v1'
    assert unresolved['audit_seed'] is None
    with pytest.raises(ValueError, match='must equal'):
        condition_settings(audit_settings(audit_seed=20260914), 1919)


def test_condition_seed_changes_auxiliary_roles_but_not_the_test_set():
    records = prepared_records()
    parts = {seed: deployment_partitions(records.labels, records.record_ids, records.record_roles, seed=seed)
             for seed in (1919, 1949, 1978)}
    for seed, p in parts.items():
        expected = np.random.default_rng(seed).permutation(600)
        assert np.array_equal(p['train'], np.sort(expected[:320]))
        assert np.array_equal(p['validation'], np.sort(expected[320:400]))
        assert np.array_equal(p['calibration'], np.sort(expected[400:]))
        assert np.array_equal(p['test'], np.arange(600, 4600))
    assert not np.array_equal(parts[1919]['train'], parts[1949]['train'])
