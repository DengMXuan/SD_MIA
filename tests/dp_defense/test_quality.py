"""The DP quality entry stays bound to its frozen epsilon-8 WikiTection runs."""
import json
from types import SimpleNamespace

import pytest

from experiments.dp_defense import quality as dp_quality
from experiments.dp_defense.quality_cli import make_tasks
from experiments.shared.audit.artifacts import digest
from experiments.shared.evaluation import quality


def test_matrix_is_three_seeds_and_two_independent_evaluations(tmp_path):
    args = SimpleNamespace(seeds=list(dp_quality.SEEDS), evaluations=list(quality.KINDS),
                           bootstrap_repeats=1000, samples=500, per_class=256, batch_size=4,
                           output_root=tmp_path / 'dp_quality')
    tasks = make_tasks(args)
    assert len(tasks) == len({task['output'] for task in tasks}) == 6
    assert {task['condition']['condition_seed'] for task in tasks} == set(dp_quality.SEEDS)
    assert {task['evaluation'] for task in tasks} == set(quality.KINDS)
    assert all(task['evaluation_track'] == dp_quality.DP_TRACK for task in tasks)
    assert all(task['draft_role'] == 'draft_auxiliary_distilled'
               for task in tasks if task['evaluation'] == 'acceptance')
    assert not args.output_root.exists()


def test_dp_passport_requires_exact_budget_kd_and_stage_markers(tmp_path, monkeypatch):
    monkeypatch.setattr(dp_quality, 'RUN_ROOT', tmp_path)
    run = tmp_path / 'seed1919'
    request = dict(model_pair='qwen3', draft_variants=['kd'], config={'seed': 1919},
                   plans={'target': {'epsilon': 8.0, 'delta': 5e-6}})
    artifact = dict(config=request['config'], privacy=dict(
        request_key=digest(request), draft_variants=['kd'], draft_roles=['draft_auxiliary_distilled'],
        pairs={'draft_auxiliary_distilled': dict(epsilon=7.9994, epsilon_cap=8.0,
                                                 delta=5e-6, composition='target_postprocessing')},
        stages={}))
    for role in ('target', 'draft_auxiliary_distilled'):
        teacher = 'target-sha' if role != 'target' else None
        marker = dict(role=role, key=dp_quality.stage_key(request, role, teacher))
        stage = dict(marker, checkpoint_sha256='target-sha' if role == 'target' else 'draft-sha')
        artifact['privacy']['stages'][role] = stage
        folder = run / 'checkpoints' / role
        folder.mkdir(parents=True)
        (folder / 'DP_STAGE.json').write_text(json.dumps(marker))
    (run / 'DP_REQUEST.json').write_text(json.dumps(request))
    task = dp_quality.make_task(1919, 'acceptance', output_root=tmp_path / 'reports')
    dp_quality.validate_dp_task(task, artifact)
    wrong = dict(task, run_dir=str(tmp_path / 'other'))
    with pytest.raises(ValueError, match='outside'):
        dp_quality.validate_dp_task(wrong, artifact)
    artifact['privacy']['pairs']['draft_auxiliary_distilled']['epsilon'] = 8.01
    with pytest.raises(ValueError, match='budget'):
        dp_quality.validate_dp_task(task, artifact)
    artifact['privacy']['pairs']['draft_auxiliary_distilled']['epsilon'] = 7.9994
    marker = run / 'checkpoints' / 'draft_auxiliary_distilled' / 'DP_STAGE.json'
    marker.write_text(json.dumps({'role': 'draft_auxiliary_distilled', 'key': 'wrong'}))
    with pytest.raises(ValueError, match='stage marker'):
        dp_quality.validate_dp_task(task, artifact)


def test_private_worker_verifies_training_before_gpu_use(tmp_path, monkeypatch):
    task = dp_quality.make_task(1919, 'acceptance', output_root=tmp_path)
    calls = []
    monkeypatch.setattr(dp_quality, 'validate_task', lambda _: calls.append('preflight'))
    monkeypatch.setattr(dp_quality, 'verify_run', lambda _: calls.append('verify'))
    monkeypatch.setattr(dp_quality.torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='requires CUDA'):
        dp_quality.evaluate_quality(task)
    assert calls == ['preflight', 'verify']


def test_dp_runtime_closure_contains_dp_entry_and_verifier():
    paths = set(dp_quality.runtime_files())
    for name in ('quality.py', 'quality_cli.py', 'artifacts.py'):
        assert any(path.name == name and 'dp_defense' in path.parts for path in paths)
