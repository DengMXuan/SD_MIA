"""Exercise the shell entry's scheduling-only override without launching models."""
from pathlib import Path
from unittest.mock import Mock

import pytest

from experiments.sd_membership_sft.audit import qwen_audit_matrix as matrix


def launcher_namespace():
    wrapper = Path(matrix.__file__).with_name("cli.py")
    source = wrapper.read_text()
    namespace = {"__name__": "launcher_test"}
    exec(compile(source, str(wrapper), "exec"), namespace)
    return namespace


def launch_check(monkeypatch, gpus, inventory):
    query = Mock(return_value=inventory)
    monkeypatch.setattr(matrix.subprocess, "check_output", query)
    launcher_namespace()["check_gpus"](gpus)
    return query


@pytest.mark.parametrize("used", [0, 400, 503, 1024])
def test_launcher_allows_small_background_allocations(monkeypatch, used):
    monkeypatch.delenv("SD_AUDIT_GPU_MAX_USED_MIB", raising=False)
    query = launch_check(monkeypatch, [0, 2], f"0, {used}\n2, 8\n")
    assert "--query-gpu=index,memory.used" in query.call_args.args[0]


@pytest.mark.parametrize("gpus,inventory,error", [
    ([0], "0, 1025\n", RuntimeError),
    ([0], "0, 40000\n", RuntimeError),
    ([2], "0, 400\n", RuntimeError),
    ([0, 0], "0, 400\n", ValueError),
    ([-1], "0, 400\n", ValueError),
    ([], "0, 400\n", ValueError),
    ([0], "0, [N/A]\n", ValueError),
])
def test_launcher_still_rejects_busy_missing_or_invalid_gpus(monkeypatch, gpus, inventory, error):
    monkeypatch.delenv("SD_AUDIT_GPU_MAX_USED_MIB", raising=False)
    with pytest.raises(error):
        launch_check(monkeypatch, gpus, inventory)


def test_launcher_threshold_is_configurable(monkeypatch):
    monkeypatch.setenv("SD_AUDIT_GPU_MAX_USED_MIB", "2048")
    launch_check(monkeypatch, [0], "0, 1500\n")
    monkeypatch.setenv("SD_AUDIT_GPU_MAX_USED_MIB", "100")
    with pytest.raises(RuntimeError, match="100 MiB"):
        launch_check(monkeypatch, [0], "0, 400\n")


@pytest.mark.parametrize("value", ["-1", "invalid", "nan"])
def test_launcher_invalid_threshold_fails(monkeypatch, value):
    monkeypatch.setenv("SD_AUDIT_GPU_MAX_USED_MIB", value)
    with pytest.raises(ValueError):
        launch_check(monkeypatch, [0], "0, 400\n")


def test_fixed_scope_keeps_original_task_signatures(tmp_path):
    from tests.sft.test_qwen_audit_matrix import settings
    from experiments.shared.audit.artifacts import digest

    args = (tmp_path / "models", tmp_path / "results", ["wikitection", "newstection", "arxivtection"],
            [1, 3], [1919, 1949, 1978], settings())
    original = matrix.make_tasks(*args)
    filtered = launcher_namespace()["fixed_tasks"](*args)
    assert len(filtered) == 54
    assert sum(len(t["methods"]) for t in filtered) == 234
    assert all(t.get("protocol") != "natural" for t in filtered)
    assert filtered == [t for t in original if t.get("protocol") != "natural"]
    for task in filtered:
        previous = next(t for t in original if t["id"] == task["id"])
        assert digest(task) == digest(previous)


def test_fixed_summary_preserves_existing_outputs_and_excludes_natural_cost(tmp_path):
    import json
    from tests.sft.test_qwen_audit_matrix import tasks_at, write_example_result

    tasks = tasks_at(tmp_path, seeds=(1919,))
    for task in tasks:
        for method in task["methods"]:
            write_example_result(task, method, .8)
    root = tmp_path / "results"
    (root / "SUMMARY.json").write_text('{"historical":true}')
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    for i, task in enumerate((tasks[0], next(t for t in tasks if t.get('protocol') == 'natural'))):
        folder = root / "executions" / str(i)
        folder.mkdir(parents=True)
        (folder / "STATUS.json").write_text(json.dumps(dict(task=task['id'], worker_wall_seconds=10)))
    original_methods = matrix.ALL_METHODS
    result = launcher_namespace()["summarize_fixed"]([t for t in tasks if t.get('protocol') != 'natural'], root)
    assert result['complete'] and result['expected_rows'] == result['completed_rows'] == 24
    assert result['attempted_worker_wall_seconds_sum'] == 10
    assert result['unique_successful_execution_groups'] == 13
    assert all('natural' not in row['method'] for row in result['rows'])
    assert matrix.ALL_METHODS == original_methods
    assert (root / 'reports' / 'RESULTS.csv').exists()
    assert all(p.read_bytes() == content for p, content in before.items())


def test_fixed_launcher_dispatch_and_status_counts(tmp_path, monkeypatch, capsys):
    import json
    import sys

    ns = launcher_namespace()
    monkeypatch.setattr(matrix, 'inspect_task', lambda task: {'state': 'ready', 'completed_methods': []})
    monkeypatch.setattr(sys, 'argv', ['launcher', 'dry-run', '--output-root', str(tmp_path)])
    ns['main']()
    status = json.loads(capsys.readouterr().out)
    assert (status['audit_configurations'], status['worker_tasks'], status['expected_method_rows']) == (36, 54, 432)
    run = Mock(return_value=False)
    summary = Mock(return_value=dict(complete=False, expected_rows=432, completed_rows=0, errors=[]))
    monkeypatch.setattr(matrix, 'run_tasks', run)
    monkeypatch.setattr(matrix, 'check_gpus', matrix.check_gpus)
    ns['summarize_fixed'] = summary
    monkeypatch.setattr(sys, 'argv', ['launcher', 'run', '--output-root', str(tmp_path), '--gpus', '0', '1'])
    with pytest.raises(SystemExit) as error:
        ns['main']()
    assert error.value.code == 2
    tasks, root, gpus = run.call_args.args
    assert len(tasks) == 54 and gpus == [0, 1]
    assert not any(t.get('protocol') == 'natural' for t in tasks)
    assert matrix.check_gpus is ns['check_gpus']
