from functools import lru_cache
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.shared.audit.metrics import metrics, roc_operating_point
from experiments.shared.audit.artifacts import digest, read_result, save_result
from experiments.shared.audit.baselines import BASELINE_DEFAULTS
from experiments.sd_membership_sft.audit.qwen_audit_matrix import make_tasks, summarize, ready, check_gpus


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def settings():
    return dict(starts=["suffix64"], rounds_per_start=32, audit_seed=20260914,
                detector_epochs=1, baseline=dict(BASELINE_DEFAULTS))


def tasks_at(tmp_path, seeds=(1919, 1949, 1978)):
    return make_tasks(tmp_path / "models", tmp_path / "results", ["wikitection"], [1], list(seeds), settings())


def test_matrix_has_36_configs_54_jobs_and_432_method_rows(tmp_path):
    tasks = make_tasks(tmp_path, tmp_path / "out", ["wikitection", "newstection", "arxivtection"],
                       [1, 3], [1919, 1949, 1978], settings())
    assert len(tasks) == len({t["id"] for t in tasks}) == 54
    assert sum(t["kind"] == "baseline" for t in tasks) == 18
    assert sum(t.get("protocol") == "fixed" for t in tasks) == 36
    assert sum(len(t["methods"]) for t in tasks) == 234  # unique method outputs
    assert 18 * 2 * (11 + 1) == 432
    assert len({t["output"] for t in tasks}) == 54


def test_roc_does_not_split_ties_and_pauc_conventions_are_explicit():
    labels = np.array([0, 0, 1, 1])
    assert roc_operating_point(np.ones(4), labels, .1) == (0., 0.)
    # Calibration records are separate from the four test records.
    all_labels = np.r_[0, labels]
    perfect = metrics(np.array([.5, 0, 0, 1, 1]), all_labels, [0], [1, 2, 3, 4], bootstrap=0)
    assert perfect["auc"] == 1.
    assert perfect["pauc_10_raw"] == pytest.approx(.1)
    assert perfect["pauc_10_normalized"] == pytest.approx(1.)
    assert perfect["roc_tpr_at_1pct_fpr"] == 1.
    tied = metrics(np.ones(5), all_labels, [0], [1, 2, 3, 4], bootstrap=0)
    assert tied["auc"] == .5
    assert tied["pauc_10_raw"] == pytest.approx(.005)
    assert tied["pauc_10_normalized"] == pytest.approx(.05)


def test_deployment_threshold_does_not_use_test_nonmember_scores():
    labels = np.r_[np.zeros(200), [1, 1, 0, 0]].astype(int)
    scores = np.r_[np.linspace(0, 1, 200), [2., 3., .5, .9]]
    first = metrics(scores, labels, np.arange(200), np.arange(200, 204), bootstrap=0)
    scores[-2:] = 100
    second = metrics(scores, labels, np.arange(200), np.arange(200, 204), bootstrap=0)
    assert first["calibrated_tpr_at_1pct"] == second["calibrated_tpr_at_1pct"] == 1.
    assert first["roc_tpr_at_1pct_fpr"] == 1.
    assert second["roc_tpr_at_1pct_fpr"] == 0.
    assert second["calibrated_actual_fpr_at_1pct"] == 1.
    with pytest.raises(ValueError):
        metrics(scores, labels, [200], np.arange(201, 204), bootstrap=0)


def write_example_result(task, method, value):
    folder = Path(task["output"]) / method
    save_result(folder, record_ids=np.array(["cal", "member", "nonmember"]), labels=np.array([0, 1, 0]),
                scores=np.array([.5, .8, .2]), calibration=np.array([0]), test=np.array([1, 2]),
                report=dict(method=method, request_key=digest({"task": task, "method": method}),
                            sources={"files": [], "checkpoints": []}, metrics={"auc": value},
                            cost={"total_seconds": 10., "execution_group": task["id"] + "/" + method},
                            access_channel="target_probabilities"))
    return folder


def test_result_recovery_checks_scores_and_parameters(tmp_path):
    task = tasks_at(tmp_path)[0]
    folder = write_example_result(task, "loss", .8)
    assert read_result(folder)["metrics"]["auc"] == .8
    with pytest.raises(ValueError, match="parameters"):
        read_result(folder, "wrong-key")
    with (folder / "scores.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        read_result(folder)


def test_baseline_reuse_does_not_double_costs_or_seed_sample_size(tmp_path):
    tasks = tasks_at(tmp_path)
    for index, task in enumerate(t for t in tasks if t["kind"] == "baseline"):
        write_example_result(task, "loss", .7 + .1 * index)
    result = summarize(tasks, tmp_path / "results")
    assert result["expected_rows"] == 72
    assert result["completed_rows"] == 6
    assert not result["complete"]
    assert result["unique_successful_execution_groups"] == 3
    assert result["unique_successful_measured_method_seconds"] == 30
    for row in result["seed_summary"]:
        if row["method"] == "loss":
            assert row["completed_seeds"] == 3
            assert row["auc_mean"] == pytest.approx(.8)
            assert row["auc_std"] == pytest.approx(.1)


def prepared_records():
    from experiments.shared.data.data import SFTRecord

    roles = np.array(["audit_auxiliary"] * 600 + ["member"] * 2000 + ["nonmember"] * 2000)
    rows = [SFTRecord(str(i), "test", (0, 1, 2), str(i), prompt_ids=(1,)) for i in range(4600)]
    return SimpleNamespace(records=rows, labels=(roles == "member").astype(int), record_roles=roles,
                           record_ids=np.array([r.record_id for r in rows]), tokenizer=SimpleNamespace(eos_token_id=2))


class FakeScorer:
    def __init__(self, *args):
        self.cost_meter = None

    def stats(self, record):
        pass

    @lru_cache(None)
    def _probe_vector(self, record):
        pass


def test_baseline_adapter_uses_only_400_fit_records_and_200_independent_calibration(tmp_path, monkeypatch):
    import experiments.shared.audit.baselines as runner
    from experiments.shared.core.audit_partitions import deployment_partitions

    task = tasks_at(tmp_path)[0]
    task["methods"] = ["petal", "loss"]
    prepared = prepared_records()
    parts = deployment_partitions(prepared.labels, prepared.record_ids, prepared.record_roles)
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(runner, "load_finetuned_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(runner, "TargetScorer", FakeScorer)
    called = []
    def score(args, progress, scorer, records, auxiliary, tokenizer, methods, *, reference_cache):
        method = methods[0]
        called.append(method)
        assert len(records) == 4200
        assert [r.record.record_id for r in records[:200]] == prepared.record_ids[parts["calibration"]].tolist()
        assert {r.record_id for r in auxiliary} == (set(prepared.record_ids[parts["reference"]]) if method == "petal" else set())
        assert not {r.record_id for r in auxiliary}.intersection(r.record.record_id for r in records)
        assert all(not r.record.append_eos for r in records)
        for record in progress.track(auxiliary, "petal calibration"):
            scorer.cost_meter.forward(4)
        values = []
        for row in progress.track(records, "teacher-forced scoring"):
            scorer.cost_meter.forward(4)
            values.append(float(int(row.record.record_id) % 7))
        return {method: values}
    monkeypatch.setattr(runner, "score_methods", score)
    source = {"files": [], "checkpoints": []}
    runner.run_baselines(task, "cpu", SimpleNamespace(target_model="toy"), prepared, source)
    petal = read_result(Path(task["output"]) / "petal")
    assert petal["phase_work"]["preparation"]["forward_sequences"] == 400
    assert petal["phase_work"]["calibration"]["forward_sequences"] == 200
    assert petal["phase_work"]["test"]["forward_sequences"] == 4000
    assert petal["reference_records_used"] == 400
    assert petal["cost"]["target_sequences"] == 4600
    runner.run_baselines(task, "cpu", SimpleNamespace(target_model="toy"), prepared, source)
    assert called == ["petal", "loss"]  # checked completed methods are not re-scored


def test_optimized_baseline_adapter_reuses_reference_with_explicit_cost_mode(tmp_path, monkeypatch):
    import experiments.shared.audit.baselines as runner

    task = tasks_at(tmp_path)[0]
    task["methods"] = ["ws", "rs", "bt"]
    prepared = prepared_records()
    monkeypatch.setattr(runner, "load_finetuned_model", lambda *args, **kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr(runner, "TargetScorer", FakeScorer)
    observed = []

    def score(args, progress, scorer, records, auxiliary, tokenizer, methods, *, reference_cache):
        reused = "texts" in reference_cache
        observed.append((methods[0], reused))
        reference_cache.setdefault("texts", ["reference"] * len(records))
        # WS pays for reference + perturbation, RS only its perturbation,
        # and BT pays for its rewrite + scoring generations.
        prompts = [[1]] * len(records)
        outputs = [[[2]]] * len(records)
        for _ in range((0 if reused else 1) + (2 if methods[0] == "bt" else 1)):
            scorer.cost_meter.generation(prompts, outputs, set())
        return {methods[0]: np.linspace(0., 1., len(records))}

    monkeypatch.setattr(runner, "score_methods", score)
    runner.run_baselines(task, "cpu", SimpleNamespace(target_model="toy"), prepared,
                         {"files": [], "checkpoints": []})
    assert observed == [("ws", False), ("rs", True), ("bt", True)]
    expected_sequences = {"ws": 2 * 4200, "rs": 4200, "bt": 2 * 4200}
    for method, reused in observed:
        report = read_result(Path(task["output"]) / method)
        assert report["baseline_execution_mode"] == "shared_robustness_reference"
        assert report["cost"]["reference_reused"] is reused
        assert "incremental cost" in report["cost_conventions"]["reuse"]
        if reused:
            assert "amortized_ms_per_record" not in report["cost"]
            assert "total_seconds" not in report["cost"]
            assert "physical_incremental_amortized_ms_per_record" in report["cost"]
            assert report["cost"]["execution_group_seconds"] == report["cost"]["physical_incremental_total_seconds"]
            assert report["cost"]["physical_incremental_target_sequences"] == expected_sequences[method]
            assert report["cost"]["physical_incremental_generated_tokens"] == expected_sequences[method]
            assert "physical_incremental_phase_work" in report
        else:
            assert report["cost"]["cost_basis"] == "standalone_measured"
            assert "amortized_ms_per_record" in report["cost"]
            assert report["cost"]["target_sequences"] == expected_sequences[method]
            assert report["cost"]["generated_tokens"] == expected_sequences[method]

    from experiments.sd_membership_sft.audit import qwen_audit_matrix as matrix
    monkeypatch.setattr(matrix, "ALL_METHODS", ("ws", "rs", "bt"))
    summary = matrix.summarize([task], tmp_path / "summary")
    assert summary["complete"]
    physical = sum(read_result(Path(task["output"]) / method)["cost"].get(
        "execution_group_seconds", read_result(Path(task["output"]) / method)["cost"].get("total_seconds"))
        for method in ("ws", "rs", "bt"))
    assert summary["unique_successful_measured_method_seconds"] == pytest.approx(physical)
    rs_row = next(row for row in summary["rows"] if row["method"] == "rs")
    assert "amortized_ms_per_record" not in rs_row
    assert "physical_incremental_amortized_ms_per_record" in rs_row


def test_failed_reference_owner_does_not_make_next_cost_incremental(tmp_path, monkeypatch):
    import experiments.shared.audit.baselines as runner

    task = tasks_at(tmp_path)[0]
    task["methods"] = ["ws", "rs"]
    monkeypatch.setattr(runner, "load_finetuned_model", lambda *args, **kwargs: torch.nn.Linear(1, 1))
    monkeypatch.setattr(runner, "TargetScorer", FakeScorer)
    observed = []

    def score(args, progress, scorer, records, auxiliary, tokenizer, methods, *, reference_cache):
        observed.append((methods[0], "texts" in reference_cache))
        reference_cache.setdefault("texts", ["reference"] * len(records))
        scorer.cost_meter.generation([[1]] * len(records), [[[2]]] * len(records), set())
        if methods[0] == "ws":
            raise RuntimeError("scoring failed after reference generation")
        return {methods[0]: np.linspace(0., 1., len(records))}

    monkeypatch.setattr(runner, "score_methods", score)
    with pytest.raises(RuntimeError, match="failed baseline methods"):
        runner.run_baselines(task, "cpu", SimpleNamespace(target_model="toy"), prepared_records(),
                             {"files": [], "checkpoints": []})
    assert observed == [("ws", False), ("rs", False)]
    assert read_result(Path(task["output"]) / "rs")["cost"]["cost_basis"] == "standalone_measured"


def test_shared_reference_cli_protects_historical_output_root(monkeypatch):
    import sys
    from experiments.sd_membership_sft.audit.qwen_audit_matrix import main
    from experiments.sd_membership_sft.audit.cli import main as current_main

    from experiments.sd_membership_sft.audit.qwen_audit_matrix import LEGACY_OUTPUT_ROOT

    for entry in (main, current_main):
        monkeypatch.setattr(sys, "argv", ["qwen_audit_matrix", "status", "--output-root", str(LEGACY_OUTPUT_ROOT)])
        with pytest.raises(SystemExit) as error:
            entry()
        assert error.value.code == 2


def test_shared_reference_cli_records_mode_in_task_settings(tmp_path, monkeypatch, capsys):
    import sys
    from experiments.sd_membership_sft.audit import cli

    settings_seen = []
    monkeypatch.setattr(cli, "fixed_tasks", lambda *args: settings_seen.append(args[-1]) or [])
    monkeypatch.setattr(sys, "argv", ["audit", "status", "--output-root", str(tmp_path)])
    cli.main()
    capsys.readouterr()
    assert settings_seen[0]["baseline_execution"] == "shared_robustness_reference_v1"


@pytest.mark.parametrize("role", ["draft_auxiliary_distilled", "draft_member_sft"])
def test_main_matrix_uses_both_draft_roles_and_resumes_frozen_detector(tmp_path, role):
    from experiments.shared.audit.main import run_main
    from experiments.shared.protocols.protocol_archive import save_archive

    task = next(t for t in tasks_at(tmp_path) if t.get("protocol") == "fixed" and t.get("draft_role") == role)
    prepared = prepared_records()
    output = Path(task["output"])
    output.mkdir(parents=True)
    x = np.zeros((4600, 6), dtype=np.float32)
    x[:, 0], x[:, 1], x[:, 5] = -.8, .5, .5
    counts = (np.arange(4600) % 3).astype(np.uint8)
    data = dict(features=x, counts=counts, lengths=np.ones(4600, dtype=int),
                document_indices=np.arange(4600), start_indices=np.zeros(4600, dtype=int),
                record_ids=prepared.record_ids, record_roles=prepared.record_roles, labels=prepared.labels)
    sources = {"files": [], "checkpoints": []}
    contract = dict(protocol="fixed", starts=["fixed"],
                    rounds_per_start=0, sources=sources, matrix_request_key=digest(task), hardware={"device": "cpu"})
    costs = [dict(record_id=str(i), seconds=.01, target_forward_calls=1, draft_forward_calls=1,
                  target_input_tokens=3, draft_input_tokens=3, generated_tokens=1,
                  peak_allocated_gpu_bytes=None) for i in range(4600)]
    save_archive(output / "observations.npz", data, contract, costs)
    run_main(task, "cpu", None, prepared, sources)
    report = read_result(output / task["methods"][0])
    assert report["training_member_count"] == 0
    assert report["metrics"]["n_calibration"] == 200
    assert report["metrics"]["n_test_member"] == 2000
    assert report["draft_role"] == role
    assert report["collection_phase_seconds"]["preparation"] == pytest.approx(4.)
    old_detector = (output / "detector.pt").read_bytes()
    (output / task["methods"][-1] / "REPORT.json").unlink()
    run_main(task, "cpu", None, prepared, sources)
    assert (output / "detector.pt").read_bytes() == old_detector


def test_gpu_preflight_rejects_busy_or_invalid_indices(monkeypatch):
    import experiments.sd_membership_sft.audit.qwen_audit_matrix as launcher
    def query(command, **kwargs):
        return "0, GPU-a\n1, GPU-b\n" if "--query-gpu=index,uuid" in command else "GPU-b, 123\n"
    monkeypatch.setattr(launcher.subprocess, "check_output", query)
    check_gpus([0])
    for values in ([1], [2], [0, 0], [-1]):
        with pytest.raises((ValueError, RuntimeError)):
            check_gpus(values)


def test_draft_checkpoint_roles_cannot_silently_fall_back_to_kd(tmp_path):
    from experiments.shared.models.loading import checkpoint_paths

    for role in ("target", "draft_auxiliary_distilled", "draft_member_sft"):
        (tmp_path / "checkpoints" / role).mkdir(parents=True)
    assert checkpoint_paths(tmp_path, "plain", "draft_member_sft")[1].name == "draft_member_sft"
    with pytest.raises(ValueError):
        checkpoint_paths(tmp_path, "plain", "wrong")
    with pytest.raises(FileNotFoundError, match="member_head"):
        checkpoint_paths(tmp_path, "mtp", "draft_member_sft")


def test_missing_training_is_pending_and_wrong_model_is_invalid(tmp_path):
    task = tasks_at(tmp_path)[0]
    assert ready(task)[1].startswith("pending")
    run = Path(task["run_dir"])
    run.mkdir(parents=True)
    (run / "results.json").write_text(json.dumps({"material_passport": {"status": "COMPLETED"}, "config": {}}))
    assert ready(task)[1].startswith("invalid")


def test_modified_source_and_partial_results_are_not_complete(tmp_path):
    from experiments.shared.core.deployment_archive import sha256_file
    from experiments.sd_membership_sft.audit.qwen_audit_matrix import inspect_task

    task = tasks_at(tmp_path)[0]
    folder = write_example_result(task, "loss", .8)
    assert inspect_task(task)["state"] != "complete"
    source = tmp_path / "source.py"
    source.write_text("original")
    path = folder / "REPORT.json"
    report = json.loads(path.read_text())
    report["sources"]["files"] = [{"path": str(source), "sha256": sha256_file(source)}]
    path.write_text(json.dumps(report))
    read_result(folder)
    source.write_text("modified")
    assert inspect_task(task)["state"] == "stale"


def test_invalid_partition_indices_fail_clearly(tmp_path):
    for bad in ([-1], [3], [[0]]):
        with pytest.raises(ValueError, match="partitions"):
            metrics([.1, .2, .3], [0, 1, 0], bad, [1, 2], bootstrap=0)
    task = tasks_at(tmp_path)[0]
    folder = write_example_result(task, "loss", .8)
    report = json.loads((folder / "REPORT.json").read_text())
    save_result(folder, record_ids=np.array(["cal", "member", "nonmember"]),
                labels=np.array([0, 1, 0]), scores=np.array([.5, .8, .2]),
                calibration=np.array([0.]), test=np.array([1, 2]), report=report)
    with pytest.raises(ValueError, match="partition"):
        read_result(folder)


@pytest.mark.parametrize("force_kill", [False, True])
def test_cancelled_worker_is_reaped_before_return(force_kill):
    import subprocess
    import threading
    from experiments.sd_membership_sft.audit.qwen_audit_matrix import wait_worker

    stop = threading.Event()
    class Child:
        terminated = killed = reaped = False
        def poll(self):
            return None
        def terminate(self):
            self.terminated = True
        def kill(self):
            self.killed = True
        def wait(self, timeout=None):
            if timeout == 1:
                stop.set()
                raise subprocess.TimeoutExpired("fake", timeout)
            if timeout == 10 and force_kill:
                raise subprocess.TimeoutExpired("fake", timeout)
            self.reaped = True
            return -9 if self.killed else -15
    child = Child()
    assert wait_worker(child, stop, lambda: None) is None
    assert child.terminated and child.reaped
    assert child.killed == force_kill


def test_generation_batch_accounting_keeps_calibration_separate():
    from experiments.shared.audit.baselines import PhaseMeter, PhaseProgress

    meter = PhaseMeter("cpu", 4000)
    progress = PhaseProgress(meter, "cpu", 200, 8)
    phases = []
    for start in progress.track([192, 200], "generation", unit="batches"):
        phases.append(meter.phase)
    assert phases == ["calibration", "test"]
    with pytest.raises(ValueError, match="crosses"):
        list(progress.track([196], "generation", unit="batches"))
