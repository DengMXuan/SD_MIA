"""DP-specific tests are optional unless the project's `dp` extra is installed."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("opacus")

from experiments.dp_defense.accounting import make_plan, epsilon_for, pair_budgets
from experiments.dp_defense.training import DocumentGradientSum, PrivateRandomness, dp_sft_train
from experiments.dp_defense.artifacts import owned_run, save_stage, read_stage, stage_key, verify_run
from experiments.dp_defense.audit import make_tasks
from experiments.dp_defense.sweep import commands
from experiments.sd_membership_sft.data import SFTRecord
from experiments.sd_membership_sft.matrix_artifacts import digest, runtime_files, save_result


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("epsilon", [1., 4., 8.])
def test_noise_calibration_and_pair_budgets(epsilon):
    plan = make_plan(epsilon=epsilon)
    assert plan.steps == 125 and plan.sample_rate == .008
    assert 0 < plan.accounted_epsilon <= epsilon
    assert plan.accounted_epsilon == epsilon_for(plan.noise_multiplier, .008, 125, 5e-6)
    pairs = pair_budgets(plan.as_dict(), plan.as_dict())
    assert pairs["draft_auxiliary_distilled"]["epsilon_cap"] == epsilon
    assert pairs["draft_member_sft"]["epsilon_cap"] == 2 * epsilon
    assert pairs["draft_auxiliary_distilled"]["delta"] == 5e-6
    assert pairs["draft_member_sft"]["delta"] == 1e-5
    # More queries to a trained model are not events in this training accountant.
    assert epsilon_for(plan.noise_multiplier, .008, 250, 5e-6) > plan.accounted_epsilon


@pytest.mark.parametrize("kwargs", [dict(epsilon=0), dict(epsilon=float("nan")),
    dict(epsilon=1, delta=0), dict(epsilon=1, max_grad_norm=0),
    dict(epsilon=1, epochs=0), dict(epsilon=1, expected_batch_size=2001)])
def test_invalid_plans_fail_before_training(kwargs):
    with pytest.raises(ValueError):
        make_plan(**kwargs)


class FixedNoise:
    def __init__(self, selected=(), value=0.):
        self.selected, self.value, self.calls = selected, value, 0

    def select(self, size, probability):
        return self.selected

    def normal(self, size, device):
        self.calls += 1
        return torch.full((size,), self.value, device=device, dtype=torch.float32)


def test_clip_each_document_globally_before_aggregation():
    a, b = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    accumulator = DocumentGradientSum([a, b], 1.)
    a.grad, b.grad = torch.tensor([3.]), torch.tensor([4.])
    accumulator.add_document()
    torch.testing.assert_close(accumulator.sums[0], torch.tensor([.6]))
    torch.testing.assert_close(accumulator.sums[1], torch.tensor([.8]))
    a.grad, b.grad = torch.tensor([-300.]), torch.tensor([-400.])
    accumulator.add_document()
    accumulator.set_noisy_gradients(1., 2, FixedNoise())
    torch.testing.assert_close(a.grad, torch.zeros(1), atol=1e-6, rtol=0)
    torch.testing.assert_close(b.grad, torch.zeros(1), atol=1e-6, rtol=0)


def test_noise_is_scaled_before_fixed_normalization_and_covers_unused_parameters():
    a, unused = torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(1))
    accumulator = DocumentGradientSum([a, unused], 2.)
    a.grad = torch.tensor([2., 0.])
    accumulator.add_document()
    accumulator.set_noisy_gradients(3., 4, FixedNoise(value=1.))
    torch.testing.assert_close(a.grad, torch.tensor([2., 1.5]))
    torch.testing.assert_close(unused.grad, torch.tensor([1.5]))
    assert all(torch.count_nonzero(s) == 0 for s in accumulator.sums)


def test_nonfinite_document_maps_to_zero_without_contaminating_other_documents():
    p = torch.nn.Parameter(torch.zeros(2))
    accumulator = DocumentGradientSum([p], 1.)
    p.grad = torch.tensor([float("nan"), 1.])
    accumulator.add_document()
    p.grad = torch.tensor([0., 2.])
    accumulator.add_document()
    accumulator.set_noisy_gradients(1., 1, FixedNoise())
    torch.testing.assert_close(p.grad, torch.tensor([0., 1.]))


def test_randomness_is_not_reconstructed_by_public_training_seed():
    torch.manual_seed(1919)
    first = PrivateRandomness("cpu").normal(100000, "cpu")
    torch.manual_seed(1919)
    second = PrivateRandomness("cpu").normal(100000, "cpu")
    assert not torch.equal(first, second)
    assert abs(first.mean().item()) < .025
    assert .96 < first.var().item() < 1.04


def test_empty_poisson_batches_still_update_with_noise_and_spend_budget(monkeypatch):
    from experiments.dp_defense import training
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.zero_()
    plan = make_plan(epsilon=4, population=2, expected_batch_size=1)
    monkeypatch.setattr(training, "_make_optimizer", lambda model, lr, name: torch.optim.SGD(model.parameters(), lr=lr))
    rng, progress = FixedNoise(value=1.), []
    result = dp_sft_train(model, [], SimpleNamespace(pad_token_id=0), "cpu", plan,
                           lr=.1, optimizer_name="adamw", _randomness=rng,
                           progress=lambda step, total: progress.append((step, total)))
    assert rng.calls == plan.steps == result["completed_steps"]
    assert progress == [(1, 2), (2, 2)]
    assert model.weight.item() == pytest.approx(-.1 * plan.steps * plan.noise_multiplier)
    assert result["private_training_metrics_released"] is False
    assert "loss" not in result and "seed" not in result


def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=32, tie_word_embeddings=True))


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def save_pretrained(self, path):
        (path / "tokenizer_config.json").write_text('{"test": true}')


def test_real_tiny_qwen_full_parameter_training_and_tied_weights():
    model = tiny_model()
    before = model.model.embed_tokens.weight.detach().clone()
    records = [SFTRecord(str(i), "test", (4, 5, 6), str(i), prompt_ids=(1, 3)) for i in range(2)]
    plan = make_plan(epsilon=4, population=2, expected_batch_size=2)
    result = dp_sft_train(model, records, TinyTokenizer(), "cpu", plan, lr=.01, optimizer_name="adamw")
    assert result["completed_steps"] == 1
    assert not torch.equal(before, model.model.embed_tokens.weight)
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert not model.training and all(p.grad is None for p in model.parameters())


def test_duplicate_document_ids_rejected():
    record = SFTRecord("a", "test", (4, 5), "hash", prompt_ids=(1,))
    plan = make_plan(epsilon=4, population=2, expected_batch_size=2)
    with pytest.raises(ValueError, match="duplicate document"):
        dp_sft_train(tiny_model(), [record, record], TinyTokenizer(), "cpu", plan)


def test_output_ownership_rejects_old_results_and_changed_budget(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    (old / "results.json").write_text('{}')
    with pytest.raises(ValueError, match="nonempty"):
        with owned_run(old, {"epsilon": 1}):
            pass
    output = tmp_path / "new"
    with owned_run(output, {"epsilon": 1}):
        pass
    with owned_run(output, {"epsilon": 1}):
        pass
    with pytest.raises(ValueError, match="changed"):
        with owned_run(output, {"epsilon": 4}):
            pass
    assert json.loads((old / "results.json").read_text()) == {}


def test_atomic_stage_recovery_and_hash_teacher_checks(tmp_path):
    request = {"epsilon": 1}
    with owned_run(tmp_path / "run", request) as output:
        key = stage_key(request, "target")
        saved = save_stage(output, "target", key, tiny_model(), TinyTokenizer(), {"epsilon": 1})
        assert read_stage(output, "target", key) == saved
        with pytest.raises(ValueError, match="teacher mismatch"):
            read_stage(output, "target", "bad-key")
        with pytest.raises(ValueError, match="overwrite"):
            save_stage(output, "target", key, tiny_model(), TinyTokenizer(), {})
        (output / "checkpoints/target/tokenizer_config.json").write_text('{}')
        with pytest.raises(ValueError, match="checksum"):
            read_stage(output, "target", key)
    assert stage_key(request, "draft_auxiliary_distilled", "teacher-1") != stage_key(request, "draft_auxiliary_distilled", "teacher-2")


def test_complete_passport_accounting_verified_and_tampering_rejected(tmp_path):
    plan = make_plan(epsilon=4, population=2, expected_batch_size=2)
    dp = {**plan.as_dict(), "completed_steps": plan.steps}
    request = {"plans": {"target": plan.as_dict(), "draft_member_sft": plan.as_dict()}, "sources": []}
    with owned_run(tmp_path / "run", request) as output:
        stages = {}
        for role in ("target", "draft_auxiliary_distilled", "draft_member_sft"):
            teacher = stages["target"]["checkpoint_sha256"] if role == "draft_auxiliary_distilled" else None
            stages[role] = save_stage(output, role, stage_key(request, role, teacher), tiny_model(), TinyTokenizer(),
                                      dp if role != "draft_auxiliary_distilled" else {"teacher_sha256": teacher})
        artifact = {"privacy": {"request_key": digest(request), "stages": stages, "pairs": pair_budgets(dp, dp)}}
        (output / "results.json").write_text(json.dumps(artifact))
        assert verify_run(output) == artifact
        artifact["privacy"]["pairs"]["draft_member_sft"]["epsilon"] = .1
        (output / "results.json").write_text(json.dumps(artifact))
        with pytest.raises(ValueError, match="pair accounting"):
            verify_run(output)


def test_training_lifecycle_uses_base_initializations_and_resumes_completed_stages(tmp_path, monkeypatch):
    from experiments.dp_defense import train as runner
    from experiments.dp_defense import training as dp_training
    from experiments.sd_membership_sft import training as legacy_training
    from experiments.sd_membership_sft.drafts import plain
    from experiments.sd_membership_sft.config import Config
    from experiments.sd_membership_sft.data import records_metadata

    output = tmp_path / "run"
    cfg = Config(trainer="full", optimizer="adamw", output_dir=output, n_per_class=2,
                 n_aux=2, n_audit_aux=2, target_batch_size=1, target_grad_accum=2,
                 draft_batch_size=1, draft_grad_accum=2)
    plan = make_plan(epsilon=4, population=2, expected_batch_size=2)
    plans = {"target": plan, "draft_member_sft": plan}
    roles = [[SFTRecord(f"{role}-{i}", "test", (4, 5), f"{role}-{i}", prompt_ids=(1,))
              for i in range(2)] for role in ("members", "nonmembers", "auxiliary", "audit_auxiliary")]
    metadata = {"shared_split_sha256": "split"}
    reference = {"material_passport": {"status": "COMPLETED"},
                 "records": dict(zip(("members", "nonmembers", "auxiliary", "audit_auxiliary"),
                                     [records_metadata(r) for r in roles]))}
    request = {"config": cfg.as_dict(), "plans": {k: v.as_dict() for k,v in plans.items()},
               "sources": [], "scope": "models_only"}
    monkeypatch.setattr(runner, "prepare_request", lambda *args: (cfg, reference, tmp_path / "split", plans, request))
    monkeypatch.setattr(plain, "_load_condition_split", lambda *args: (*roles, metadata))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(legacy_training, "set_seed", lambda seed: None)
    tokenizer = TinyTokenizer()
    tokenizer.get_vocab = lambda: {str(i): i for i in range(32)}
    monkeypatch.setattr(legacy_training, "load_tokenizer", lambda *args, **kwargs: tokenizer)
    loaded, trained, distilled = [], [], []

    def load(model_id, device, **kwargs):
        loaded.append(model_id)
        return tiny_model()

    calls = 0
    def dp_train(model, records, tokenizer, device, plan, **kwargs):
        nonlocal calls
        calls += 1
        trained.append([r.record_id for r in records])
        if calls == 2:
            raise RuntimeError("simulated interruption during member training")
        return {**plan.as_dict(), "completed_steps": plan.steps}

    def distill(model, teacher, records, *args, **kwargs):
        distilled.append([r.record_id for r in records])

    monkeypatch.setattr(legacy_training, "load_causal_lm", load)
    monkeypatch.setattr(legacy_training, "distill_on_auxiliary", distill)
    monkeypatch.setattr(dp_training, "dp_sft_train", dp_train)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        runner.run(tmp_path / "reference", output, 4, 1., 0)
    assert (output / "checkpoints/target/DP_STAGE.json").exists()
    assert (output / "checkpoints/draft_auxiliary_distilled/DP_STAGE.json").exists()
    assert not (output / "results.json").exists()
    assert loaded == [cfg.target_model, cfg.draft_model, str(output / "checkpoints/target"), cfg.draft_model]
    runner.run(tmp_path / "reference", output, 4, 1., 0)
    assert loaded[-1] == cfg.draft_model and len(loaded) == 5
    assert len(distilled) == 1 and distilled[0] == [r.record_id for r in roles[2]]
    assert all(ids == [r.record_id for r in roles[0]] for ids in trained)
    result = verify_run(output)
    assert result["privacy"]["pairs"]["draft_member_sft"]["epsilon_cap"] == 8
    runner.run(tmp_path / "reference", output, 4, 1., 0)
    assert len(loaded) == 5


def test_audit_tasks_retain_legacy_settings_and_separate_detectors(tmp_path):
    artifact = {"config": {"benchmark": "wikitection", "target_epochs": 1, "seed": 1919},
                "privacy": {"request_key": "key"}}
    tasks = make_tasks(tmp_path / "models", tmp_path / "audit", artifact, baselines=True)
    assert len(tasks) == 3 and len(tasks[-1]["methods"]) == 11
    main = tasks[:2]
    assert all(t["protocol"] == "fixed" and t["methods"] == ["main_fixed_sparse_positive"] for t in main)
    assert main[0]["output"] != main[1]["output"]
    assert main[0]["run_dir"] == main[1]["run_dir"]
    assert all(t["settings"]["detector_epochs"] == 30 for t in main)
    assert not any("dp_defense" in p.parts for p in runtime_files())


def test_default_matrix_has_54_conditions_with_one_shared_target_each(tmp_path):
    args = SimpleNamespace(benchmarks=["wikitection", "newstection", "arxivtection"], epochs=[1, 3],
        seeds=[1919, 1949, 1978], epsilons=[1., 4., 8.], reference_root=tmp_path / "reference",
        model_root=tmp_path / "models", audit_root=tmp_path / "audits", gpu=0, include_baselines=False)
    tasks = commands(args)
    assert len(tasks) == 54
    assert len({tuple(t["train"]) for t in tasks}) == 54
    assert all("experiments.dp_defense.audit" in task["audit"] for task in tasks)


@pytest.mark.parametrize("dp_pair,reference_pair,dp_role,reference_role", [
    ({}, {}, "draft_member_sft", "draft_member_sft"),
    ({"model_pair": "qwen3"}, {}, "draft_member_sft", "draft_member_sft"),
    ({"model_pair": "qwen3_8b_eagle3"}, {"pair": "qwen3_8b_eagle3"},
     "member_head", "draft_member_sft"),
])
def test_comparison_requires_identical_records_and_preserves_signed_changes(
        tmp_path, dp_pair, reference_pair, dp_role, reference_role):
    from experiments.dp_defense.compare import compare_reports, FIELDS
    common = dict(condition={"benchmark": "wikitection", "epoch": 1, "condition_seed": 1919},
                  method="main_fixed_sparse_positive", draft_role="draft_member_sft", settings={"seed": 1},
                  request_key="key", sources={"files": [], "checkpoints": []})
    for name, auc in (("dp", .5), ("reference", .8)):
        report = {**common, "metrics": {field: auc for field in FIELDS}}
        report["condition"] = {**common["condition"], **(dp_pair if name == "dp" else reference_pair)}
        report["draft_role"] = dp_role if name == "dp" else reference_role
        if name == "dp":
            report["privacy"] = dict(target_epsilon_cap=1., epsilon=2., delta=1e-5)
        save_result(tmp_path / name, record_ids=np.array(["cal", "mem", "non"]), labels=np.array([0, 1, 0]),
                    scores=np.array([0., 1., .5]), calibration=np.array([0]), test=np.array([1, 2]), report=report)
    row = compare_reports(tmp_path / "dp/REPORT.json", tmp_path / "reference/REPORT.json")
    assert row["change_auc"] == pytest.approx(-.3)
    assert row["model_pair"] == dp_pair.get("model_pair", "qwen3")
    reference = json.loads((tmp_path / "reference/REPORT.json").read_text())
    reference["draft_role"] = "draft_auxiliary_distilled"
    (tmp_path / "reference/REPORT.json").write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="draft"):
        compare_reports(tmp_path / "dp/REPORT.json", tmp_path / "reference/REPORT.json")
    reference["draft_role"] = reference_role
    reference["condition"]["model_pair"] = "gemma4"
    (tmp_path / "reference/REPORT.json").write_text(json.dumps(reference))
    with pytest.raises(ValueError, match="condition"):
        compare_reports(str(tmp_path / "dp/REPORT.json"), str(tmp_path / "reference/REPORT.json"))
