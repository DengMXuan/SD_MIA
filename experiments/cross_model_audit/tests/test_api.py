"""Single-condition APIs: real CPU detector fitting with synthetic observations."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.cross_model_audit import api, engine, models, main_method, head_validation
from experiments.cross_model_audit.model_registry import MODEL_PAIRS
from experiments.sd_membership_sft.core.deployment_archive import sha256_file
from experiments.sd_membership_sft.protocols.protocol_archive import save_archive
from experiments.sd_membership_sft.tests.test_qwen_audit_matrix import prepared_records


@pytest.mark.parametrize("pair", list(MODEL_PAIRS))
@pytest.mark.parametrize("private", [False, True])
def test_evaluate_main_fits_resumes_and_accounts_all_pairs(tmp_path, monkeypatch, pair, private):
    torch.set_num_threads(1)
    spec = MODEL_PAIRS[pair]
    run = tmp_path / "models"
    run.mkdir()
    cfg = dict(target_model=spec.target, draft_model=spec.draft,
               target_revision=spec.target_revision, draft_revision=spec.draft_revision,
               benchmark="wikitection", target_epochs=1, seed=1919)
    artifact = dict(config=cfg, protocol_track=dict(pair=spec.name))
    verified = []
    if private:
        from experiments.dp_defense import artifacts
        artifact["privacy"] = dict(request_key="dp-key",
            stages={"target": {"privacy": {"epsilon": 4.}}},
            pairs={"draft_auxiliary_distilled": dict(epsilon=4., delta=5e-6),
                   "draft_member_sft": dict(epsilon=8., delta=1e-5)})
        (run / "DP_REQUEST.json").write_text('{}')
        (run / ".dp.lock").touch()
        def verify(path):
            verified.append(path)
            return artifact
        monkeypatch.setattr(artifacts, "verify_run", verify)
    passport = run / "results.json"
    passport.write_text(json.dumps(artifact))
    monkeypatch.setattr(engine, "ready", lambda _: (True, "ready"))
    prepared = prepared_records()
    routes = []
    def prepare(path, adapter, role):
        routes.append((adapter, role))
        return SimpleNamespace(**cfg), prepared
    monkeypatch.setattr(models, "prepare_records", prepare)
    monkeypatch.setattr(api, "sources_for", lambda *a, **k: dict(checkpoints=[],
        files=[dict(path=str(passport), sha256=sha256_file(passport))]))
    monkeypatch.setattr(main_method, "load_adapter", lambda *a: SimpleNamespace(
        target=torch.nn.Linear(1, 1), device=torch.device("cpu"), kind=spec.adapter))
    monkeypatch.setattr(main_method, "fixed_trace", lambda *a, **k: None)
    monkeypatch.setattr(head_validation, "validate_adapter", lambda *a: dict(status="passed"))
    collected = []
    def collect(records, adapter, output, contract):
        collected.append(output)
        x = np.zeros((4600, 6), np.float32)
        x[:, 0], x[:, 1] = -.8, .5
        arrays = dict(features=x, counts=(np.arange(4600) % 3).astype(np.uint8),
            lengths=np.ones(4600, dtype=int), document_indices=np.arange(4600),
            start_indices=np.zeros(4600, dtype=int), record_ids=records.record_ids,
            record_roles=records.record_roles, labels=records.labels)
        costs = [dict(record_id=str(i), seconds=.01, target_forward_calls=1,
            draft_forward_calls=1, target_input_tokens=3, draft_input_tokens=3,
            generated_tokens=0, hidden_state_bytes=24, supported_candidates=1,
            candidate_positions=1) for i in range(4600)]
        save_archive(output / "observations.npz", arrays, contract, costs)
    monkeypatch.setattr(main_method, "collect_records", collect)
    for index, role in enumerate(spec.roles):
        output = tmp_path / "audit" / role
        report = api.evaluate_main(run, output, draft_role=role, device="cpu", detector_epochs=1)
        assert routes[-1] == (spec.adapter, role)
        assert report["training_member_count"] == 0
        assert report["metrics"]["n_calibration"] == 200
        assert report["model_pair"] == pair
        assert report["detector_features"] == "draft_features_and_acceptance_only"
        if spec.adapter != "plain":
            assert "hidden_states" in report["access_channel"]
            assert report["head_validation"]["status"] == "passed"
        if private:
            assert report["privacy"]["epsilon"] == (4. if index == 0 else 8.)
            assert report["privacy"]["dp_request_key"] == "dp-key"
            # Resume the gap between saving scores and attaching the privacy ledger.
            path = output / report["method"] / "REPORT.json"
            unannotated = {k: v for k, v in report.items() if k != "privacy"}
            path.write_text(json.dumps(unannotated))
        checkpoint = (output / "detector.pt").read_bytes()
        again = api.evaluate_main(run, output, draft_role=role, device="cpu", detector_epochs=1)
        assert again == report
        assert (output / "detector.pt").read_bytes() == checkpoint
        with pytest.raises(ValueError, match="request changed"):
            api.evaluate_main(run, output, draft_role=role, device="cpu", detector_epochs=2)
    assert len(collected) == 2
    assert bool(verified) == private


def test_private_request_without_private_passport_cannot_be_audited(tmp_path):
    (tmp_path / "results.json").write_text('{"config": {}}')
    (tmp_path / "DP_REQUEST.json").write_text('{}')
    with pytest.raises((ValueError, KeyError)):
        api.inspect_run(tmp_path)


def test_api_protects_active_outputs_and_resolved_model_paths(tmp_path):
    from experiments.paths import QWEN_AUDIT
    run, weights = tmp_path / "run", tmp_path / "weights"
    run.mkdir()
    weights.mkdir()
    (run / "heads").symlink_to(weights, target_is_directory=True)
    alias = tmp_path / "alias"
    alias.symlink_to(QWEN_AUDIT, target_is_directory=True)
    for output in (run, run / "audit", run.parent, weights / "audit", alias):
        with pytest.raises(ValueError, match="separate"):
            api.evaluate_main(run, output, draft_role="member_head", device="cpu")


def test_head_gate_rejects_future_context_leakage():
    from experiments.sd_membership_sft.protocols.sd_protocol import RuntimeCost
    class Adapter:
        device = torch.device("cpu")
        cost = RuntimeCost()
        leak = False
        def rows(self, tokens):
            logits = torch.zeros(len(tokens), 4)
            if self.leak:
                logits[:, 0] = len(tokens)
            logp = logits.log_softmax(-1)
            return logp, logp
        def next(self, tokens):
            p, q = self.rows(tokens)
            return p[-1], q[-1]
    adapter = Adapter()
    assert head_validation.validate_adapter(adapter, [0, 1], [2, 3, 1, 2])["status"] == "passed"
    adapter.leak = True
    with pytest.raises(ValueError, match="future-token"):
        head_validation.validate_adapter(adapter, [0, 1], [2, 3, 1, 2])
