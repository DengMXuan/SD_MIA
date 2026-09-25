from dataclasses import asdict
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from experiments.shared.protocols.sd_protocol import FEATURE_NAMES, FrozenAdapter, RuntimeCost, draft_features, fixed_trace, map_eagle_logits, trajectory_seed
from experiments.shared.protocols.protocol_archive import load_archive, save_archive, validate_arrays
from experiments.shared.methods.protocol_accept_only import document_scores, evidence_components, trajectory_partitions


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_document_random_streams_are_stable_and_distinct():
    assert trajectory_seed(7, "doc-a", "fixed") == trajectory_seed(7, "doc-a", "fixed")
    assert trajectory_seed(7, "doc-a", "fixed") != trajectory_seed(7, "doc-b", "fixed")
    assert trajectory_seed(7, "doc-a", "fixed") != trajectory_seed(8, "doc-a", "fixed")


class ControlledAdapter:
    device = torch.device("cpu")

    def __init__(self):
        self.contexts = []
        self.cost = RuntimeCost()



    def rows(self, tokens):
        self.contexts.append(list(tokens))
        self.cost.target_forward_calls += 1
        q = torch.tensor([.8, .2, 0.]).log().repeat(len(tokens), 1)
        return q.clone(), q






def test_fixed_probes_exclude_and_count_unsupported_candidates():
    trace = fixed_trace(ControlledAdapter(), [0], [1, 2, 0], seed=3)
    assert trace["counts"].tolist() == [2, 2]
    assert trace["candidate_positions"] == 3
    assert trace["supported_candidates"] == 2
    assert trace["generated_tokens"] == 0
    with pytest.raises(ValueError, match="no supported"):
        fixed_trace(ControlledAdapter(), [0], [2], seed=3)


def test_eagle_mapping_uses_offsets_and_preserves_zero_support():
    logits = torch.tensor([[[1., 2., 3.]]])
    mapped = map_eagle_logits(logits, torch.tensor([1, 2, 2]), 6)
    torch.testing.assert_close(mapped[..., [1, 3, 4]], logits)
    assert torch.isneginf(mapped[..., [0, 2, 5]]).all()
    logq = mapped[0, 0].log_softmax(-1)
    assert np.isfinite(draft_features(logq, 3, .5, .75)).all()
    with pytest.raises(ValueError, match="outside draft support"):
        draft_features(logq, 0, 0., .5)
    with pytest.raises(ValueError, match="mapping"):
        map_eagle_logits(logits, torch.tensor([1, 0, -1]), 6)


class TinyTarget(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=8)
        self.weight = torch.nn.Parameter(torch.eye(4))
        self.calls = []

    def forward(self, input_ids, use_cache=False, output_hidden_states=False):
        self.calls.append(input_ids.clone())
        logits = self.weight[input_ids]
        hidden = tuple(logits + i for i in range(9)) if output_hidden_states else None
        return SimpleNamespace(logits=logits, hidden_states=hidden)


class TinyEagle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.Identity()
        self.lm_head = torch.nn.Linear(12, 2, bias=False)
        self.d2t = torch.tensor([1, 2])  # target IDs 1 and 3

    def forward(self, input_ids, hidden_states, **kwargs):
        self.hidden = hidden_states
        self.norm(hidden_states)


class TinyMTP(torch.nn.Module):
    """Enforce the installed MTP public API's embedding/target offset contract."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(num_speculative_steps=1)

    def forward(self, input_ids, hidden_states, loss_mask, **kwargs):
        valid = input_ids.shape[1] - 2
        self.embedded = input_ids[:, 1:1 + valid].clone()
        self.used_hidden = hidden_states[:, :valid].clone()
        assert not loss_mask.any()
        logits = self.used_hidden + torch.nn.functional.one_hot(self.embedded, 4)
        return [logits], None, {}


def test_plain_and_eagle_adapters_are_frozen_and_do_not_return_hidden_states():
    for kind, draft in (("plain", TinyTarget()), ("eagle3", TinyEagle())):
        target = TinyTarget()
        adapter = FrozenAdapter(target, draft, kind, "cpu")
        p, q = adapter.next([0, 1, 2])
        assert p.shape == q.shape == (4,)
        assert not p.requires_grad and not q.requires_grad
        assert all(not v.requires_grad for m in (target, draft) for v in m.parameters())
        if kind == "eagle3":
            assert torch.isneginf(q[[0, 2]]).all()
            assert draft.hidden.shape == (1, 3, 12)
            assert adapter.cost.hidden_state_bytes == 3 * 12 * 4


def test_mtp_next_prediction_uses_penultimate_hidden_and_last_context_token():
    target, head = TinyTarget(), TinyMTP()
    adapter = FrozenAdapter(target, head, "mtp", "cpu")
    p, q = adapter.next([0, 1, 2, 3])
    assert head.embedded.tolist() == [[1, 2, 3]]
    torch.testing.assert_close(head.used_hidden[0, -1], target.weight[2] + 8)
    expected = (target.weight[2] + 8 + torch.eye(4)[3]).log_softmax(-1)
    torch.testing.assert_close(q, expected)
    assert target.calls[-1].tolist() == [[0, 1, 2, 3]]  # placeholder never reaches target
    p, q = adapter.rows([0, 1, 2, 3])
    assert torch.isneginf(q[0]).all()
    torch.testing.assert_close(q[-1], expected)


def test_multi_step_mtp_exports_fail_explicitly():
    head = TinyMTP()
    head.config.num_speculative_steps = 2
    with pytest.raises(ValueError, match="depth-1"):
        FrozenAdapter(TinyTarget(), head, "mtp", "cpu")


def archive_fixture(n=4, starts=1):
    lengths = np.full(n * starts, 3, dtype=np.int64)
    x = np.zeros((int(lengths.sum()), 6), dtype=np.float32)
    x[:, 0] = -.7
    x[:, 1] = .5
    x[:, 4:6] = .5
    roles = np.asarray(["audit_auxiliary"] * (n - 2) + ["member", "nonmember"])
    data = dict(features=x, counts=np.zeros(len(x), dtype=np.uint8), lengths=lengths,
                document_indices=np.repeat(np.arange(n), starts), start_indices=np.tile(np.arange(starts), n),
                record_ids=np.asarray([f"record-{i}" for i in range(n)]), record_roles=roles,
                labels=(roles == "member").astype(int))
    contract = {"protocol": "fixed", "starts": ["fixed"],
                "rounds_per_start": 0, "sources": {}, "data_contract": "four_role_600"}
    return data, contract


def test_archive_roundtrip_and_checksum_rejects_changes(tmp_path):
    data, contract = archive_fixture()
    path = tmp_path / "observations.npz"
    save_archive(path, data, contract, [{}] * len(data["lengths"]))
    loaded, _ = load_archive(path)
    for key in data:
        np.testing.assert_array_equal(loaded[key], data[key])
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_archive(path)


@pytest.mark.parametrize("mutation", ["logp", "duplicate", "counts", "labels", "starts", "nan", "protocol"])
def test_archive_rejects_invalid_or_forbidden_data(mutation):
    data, contract = archive_fixture()
    if mutation == "logp":
        data["logp"] = np.zeros(len(data["features"]))
    elif mutation == "duplicate":
        data["document_indices"][1] = 0
    elif mutation == "counts":
        data["counts"][0] = 3
    elif mutation == "labels":
        data["labels"][0] = 1
    elif mutation == "starts":
        contract["starts"] = ["0.5"]
    elif mutation == "protocol":
        contract["protocol"] = "natural"
    else:
        data["features"][0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_arrays(data, contract)


def test_training_and_calibration_split_at_document_boundary():
    parts = {"train": np.array([0]), "validation": np.array([1]), "calibration": np.array([2]), "test": np.array([3])}
    owners = np.arange(4)
    mapped = trajectory_partitions(parts, owners)
    assert mapped["train"].tolist() == [0]
    assert mapped["calibration"].tolist() == [2]
    assert not set(mapped["train"]).intersection(mapped["test"])




def test_fit_is_independent_of_test_and_calibration_features_and_outcomes():
    data, _ = archive_fixture(n=6)
    parts = trajectory_partitions({"train": [0, 1], "validation": [2], "calibration": [3], "test": [4, 5]}, data["document_indices"])
    from experiments.shared.methods.difficulty_accept_only import fit
    x = data["features"][:, [0, 4, 1, 2, 3]].copy()
    first = fit(x, data["counts"], data["lengths"], parts, seed=17, device="cpu", epochs=2)
    x[9:] = 200
    counts = data["counts"].copy()
    counts[9:] = 2
    second = fit(x, counts, data["lengths"], parts, seed=17, device="cpu", epochs=2)
    for key, value in first[0].state_dict().items():
        torch.testing.assert_close(value, second[0].state_dict()[key], rtol=0, atol=0)
    np.testing.assert_array_equal(first[1], second[1])
    assert first[3:] == second[3:]


def test_signed_sparse_scores_match_explicit_likelihood_mixture():
    counts = np.array([1, 0], dtype=np.uint8)
    pmf = np.array([[.8, .2], [.4, .6]])
    components = evidence_components(np.log(pmf), counts, sparse=False)
    for i, eta in enumerate([.5, 1., 2., -.5, -1., -2.]):
        expected = counts.astype(float) * eta - np.log(pmf[:, 0] + pmf[:, 1] * np.exp(eta))
        np.testing.assert_allclose(components[:, 0, i], expected)
    sparse = evidence_components(np.log(pmf), counts, sparse=True)
    for j, rho in enumerate([.05, .1, .25]):
        np.testing.assert_allclose(sparse[:, j], np.log(1 - rho + rho * np.exp(components[:, 0])))


def test_combined_scores_are_document_level_and_order_invariant():
    data, _ = archive_fixture()
    data["counts"][::2] = 1
    logpmf = np.tile(np.log([.7, .3]), (len(data["counts"]), 1))
    combined = document_scores(data, logpmf)
    assert all(value.shape == (4,) for value in combined.values())
    order = np.arange(4)[::-1]
    events = np.concatenate([np.arange(3 * i, 3 * i + 3) for i in order])
    reordered = {**data, "features": data["features"][events], "counts": data["counts"][events],
                 "document_indices": data["document_indices"][order], "start_indices": data["start_indices"][order]}
    for key, values in document_scores(reordered, logpmf[events]).items():
        np.testing.assert_allclose(values, combined[key])


def test_installed_mtp_forward_placeholder_is_not_a_future_input(monkeypatch):
    """Exercise the actual installed forward's slicing with tiny layer stand-ins."""
    import inspect
    import speculators.models.mtp.core as core

    captured = []
    def layer(**kwargs):
        captured.append(kwargs)
        return kwargs["hidden_states"] + kwargs["token_embeddings"]
    monkeypatch.setattr(core, "create_causal_mask", lambda **kwargs: None)
    stub = SimpleNamespace(
        config=SimpleNamespace(num_speculative_steps=1, transformer_layer_config=None),
        embed_tokens=torch.nn.Embedding.from_pretrained(torch.eye(4)),
        rotary_emb=lambda h, p: (h, p), mtp_layers=[layer], lm_head=torch.nn.Identity(),
    )
    forward = inspect.unwrap(core.MTPDraftModel.forward)
    hidden = torch.arange(16).reshape(1, 4, 4).float()
    outputs = []
    for placeholder in (0, 3):
        ids = torch.tensor([[0, 1, 2, 3, placeholder]])
        logits, _, _ = forward(stub, ids, hidden, loss_mask=torch.zeros_like(ids))
        outputs.append(logits[0])
    torch.testing.assert_close(outputs[0], outputs[1])
    torch.testing.assert_close(outputs[0][0, -1], hidden[0, 2] + torch.eye(4)[3])
    assert captured[0]["position_ids"].tolist() == [[0, 1, 2]]


def test_transformers_batchencoding_prompt_compatibility():
    from experiments.shared.protocols.collect_protocol_observations import protocol_prompt_ids
    from experiments.shared.data.data import SFTRecord

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2] if kwargs.get("return_dict") is False else {"input_ids": [1, 2]}
    record = SFTRecord("a", "test", (1, 2), "hash", prompt_text="continue")
    assert protocol_prompt_ids(record, Tokenizer()) == [1, 2]


def test_fixed_collection_resumes_and_detects_corruption(tmp_path):
    from experiments.shared.protocols.collect_protocol_observations import collect_records
    from experiments.shared.data.data import SFTRecord

    record = SFTRecord("doc", "test", tuple([0, 1] * 4), "hash", prompt_ids=(1,))
    prepared = SimpleNamespace(
        records=[record], tokenizer=SimpleNamespace(eos_token_id=None),
        record_ids=np.array(["doc"]), record_roles=np.array(["smoke"]), labels=np.array([0]),
    )
    contract = {"starts": ["fixed"], "protocol": "fixed", "seed": 9,
                "rounds_per_start": 0, "sources": {}, "data_contract": "runtime_smoke_no_membership_claim"}
    first = ControlledAdapter()
    path = collect_records(prepared, first, tmp_path, contract)
    assert first.contexts == [[1, 0, 1, 0, 1, 0, 1, 0, 1]]
    data, envelope = load_archive(path)
    assert data["document_indices"].tolist() == [0]
    assert data["start_indices"].tolist() == [0]
    second = ControlledAdapter()
    collect_records(prepared, second, tmp_path, contract)
    assert not second.contexts
    with pytest.raises(ValueError, match="configuration"):
        collect_records(prepared, second, tmp_path, {**contract, "seed": 10})
    with (tmp_path / "trajectories/0_0.npz").open("ab") as stream:
        stream.write(b"bad")
    with pytest.raises(ValueError, match="hash mismatch"):
        collect_records(prepared, second, tmp_path, contract)


def test_evaluation_pipeline_calibrates_documents_and_saves_detector(tmp_path, monkeypatch):
    import experiments.shared.methods.protocol_accept_only as scoring

    data, contract = archive_fixture(n=10)
    # Small controlled fixture; production evaluate still requires 600+2000+2000.
    parts = {"train": np.array([0, 1, 2]), "validation": np.array([3, 4]),
             "reference": np.arange(5), "calibration": np.array([5, 6, 7]), "test": np.array([8, 9])}
    def partitions(*args, seed):
        assert seed == 1919
        return parts
    monkeypatch.setattr(scoring, "deployment_partitions", partitions)
    contract.update(protocol="fixed", adapter="plain", head_real_model_validation="not_applicable",
                    execution="full_context_reconstruction", seed=1919)
    data["counts"][::2] = 2
    archive = tmp_path / "observations.npz"
    save_archive(archive, data, contract, [{}] * len(data["lengths"]))
    scoring.evaluate(archive, tmp_path / "evaluation", epochs=2)
    report = json.loads((tmp_path / "evaluation/REPORT.json").read_text())
    assert report["training_member_count"] == 0
    assert report["partition_documents"]["calibration"] == 3
    assert "sparse_two_sided" in report["metrics"]["combined"]
    with np.load(tmp_path / "evaluation/scores.npz") as scores:
        assert scores["combined__sparse_positive"].shape == (10,)
    saved = torch.load(tmp_path / "evaluation/detector.pt", weights_only=True)
    assert saved["architecture"] == "difficulty_tcn_count_b2"
    assert saved["seed"] == report["source"]["seed"] == 1919


def test_smoke_archive_cannot_be_used_as_membership_evaluation(tmp_path):
    from experiments.shared.methods.protocol_accept_only import evaluate

    data, contract = archive_fixture()
    contract["data_contract"] = "runtime_smoke_no_membership_claim"
    path = tmp_path / "observations.npz"
    save_archive(path, data, contract, [{}] * len(data["lengths"]))
    with pytest.raises(ValueError, match="smoke"):
        evaluate(path, tmp_path / "evaluation")


def test_removed_protocol_cannot_collect_or_launch_a_saved_task(tmp_path):
    from experiments.shared.protocols.collect_protocol_observations import collect_records
    from experiments.shared.audit import main, fixed
    from experiments.sd_membership_sft.audit.qwen_audit_matrix import execute_worker

    output = tmp_path / "removed"
    contract = {"protocol": "natural", "starts": ["suffix64"]}
    with pytest.raises(ValueError, match="unsupported audit protocol"):
        collect_records(None, None, output, contract)
    task = dict(kind="main", protocol="natural", output=str(output))
    for runner in (main.run_main, fixed.run_main):
        with pytest.raises(ValueError, match="unsupported audit protocol"):
            runner(task, "cpu", None, None, {})
    with pytest.raises(ValueError, match="unsupported audit protocol"):
        execute_worker(task)
    assert not output.exists()
    assert main.MAIN_METHODS == fixed.MAIN_METHODS == {"fixed": ("main_fixed_sparse_positive",)}


def test_retired_serial_entry_points_are_not_importable():
    import importlib

    for name in ("experiments.shared.protocols.serial_accept_only",
                 "experiments.sd_membership_sft.serial_accept_only",
                 "experiments.sd_membership_sft.protocols.serial_accept_only"):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)


def test_direction_cost_summary_keeps_active_costs_without_serial_inputs(tmp_path):
    from experiments.sd_membership_sft.analysis.summarize_direction_validation import protocol_costs

    assert protocol_costs(tmp_path)["joint_positive_stopping_macro"] == {}
    folder = tmp_path / "active/condition/seed1"
    folder.mkdir(parents=True)
    expected = {level: dict(tpr=.5, actual_fpr=.01, mean_decisions=128., max_decisions=512.)
                for level in ("0.01", "0.05")}
    (folder / "REPORT.json").write_text(json.dumps({"positive_stopping": expected}))
    result = protocol_costs(tmp_path)
    assert result["joint_positive_stopping_macro"] == expected
    assert "serial_test_costs" not in result


def test_eagle_remote_head_receives_mask_needed_for_causality():
    class MaskDependentEagle(TinyEagle):
        # Like the checkpoint's remote implementation, causal masking is only
        # constructed when the caller supplies a 2-D attention mask.
        def forward(self, input_ids, hidden_states, attention_mask=None, **kwargs):
            if attention_mask is None:
                context = hidden_states.mean(1, keepdim=True).expand_as(hidden_states)
            else:
                assert torch.equal(attention_mask, torch.ones_like(input_ids))
                divisor = torch.arange(1, input_ids.shape[1] + 1).reshape(1, -1, 1)
                context = hidden_states.cumsum(1) / divisor
            self.norm(context)
    torch.manual_seed(0)
    adapter = FrozenAdapter(TinyTarget(), MaskDependentEagle(), 'eagle3', 'cpu')
    _, full = adapter.rows([0, 1, 2, 3])
    _, changed = adapter.rows([0, 1, 2, 0])
    _, prefix = adapter.next([0, 1, 2])
    torch.testing.assert_close(full[2], changed[2])
    torch.testing.assert_close(full[2], prefix)
