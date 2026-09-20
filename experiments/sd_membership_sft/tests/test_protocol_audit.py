from dataclasses import asdict
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from experiments.sd_membership_sft.sd_protocol import (
    FEATURE_NAMES, FrozenAdapter, RuntimeCost, draft_features, fixed_trace,
    map_eagle_logits, natural_trace, resolve_starts, trajectory_seed,
)
from experiments.sd_membership_sft.protocol_archive import (
    load_archive, save_archive, validate_arrays,
)
from experiments.sd_membership_sft.protocol_accept_only import (
    CausalAcceptanceGRU, document_scores, evidence_components, fit_causal,
    natural_features, trajectory_partitions,
)


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_start_positions_and_stable_streams():
    assert resolve_starts(1000, ["0.5", "0.75", "suffix64"]) == [500, 750, 936]
    assert resolve_starts(7, ["0.5", "0.75"]) == [3, 5]
    for starts in (["nan"], ["0"], ["1"], ["0.5", "0.51"], ["suffix64"], []):
        with pytest.raises(ValueError):
            resolve_starts(8, starts)
    assert trajectory_seed(7, "doc-a", "0.5") == trajectory_seed(7, "doc-a", "0.5")
    assert trajectory_seed(7, "doc-a", "0.5") != trajectory_seed(7, "doc-a", "0.75")


class ControlledAdapter:
    device = torch.device("cpu")

    def __init__(self, reject=False):
        self.reject = reject
        self.contexts = []
        self.cost = RuntimeCost()

    def next(self, context):
        self.contexts.append(list(context))
        self.cost.target_forward_calls += 1
        q = torch.tensor([.8, .2, 0.]).log()
        p = torch.tensor([0., 0., 1.]).log() if self.reject else q
        return p, q

    def target_next(self, context):
        self.contexts.append(list(context))
        return torch.tensor([0., 0., 1.]).log()

    def rows(self, tokens):
        q = torch.tensor([.8, .2, 0.]).log().repeat(len(tokens), 1)
        return q.clone(), q


def test_rejection_changes_next_prefix_and_no_tokens_escape():
    adapter = ControlledAdapter(reject=True)
    prefix = [1, 0]
    trace = natural_trace(adapter, prefix, rounds=3, seed=17, start_fraction=.5)
    assert prefix == [1, 0]
    assert adapter.contexts == [[1, 0], [1, 0, 2], [1, 0, 2, 2]]
    assert trace["counts"].tolist() == [0, 0, 0]
    assert trace["generated_tokens"] == 3
    assert trace["features"].shape == (3, len(FEATURE_NAMES))
    assert not {"tokens", "logp", "hidden_states", "delta"}.intersection(trace)


def test_acceptance_bonus_and_eos_boundaries():
    trace = natural_trace(ControlledAdapter(), [0, 1], rounds=4, seed=1, start_fraction=.75)
    assert trace["counts"].tolist() == [1] * 4
    assert trace["generated_tokens"] == 8
    for reject, reason in ((False, "bonus_eos"), (True, "correction_eos")):
        trace = natural_trace(ControlledAdapter(reject), [0, 1], rounds=4, seed=1,
                              start_fraction=.75, eos_ids=(2,))
        assert trace["rounds"] == 1
        assert trace["termination"] == reason
    trace = natural_trace(ControlledAdapter(), [0, 1], rounds=4, seed=1,
                          start_fraction=.75, eos_ids=(0, 1))
    assert trace["termination"] == "accepted_eos"
    assert trace["generated_tokens"] == 1


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


def archive_fixture(n=4, starts=2):
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
    contract = {"protocol": "natural", "starts": ["0.5", "0.75"][:starts],
                "rounds_per_start": 3, "sources": {}, "data_contract": "four_role_600"}
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


@pytest.mark.parametrize("mutation", ["logp", "duplicate", "counts", "labels", "budget", "nan"])
def test_archive_rejects_invalid_or_forbidden_data(mutation):
    data, contract = archive_fixture()
    if mutation == "logp":
        data["logp"] = np.zeros(len(data["features"]))
    elif mutation == "duplicate":
        data["start_indices"][1] = 0
    elif mutation == "counts":
        data["counts"][0] = 2
    elif mutation == "labels":
        data["labels"][0] = 1
    elif mutation == "budget":
        contract["rounds_per_start"] = 2
    else:
        data["features"][0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_arrays(data, contract)


def test_training_and_calibration_split_at_document_boundary():
    parts = {"train": np.array([0]), "validation": np.array([1]), "calibration": np.array([2]), "test": np.array([3])}
    owners = np.repeat(np.arange(4), 2)
    mapped = trajectory_partitions(parts, owners)
    assert mapped["train"].tolist() == [0, 1]
    assert mapped["calibration"].tolist() == [4, 5]
    assert not set(mapped["train"]).intersection(mapped["test"])


def test_causal_inputs_reset_each_start_and_never_see_future_outcomes():
    raw = np.zeros((8, 6), dtype=np.float32)
    counts = np.ones(8, dtype=np.uint8)
    features = natural_features(raw, counts, np.array([4, 4]))
    assert features[0, -1] == features[4, -1] == 0
    changed = counts.copy()
    changed[2] = 0
    np.testing.assert_array_equal(features[:3], natural_features(raw, changed, np.array([4, 4]))[:3])
    model = CausalAcceptanceGRU(7).eval()
    x = torch.from_numpy(features[:4])[None]
    with torch.no_grad():
        original = model(x, None)
        x[:, 3] = 100
        modified = model(x, None)
    torch.testing.assert_close(original[:, :3], modified[:, :3])


def test_fit_is_independent_of_test_and_calibration_features_and_outcomes():
    data, _ = archive_fixture(n=6)
    parts = trajectory_partitions({"train": [0, 1], "validation": [2], "calibration": [3], "test": [4, 5]}, data["document_indices"])
    x = natural_features(data["features"], data["counts"], data["lengths"])
    first = fit_causal(x, data["counts"], data["lengths"], parts, seed=17, epochs=2)
    x[18:] = 200
    counts = data["counts"].copy()
    counts[18:] = 1
    second = fit_causal(x, counts, data["lengths"], parts, seed=17, epochs=2)
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
    assert not np.allclose(combined["global_positive"], document_scores(data, logpmf, start_index=0)["global_positive"])
    order = np.arange(8)[::-1]
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
    from experiments.sd_membership_sft.collect_protocol_observations import protocol_prompt_ids
    from experiments.sd_membership_sft.data import SFTRecord

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [1, 2] if kwargs.get("return_dict") is False else {"input_ids": [1, 2]}
    record = SFTRecord("a", "test", (1, 2), "hash", prompt_text="continue")
    assert protocol_prompt_ids(record, Tokenizer()) == [1, 2]


def test_multistart_collection_resumes_and_detects_corruption(tmp_path):
    from experiments.sd_membership_sft.collect_protocol_observations import collect_records
    from experiments.sd_membership_sft.data import SFTRecord

    record = SFTRecord("doc", "test", tuple([0, 1] * 4), "hash", prompt_ids=(1,))
    prepared = SimpleNamespace(
        records=[record], tokenizer=SimpleNamespace(eos_token_id=None),
        record_ids=np.array(["doc"]), record_roles=np.array(["smoke"]), labels=np.array([0]),
    )
    contract = {"starts": ["0.5", "0.75"], "protocol": "natural", "seed": 9,
                "rounds_per_start": 2, "sources": {}, "data_contract": "runtime_smoke_no_membership_claim"}
    first = ControlledAdapter(reject=True)
    path = collect_records(prepared, first, tmp_path, contract)
    assert first.contexts[0] == [1, 0, 1, 0, 1]
    assert first.contexts[2] == [1, 0, 1, 0, 1, 0, 1]
    data, envelope = load_archive(path)
    assert data["document_indices"].tolist() == [0, 0]
    assert data["start_indices"].tolist() == [0, 1]
    second = ControlledAdapter()
    collect_records(prepared, second, tmp_path, contract)
    assert not second.contexts
    with pytest.raises(ValueError, match="configuration"):
        collect_records(prepared, second, tmp_path, {**contract, "rounds_per_start": 3})
    with (tmp_path / "trajectories/0_0.npz").open("ab") as stream:
        stream.write(b"bad")
    with pytest.raises(ValueError, match="hash mismatch"):
        collect_records(prepared, second, tmp_path, contract)


@pytest.mark.parametrize("protocol", ["natural", "fixed"])
def test_evaluation_pipeline_calibrates_documents_and_saves_detector(tmp_path, monkeypatch, protocol):
    import experiments.sd_membership_sft.protocol_accept_only as scoring

    starts = 2 if protocol == "natural" else 1
    data, contract = archive_fixture(n=10, starts=starts)
    # Small controlled fixture; production evaluate still requires 600+2000+2000.
    parts = {"train": np.array([0, 1, 2]), "validation": np.array([3, 4]),
             "reference": np.arange(5), "calibration": np.array([5, 6, 7]), "test": np.array([8, 9])}
    monkeypatch.setattr(scoring, "deployment_partitions", lambda *args: parts)
    contract.update(protocol=protocol, adapter="plain", head_real_model_validation="not_applicable",
                    execution="full_context_reconstruction")
    if protocol == "fixed":
        contract["starts"] = ["fixed"]
        data["counts"][::2] = 2
    else:
        data["counts"][::2] = 1
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
    assert saved["architecture"] == ("causal_gru_binary" if protocol == "natural" else "difficulty_tcn_count_b2")


def test_smoke_archive_cannot_be_used_as_membership_evaluation(tmp_path):
    from experiments.sd_membership_sft.protocol_accept_only import evaluate

    data, contract = archive_fixture()
    contract["data_contract"] = "runtime_smoke_no_membership_claim"
    path = tmp_path / "observations.npz"
    save_archive(path, data, contract, [{}] * len(data["lengths"]))
    with pytest.raises(ValueError, match="smoke"):
        evaluate(path, tmp_path / "evaluation")
