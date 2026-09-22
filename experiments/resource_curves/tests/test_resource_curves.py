from dataclasses import asdict
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.sd_membership_sft.core.audit_partitions import deployment_partitions
from experiments.sd_membership_sft.core.data_contract import ControlledDataContract
from experiments.sd_membership_sft.methods.difficulty_accept_only import fit as original_fit
from experiments.sd_membership_sft.protocols.sd_protocol import RuntimeCost, fixed_trace as original_trace
from experiments.resource_curves.auxiliary import select_extension, extension_records, save_extension
from experiments.resource_curves.config import AuxiliaryBudget, QUERY_MULTIPLICITIES, calibration_curve, fitting_curve
from experiments.resource_curves.detector import fit_detector, fitting_data, fitting_identity, FEATURE_COLUMNS
from experiments.resource_curves.evaluation import collection_cost, evaluate
from experiments.resource_curves.observations import Observations, fixed_trace, collect_observations, load_observations
from experiments.resource_curves.partitions import base_partitions, build_study, partition_indices, prepare_study
from experiments.resource_curves.storage import digest, workspace


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ToyAdapter:
    device = torch.device("cpu")
    kind = "plain"

    def __init__(self, unsupported=False):
        self.cost = RuntimeCost()
        self.unsupported = unsupported

    def rows(self, tokens):
        self.cost.target_forward_calls += 1
        self.cost.draft_forward_calls += 1
        self.cost.target_input_tokens += len(tokens)
        self.cost.draft_input_tokens += len(tokens)
        p = torch.tensor([.1, .3, .6]).log().repeat(len(tokens), 1)
        q = torch.tensor([.2, .4, .4]).log().repeat(len(tokens), 1)
        if self.unsupported:
            q[2, 0] = -torch.inf
        return p, q


@pytest.mark.parametrize("multiplicity", QUERY_MULTIPLICITIES)
def test_fresh_nested_bits_and_count_support(multiplicity):
    prompt, response = [1, 2], [0] * 96
    full = fixed_trace(ToyAdapter(), prompt, response, seed=72, multiplicity=16)
    trace = fixed_trace(ToyAdapter(), prompt, response, seed=72, multiplicity=multiplicity)
    assert np.array_equal(trace["bits"], full["bits"][:, :multiplicity])
    assert trace["acceptance_judgments"] == 96 * multiplicity
    assert trace["supported_candidates"] == 96
    # Different columns are fresh Bernoulli draws, never copied observations.
    assert np.any(full["bits"][:, 0] != full["bits"][:, 1])
    assert np.any(full["bits"][:, 0] != full["bits"][:, 2])
    assert np.max(full["bits"].sum(1)) > 2
    assert .40 < full["bits"].mean() < .60


def test_b2_exactly_matches_original_and_counts_supported_positions():
    args = ([1, 2], [0] * 8)
    new = fixed_trace(ToyAdapter(True), *args, seed=87, multiplicity=2)
    old = original_trace(ToyAdapter(True), *args, seed=87)
    np.testing.assert_array_equal(new["features"], old["features"])
    np.testing.assert_array_equal(new["bits"].sum(1), old["counts"])
    assert new["acceptance_judgments"] == 14
    assert new["candidate_positions"] == 8


@pytest.mark.parametrize("bad", [0, 3, 32, True, 2.0])
def test_invalid_multiplicity(bad):
    with pytest.raises(ValueError):
        fixed_trace(ToyAdapter(), [1], [0], seed=1, multiplicity=bad)


def shared_split():
    return {"splits": {
        "audit_auxiliary": [{"record_id": f"a{i}"} for i in range(600)],
        "member": [{"record_id": f"m{i}"} for i in range(2)],
        "nonmember": [{"record_id": f"n{i}"} for i in range(2)],
        "auxiliary": [{"record_id": f"d{i}"} for i in range(2)],
    }}


def extension(shared, count=1000):
    return {"shared_split_digest": digest(shared), "records": [{"record_id": f"e{i}"} for i in range(count)]}


def test_default_partitions_match_old_and_requested_curves_are_nested():
    shared = shared_split()
    base = base_partitions(shared)
    ids = np.asarray([f"a{i}" for i in range(600)] + ["m0", "m1", "n0", "n1"])
    roles = np.asarray(["audit_auxiliary"] * 600 + ["member"] * 2 + ["nonmember"] * 2)
    old = deployment_partitions((roles == "member").astype(int), ids, roles,
                                ControlledDataContract(members=2, nonmembers=2))
    for role in ("train", "validation", "calibration", "test"):
        assert base[role] == ids[old[role]].tolist()
    studies = [build_study(shared, extension(shared), budgets, name=name)
               for name, budgets in (("calibration", calibration_curve()), ("fitting", fitting_curve()))]
    cal, fitting = [study["points"] for study in studies]
    assert [p["budget"]["total"] for p in cal] == [600, 1000, 1600]
    assert [p["budget"]["total"] for p in fitting] == [600, 1000, 1400]
    assert cal[0] == fitting[0]
    for previous, current in zip(cal, cal[1:]):
        assert current["partitions"]["train"] == previous["partitions"]["train"]
        assert current["partitions"]["validation"] == previous["partitions"]["validation"]
        assert current["partitions"]["calibration"][:len(previous["partitions"]["calibration"])] == previous["partitions"]["calibration"]
    for previous, current in zip(fitting, fitting[1:]):
        assert current["partitions"]["calibration"] == previous["partitions"]["calibration"]
        for role in ("train", "validation"):
            assert current["partitions"][role][:len(previous["partitions"][role])] == previous["partitions"][role]
    for point in cal + fitting:
        parts = point["partitions"]
        all_ids = sum(parts.values(), [])
        assert len(set(all_ids)) == len(all_ids)
        assert parts["test"] == base["test"]


def test_study_fails_on_insufficient_or_leaking_extensions():
    shared = shared_split()
    with pytest.raises(ValueError, match="need 1000"):
        build_study(shared, extension(shared, 999), calibration_curve(), name="cal")
    contaminated = extension(shared)
    contaminated["records"][0]["record_id"] = "d0"
    with pytest.raises(ValueError, match="overlap"):
        build_study(shared, contaminated, fitting_curve(), name="fit")
    with pytest.raises(ValueError, match="separate"):
        build_study(shared, extension(shared), (AuxiliaryBudget(400, 200), AuxiliaryBudget(800, 600)), name="mixed")


@pytest.mark.parametrize("args", [(0, 200), (100, 20), (1200, 1200), (400, 0), (400, 200, 400)])
def test_bad_auxiliary_allocations(args):
    with pytest.raises(ValueError):
        AuxiliaryBudget(*args)


def synthetic_observations(multiplicity=2):
    rng = np.random.default_rng(51)
    ids = np.asarray([f"a{i}" for i in range(600)] + ["m0", "m1", "n0", "n1"])
    roles = np.asarray(["audit_auxiliary"] * 600 + ["member"] * 2 + ["nonmember"] * 2)
    lengths = np.full(len(ids), 3, np.int64)
    features = rng.normal(size=(int(lengths.sum()), 6)).astype(np.float32)
    features[:, 0] = -abs(features[:, 0])
    arrays = dict(features=features, bits=rng.integers(0, 2, size=(len(features), multiplicity), dtype=np.uint8),
                  lengths=lengths, record_ids=ids, record_roles=roles, labels=(roles == "member").astype(int),
                  candidate_positions=lengths.copy())
    contract = {"multiplicity": multiplicity, "adapter": "plain", "seed": 51, "sources": {"synthetic": True},
                "sampling": "nested_pair_streams_v1", "runtime_sha256": "test", "hardware": {"device": "cpu"}}
    costs = [{**asdict(RuntimeCost(target_forward_calls=1, draft_forward_calls=1, target_input_tokens=4,
                                  draft_input_tokens=4)), "record_id": str(record_id), "seconds": .1,
              "acceptance_judgments": 3 * multiplicity} for record_id in ids]
    return Observations(arrays, contract, costs, multiplicity)


def small_points():
    shared = shared_split()
    return build_study(shared, extension(shared, 0), (AuxiliaryBudget(160, 40), AuxiliaryBudget(160, 80)), name="cal")["points"]


@pytest.mark.parametrize("multiplicity", QUERY_MULTIPLICITIES)
def test_train_predict_evaluate_variable_support_and_resume(tmp_path, multiplicity):
    obs = synthetic_observations(multiplicity)
    first, larger_cal = small_points()
    fit = fit_detector(obs, first, tmp_path / "fits", epochs=1)
    assert fit.model.k == multiplicity
    assert len(fit.metadata["history"]) == 1
    again = fit_detector(obs, larger_cal, tmp_path / "fits", epochs=1)
    assert again.reused and again.checkpoint == fit.checkpoint
    report = evaluate(obs, first, again, tmp_path / "result", bootstrap=0)
    assert 0 <= report["metrics"]["auc"] <= 1
    assert report["cost"]["acceptance_judgments"] == (200 + 4) * 3 * multiplicity
    assert report["cost"]["collection_seconds"] > 0
    assert report == evaluate(obs, first, again, tmp_path / "result", bootstrap=0)
    with pytest.raises(ValueError, match="resume parameters"):
        evaluate(obs, larger_cal, again, tmp_path / "result", bootstrap=0)


def test_b2_detector_unchanged_and_calibration_cannot_affect_fit(tmp_path):
    obs = synthetic_observations()
    first, larger = small_points()
    fit = fit_detector(obs, first, tmp_path / "fit", seed=81, epochs=1)
    sub, parts = fitting_data(obs, first)
    old, mean, scale, _, _ = original_fit(sub["features"][:, FEATURE_COLUMNS], sub["counts"], sub["lengths"],
                                        parts, seed=81, device="cpu", epochs=1)
    np.testing.assert_array_equal(mean, fit.mean)
    np.testing.assert_array_equal(scale, fit.scale)
    for key, tensor in old.state_dict().items():
        torch.testing.assert_close(tensor, fit.model.state_dict()[key], rtol=0, atol=0)
    identity = fitting_identity(obs, first)
    assert identity == fitting_identity(obs, larger)
    data = obs.count_data()
    indices = partition_indices(data, larger)
    obs.arrays["bits"][np.repeat(np.isin(np.arange(len(data["lengths"])), indices["calibration"]), 3)] ^= 1
    assert fitting_identity(obs, first) == identity
    again = fit_detector(obs, larger, tmp_path / "fit", seed=81, epochs=1)
    assert again.reused
    r1 = evaluate(obs, first, fit, tmp_path / "r1", bootstrap=0)
    r2 = evaluate(obs, larger, fit, tmp_path / "r2", bootstrap=0)
    # Calibration alone changes deployment thresholds, not raw test ranking.
    for key in ("auc", "pauc_10_raw", "roc_tpr_at_10pct_fpr", "roc_tpr_at_1pct_fpr"):
        assert r1["metrics"][key] == r2["metrics"][key]
    obs.arrays["bits"][3 * indices["train"][0], 0] ^= 1
    assert fitting_identity(obs, first) != identity
    with pytest.raises(ValueError, match="different training"):
        evaluate(obs, first, fit, tmp_path / "bad", bootstrap=0)


def test_higher_budget_view_has_no_invented_runtime_or_bits():
    obs = synthetic_observations(16)
    point = small_points()[0]
    view = obs.view(2)
    parts = partition_indices(view.count_data(), point)
    cost = collection_cost(view, parts)
    assert cost["collection_seconds"] is None
    assert cost["origin_collection_seconds"] > 0
    assert cost["origin_counters"]["acceptance_judgments"] == cost["acceptance_judgments"] * 8
    with pytest.raises(ValueError, match="not recorded"):
        view.view(4)


def test_fitting_growth_changes_detector_identity_and_keeps_calibration(tmp_path):
    obs = synthetic_observations()
    shared = shared_split()
    study = build_study(shared, extension(shared, 0),
                        (AuxiliaryBudget(160, 40), AuxiliaryBudget(200, 40)), name="fitting")
    small, large = study["points"]
    first = fit_detector(obs, small, tmp_path / "fits", epochs=1)
    second = fit_detector(obs, large, tmp_path / "fits", epochs=1)
    assert not second.reused and second.checkpoint != first.checkpoint
    assert small["partitions"]["calibration"] == large["partitions"]["calibration"]
    with pytest.raises(ValueError, match="different training"):
        evaluate(obs, large, first, tmp_path / "wrong-fit", bootstrap=0)


def test_prepare_study_uses_only_needed_audited_extension_records():
    shared = shared_split()
    extra_manifest = extension(shared)
    study = build_study(shared, extra_manifest, fitting_curve(), name="fitting")
    obs = synthetic_observations()
    base = SimpleNamespace(records=[SimpleNamespace(record_id=value) for value in obs.arrays["record_ids"]],
                           tokenizer=object(), record_ids=obs.arrays["record_ids"],
                           record_roles=obs.arrays["record_roles"], labels=obs.arrays["labels"])
    extra = [SimpleNamespace(record_id=row["record_id"]) for row in extra_manifest["records"]]
    prepared = prepare_study(base, extra, extra_manifest, study)
    assert len(prepared.records) == 1404
    assert len(base.records) == 604  # Existing prepared input remains untouched.
    assert not set(prepared.record_ids).intersection({f"e{i}" for i in range(800, 1000)})
    with pytest.raises(ValueError, match="identities"):
        prepare_study(base, extra[:-1], extra_manifest, study)


def test_partition_leakage_rejected():
    obs = synthetic_observations()
    point = small_points()[0]
    point["partitions"]["train"][0] = "m0"
    with pytest.raises(ValueError):
        partition_indices(obs.count_data(), point)


def test_collection_recovers_completed_records_and_checks_contract(tmp_path):
    tokenizer = SimpleNamespace(apply_chat_template=lambda *a, **k: [1, 2])
    records = [SimpleNamespace(record_id=f"a{i}", response_ids=(0, 0, 0), prompt="p", prompt_ids=(1, 2)) for i in range(3)]
    prepared = SimpleNamespace(records=records, tokenizer=tokenizer, record_ids=np.array([f"a{i}" for i in range(3)]),
                               record_roles=np.array(["audit_auxiliary"] * 3), labels=np.zeros(3, int))
    class Interrupted(ToyAdapter):
        def rows(self, tokens):
            if self.cost.target_forward_calls == 1:
                raise RuntimeError("simulated interruption")
            return super().rows(tokens)
    with pytest.raises(RuntimeError, match="interruption"):
        collect_observations(prepared, Interrupted(), tmp_path / "obs", sources={"synthetic": True}, multiplicity=4)
    adapter = ToyAdapter()
    obs = collect_observations(prepared, adapter, tmp_path / "obs", sources={"synthetic": True}, multiplicity=4)
    assert adapter.cost.target_forward_calls == 2  # First completed record was reused.
    loaded = load_observations(tmp_path / "obs")
    assert obs.signature == loaded.signature
    collect_observations(prepared, adapter, tmp_path / "obs", sources={"synthetic": True}, multiplicity=4)
    assert adapter.cost.target_forward_calls == 2
    with pytest.raises(ValueError, match="resume parameters"):
        collect_observations(prepared, adapter, tmp_path / "obs", sources={"synthetic": True}, multiplicity=8)
    with (tmp_path / "obs/observations.npz").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_observations(tmp_path / "obs")


def test_workspace_cannot_adopt_active_or_unrelated_outputs(tmp_path):
    folder = tmp_path / "active"
    folder.mkdir()
    (folder / "REPORT.json").write_text("do not touch")
    with pytest.raises(ValueError, match="unrelated"):
        with workspace(folder):
            pass
    assert (folder / "REPORT.json").read_text() == "do not touch"
    with pytest.raises(ValueError, match="dedicated"):
        with workspace("artifacts/runs/audits/qwen_fixed_v1"):
            pass


def test_extension_excludes_all_assignments_and_rechecks_quality(tmp_path):
    def doc(record_id, tokens):
        text = " ".join(map(str, tokens))
        return {"record_id": record_id, "text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
    assigned = [doc(f"a{i}", range(100 * i, 100 * i + 32)) for i in range(4)]
    candidates = [doc("raw_duplicate", range(32)), doc("near_duplicate", [9000] + list(range(1, 32))),
                  doc("short", [2, 3]), doc("good1", range(500, 532)), doc("good2", range(600, 632))]
    # Different raw bytes but identical tokenizer output must also be excluded.
    token_dup = {"record_id": "token_duplicate", "text": "  ".join(map(str, range(100, 132)))}
    candidates.append(token_dup)
    pool = tmp_path / "pool.jsonl"
    payload = "".join(json.dumps(row) + "\n" for row in assigned + candidates).encode()
    pool.write_bytes(payload)
    pool_sha = hashlib.sha256(payload).hexdigest()
    pool.with_suffix(".manifest.json").write_text(json.dumps({"benchmark": "toy", "jsonl_sha256": pool_sha,
                                                            "records": len(assigned + candidates)}))
    split = {"schema_version": 3, "benchmark": "toy", "pool_sha256": pool_sha,
             "tokenizer_sources": ["toy"], "token_band": {"min_tokens": 16, "max_tokens": 32},
             "splits": {name: [{"record_id": row["record_id"], "text_sha256": row["text_sha256"]}]
                        for name, row in zip(("member", "nonmember", "auxiliary", "audit_auxiliary"), assigned)}}
    shared_path = tmp_path / "shared.json"
    shared_path.write_text(json.dumps(split))
    def tokenize(text, **kwargs):
        try:
            tokens = [int(t) for t in text.split()]
        except ValueError:
            tokens = [1, 2]
        return SimpleNamespace(input_ids=tokens[:kwargs.get("max_length", len(tokens))])
    before = {path: path.read_bytes() for path in (pool, pool.with_suffix(".manifest.json"), shared_path)}
    manifest = select_extension(pool, shared_path, {"toy": tokenize}, count=2)
    assert {row["record_id"] for row in manifest["records"]} == {"good1", "good2"}
    assert manifest == select_extension(pool, shared_path, {"toy": tokenize}, count=2)
    assert len(extension_records(manifest, tokenize, "toy")) == 2
    saved = save_extension(tmp_path / "extension", manifest)
    assert json.loads(saved.read_text()) == manifest
    with pytest.raises(ValueError, match="insufficient"):
        select_extension(pool, shared_path, {"toy": tokenize}, count=3)
    with pytest.raises(ValueError, match="all tokenizer"):
        select_extension(pool, shared_path, {}, count=1)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not list(tmp_path.glob(".*lock*"))  # No shared pool lock or mutation.
