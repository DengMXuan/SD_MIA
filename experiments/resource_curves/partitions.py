"""Stable, nested detector allocations with a frozen evaluation set."""
import numpy as np
from types import SimpleNamespace

from experiments.shared.core.audit_runtime import SPLIT_SEED
from experiments.resource_curves.config import AuxiliaryBudget
from experiments.resource_curves.storage import checked_contract, digest, workspace

ROLES = ("train", "validation", "calibration", "test")


def base_partitions(shared):
    """Reproduce the current 320/80/200 split in original auxiliary order."""
    splits = shared["splits"]
    auxiliary = [row["record_id"] for row in splits["audit_auxiliary"]]
    if len(auxiliary) != 600:
        raise ValueError("expected the frozen 600-record audit auxiliary pool")
    shuffled = np.random.default_rng(SPLIT_SEED).permutation(600)
    return {
        "train": [auxiliary[i] for i in sorted(shuffled[:320])],
        "validation": [auxiliary[i] for i in sorted(shuffled[320:400])],
        "calibration": [auxiliary[i] for i in sorted(shuffled[400:])],
        "test": [row["record_id"] for name in ("member", "nonmember") for row in splits[name]],
    }


def build_study(shared, extension, budgets, *, name):
    """Allocate maximal role reservoirs once; every point takes nested prefixes.

    Build the calibration and fitting studies separately. Extension IDs may be
    reused across studies but never across roles within a study.
    """
    budgets = tuple(budgets)
    if not budgets or not all(isinstance(b, AuxiliaryBudget) for b in budgets):
        raise ValueError("a nonempty sequence of AuxiliaryBudget values is required")
    if (len({(b.train, b.validation) for b in budgets}) > 1
            and len({b.calibration for b in budgets}) > 1):
        raise ValueError("build fitting and calibration curves as separate studies")
    base = base_partitions(shared)
    if extension["shared_split_digest"] != digest(shared):
        raise ValueError("extension belongs to a different frozen split")
    extension_ids = [row["record_id"] for row in extension["records"]]
    excluded = {row["record_id"] for rows in shared["splits"].values() for row in rows}
    if len(set(extension_ids)) != len(extension_ids) or excluded.intersection(extension_ids):
        raise ValueError("extension IDs repeat or overlap existing model/audit assignments")
    maxima = {"train": max(b.train for b in budgets),
              "validation": max(b.validation for b in budgets),
              "calibration": max(b.calibration for b in budgets)}
    required = sum(max(0, size - len(base[role])) for role, size in maxima.items())
    if required > len(extension_ids):
        raise ValueError(f"need {required} eligible extension records, have {len(extension_ids)}")
    offset = 0
    pools = {"test": base["test"]}
    for role, size in maxima.items():
        extra = max(0, size - len(base[role]))
        pools[role] = base[role] + extension_ids[offset:offset + extra]
        offset += extra
    points = []
    for budget in budgets:
        parts = {role: pools[role][:getattr(budget, role)]
                 for role in ("train", "validation", "calibration")}
        parts["test"] = pools["test"]
        all_ids = [record for role in ROLES for record in parts[role]]
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("study roles are not disjoint")
        points.append({"budget": budget.to_dict(), "partitions": parts})
    return {"schema": "resource_study_v1", "name": name,
            "shared_split_digest": digest(shared), "extension_digest": digest(extension),
            "points": points}


def partition_indices(data, point):
    ids = data["record_ids"].astype(str).tolist()
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate observation IDs")
    lookup = {value: i for i, value in enumerate(ids)}
    try:
        parts = {role: np.asarray([lookup[value] for value in point["partitions"][role]], dtype=np.int64)
                 for role in ROLES}
    except KeyError as error:
        raise ValueError(f"missing partition/observation: {error}") from error
    combined = np.concatenate(list(parts.values()))
    if any(not len(parts[role]) for role in ROLES) or len(np.unique(combined)) != len(combined):
        raise ValueError("empty or overlapping partitions")
    for role in ("train", "validation", "calibration"):
        selected = parts[role]
        if ((data["labels"][selected] != 0).any()
                or (data["record_roles"][selected] != "audit_auxiliary").any()):
            raise ValueError("detector fitting/calibration requires trusted audit nonmembers")
        if len(selected) != point["budget"][role]:
            raise ValueError("allocation does not match declared size")
    if set(data["record_roles"][parts["test"]].tolist()) != {"member", "nonmember"}:
        raise ValueError("test must contain both frozen test roles")
    return parts


def save_study(folder, study):
    with workspace(folder) as output:
        checked_contract(output, "STUDY.json", study)
    return output / "STUDY.json"


def prepare_study(base_prepared, extra_records, extension, study):
    """Select a study's union of records without altering the existing prepared data.

    ``extra_records`` comes from auxiliary.extension_records; existing prepared
    records/tokenizer are obtained using the current model-specific loader.
    """
    if study["extension_digest"] != digest(extension):
        raise ValueError("study and auxiliary extension manifest disagree")
    allowed_extra = {row["record_id"] for row in extension["records"]}
    if {record.record_id for record in extra_records} != allowed_extra or len(extra_records) != len(allowed_extra):
        raise ValueError("extension record identities differ from the audited manifest")
    base_ids = [record.record_id for record in base_prepared.records]
    if base_ids != base_prepared.record_ids.tolist() or len(set(base_ids)) != len(base_ids):
        raise ValueError("base records and IDs differ")
    if allowed_extra.intersection(base_ids):
        raise ValueError("extension overlaps existing prepared records")
    if (len(base_prepared.record_roles) != len(base_ids)
            or not np.array_equal(base_prepared.labels, (base_prepared.record_roles == "member").astype(int))):
        raise ValueError("base roles and labels differ")
    rows = list(base_prepared.records) + list(extra_records)
    roles = list(base_prepared.record_roles) + ["audit_auxiliary"] * len(extra_records)
    wanted = {record_id for point in study["points"] for ids in point["partitions"].values() for record_id in ids}
    if wanted - {record.record_id for record in rows}:
        raise ValueError("study requires records absent from the prepared pool")
    selected = [(record, role) for record, role in zip(rows, roles) if record.record_id in wanted]
    records, selected_roles = zip(*selected)
    result = SimpleNamespace(records=list(records), tokenizer=base_prepared.tokenizer,
                             record_ids=np.asarray([record.record_id for record in records]),
                             record_roles=np.asarray(selected_roles),
                             labels=np.asarray([int(role == "member") for role in selected_roles]))
    for point in study["points"]:
        partition_indices({"record_ids": result.record_ids, "record_roles": result.record_roles,
                           "labels": result.labels}, point)
    return result
