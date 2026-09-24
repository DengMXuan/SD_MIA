"""Pair DP audit scores with existing non-DP results on identical held-out records."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.audit.artifacts import read_result

FIELDS = ("auc", "pauc_10_normalized", "roc_tpr_at_1pct_fpr", "roc_tpr_at_10pct_fpr",
          "calibrated_tpr_at_1pct", "calibrated_actual_fpr_at_1pct",
          "calibrated_tpr_at_10pct", "calibrated_actual_fpr_at_10pct")


def _condition(report):
    condition = dict(report["condition"])
    pair = condition.pop("model_pair", condition.pop("pair", "qwen3"))
    return pair, condition


def _draft_role(report):
    role = report.get("draft_role")
    return {"auxiliary_head": "draft_auxiliary_distilled", "member_head": "draft_member_sft"}.get(role, role)


def compare_reports(dp_path, reference_path):
    dp_path, reference_path = Path(dp_path), Path(reference_path)
    dp, reference = read_result(dp_path.parent), read_result(reference_path.parent)
    if _condition(dp) != _condition(reference) or dp["method"] != reference["method"]:
        raise ValueError("comparison condition or method mismatch")
    if _draft_role(dp) != _draft_role(reference) or dp["settings"] != reference["settings"]:
        raise ValueError("comparison draft or audit settings mismatch")
    if "privacy" not in dp or "privacy" in reference:
        raise ValueError("comparison requires a DP report and an undefended reference")
    with np.load(dp_path.parent / "scores.npz") as a, np.load(reference_path.parent / "scores.npz") as b:
        for name in ("record_ids", "labels", "calibration", "test"):
            if not np.array_equal(a[name], b[name]):
                raise ValueError(f"comparison {name} mismatch")
    pair, condition = _condition(dp)
    row = {**condition, "model_pair": pair,
           "draft_role": dp.get("draft_role", "target_only"), "method": dp["method"],
           "target_epsilon_cap": dp["privacy"]["target_epsilon_cap"],
           "pair_epsilon": dp["privacy"]["epsilon"], "pair_delta": dp["privacy"]["delta"],
           "dp_report": str(dp_path), "reference_report": str(reference_path)}
    for field in FIELDS:
        row["reference_" + field] = reference["metrics"][field]
        row["dp_" + field] = dp["metrics"][field]
        row["change_" + field] = dp["metrics"][field] - reference["metrics"][field]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp-root", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True, help="existing Qwen audit matrix output root")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, errors = [], []
    for path in sorted(args.dp_root.rglob("REPORT.json")):
        report = json.loads(path.read_text())
        if "privacy" not in report:
            continue
        condition = report["condition"]
        root = args.reference_root
        if "model_pair" in condition or "pair" in condition:
            root = root / _condition(report)[0]
        root = root / condition["benchmark"] / f"epoch{condition['epoch']}" / f"seed{condition['condition_seed']}"
        subdir = Path(report["draft_role"]) / "fixed" if "draft_role" in report else Path("baseline")
        reference = root / subdir / report["method"] / "REPORT.json"
        try:
            rows.append(compare_reports(path, reference))
        except (OSError, ValueError, KeyError) as error:
            errors.append({"report": str(path), "error": str(error)})
    groups = {}
    for row in rows:
        key = (row["model_pair"], row["benchmark"], row["epoch"], row["draft_role"], row["method"], row["target_epsilon_cap"])
        group = groups.setdefault(key, [])
        if any(item["condition_seed"] == row["condition_seed"] for item in group):
            raise ValueError("duplicate condition seed in DP comparison")
        group.append(row)
    aggregates = []
    for key, values in groups.items():
        item = dict(zip(("model_pair", "benchmark", "epoch", "draft_role", "method", "target_epsilon_cap"), key))
        item["completed_seeds"] = len(values)
        for field in FIELDS:
            changes = [row["change_" + field] for row in values]
            item["change_" + field + "_mean"] = float(np.mean(changes))
            item["change_" + field + "_std"] = float(np.std(changes, ddof=1)) if len(changes) > 1 else None
        aggregates.append(item)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in (("COMPARISON", rows), ("SEED_SUMMARY", aggregates)):
        with (args.output_dir / f"{name}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]) if table else [])
            writer.writeheader()
            writer.writerows(table)
    _write_json(args.output_dir / "COMPARISON.json", {"rows": rows, "seed_summary": aggregates, "errors": errors,
                "scope": "available matched reports; not a matrix completeness assertion",
                "interpretation": "negative AUC/TPR changes indicate a weaker attack; inspect actual FPR and utility separately"})
    print(json.dumps({"matched_reports": len(rows), "errors": errors}))
    if errors or not rows:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
