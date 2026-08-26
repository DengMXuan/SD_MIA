"""Run the P0 seed/epoch stability matrix and large-audit conditions."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "experiments" / "results" / "qwen3_sft" / "p0_matrix"
DATA_SEED = 20260824
AUDIT_SEED = 20260824
SEEDS = [20260824, 20260825, 20260826, 20260827, 20260828]
EPOCHS = [0, 1, 2, 4]


def run_condition(seed: int, epochs: int, n_per_class: int, label: str) -> Path:
    output = OUTPUT_ROOT / label / f"seed_{seed}" / f"epoch_{epochs}"
    output.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "experiments.sd_membership_sft.runner",
        "--gpu",
        "0",
        "--seed",
        str(seed),
        "--data-seed",
        str(DATA_SEED),
        "--audit-seed",
        str(AUDIT_SEED),
        "--target-epochs",
        str(epochs),
        "--n-per-class",
        str(n_per_class),
        "--n-aux",
        "160",
        "--audit-train-per-class",
        "48",
        "--response-tokens",
        "32",
        "--target-batch-size",
        "8",
        "--target-grad-accum",
        "1",
        "--draft-batch-size",
        "8",
        "--draft-grad-accum",
        "1",
        "--distill-steps",
        "0",
        "--bootstrap-repeats",
        "200",
        "--skip-trained-drafts",
        "--no-save-adapters",
        "--output-dir",
        str(output),
    ]
    print(f"[P0] seed={seed} epochs={epochs} n={n_per_class}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())
    return output


def load_result(path: Path) -> dict[str, Any]:
    return json.loads((path / "results.json").read_text(encoding="utf-8"))


def row(path: Path, seed: int, epochs: int, label: str) -> dict[str, Any]:
    result = load_result(path)
    metrics = result["metrics"]
    keys = [
        "control/model_less_hashed_bow",
        "base_draft/draft_min20_logp",
        "base_draft/honest_accept_rate_selected",
        "base_draft/joint_whitebox_transcript",
    ]
    output: dict[str, Any] = {
        "label": label,
        "seed": seed,
        "epochs": epochs,
        "n_per_class": result["config"]["n_per_class"],
        "audit_test_size": result["query_budget"]["audit_test_size"],
        "target_loss_first": (
            result["training"]["target_sft_loss"][0]
            if result["training"]["target_sft_loss"]
            else None
        ),
        "target_loss_last": (
            result["training"]["target_sft_loss"][-1]
            if result["training"]["target_sft_loss"]
            else None
        ),
    }
    for key in keys:
        output[key] = metrics[key]
    return output


def render_summary(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# P0 Seed/Epoch Stability Matrix",
        "",
        "## Material Passport",
        "",
        "- Status: COMPLETED",
        "- Data split seed: `20260824` for every condition",
        "- Audit calibration split seed: `20260824` for every condition",
        "- Training seeds: `20260824`–`20260828`",
        "- Epoch grid: `0, 1, 2, 4`",
        "- Main matrix: 320 records per class; large-audit conditions: 1,048 records per class",
        "- Draft condition: fixed Qwen3-1.7B base draft; trained draft variants disabled for P0",
        "",
        "## Main matrix",
        "",
        "| Condition | Seed | n/class | Target loss | BoW AUC | Draft Min-K AUC | Acceptance AUC | Joint AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in rows:
        target_loss = (
            "base"
            if item["target_loss_first"] is None
            else f"{item['target_loss_first']:.3f}→{item['target_loss_last']:.3f}"
        )
        lines.append(
            f"| `{item['label']}` | {item['seed']} | {item['n_per_class']} | {target_loss} | "
            f"{item['control/model_less_hashed_bow']['auc']:.3f} | "
            f"{item['base_draft/draft_min20_logp']['auc']:.3f} | "
            f"{item['base_draft/honest_accept_rate_selected']['auc']:.3f} | "
            f"{item['base_draft/joint_whitebox_transcript']['auc']:.3f} |"
        )

    grouped: dict[tuple[str, int], list[float]] = {}
    for item in rows:
        if item["n_per_class"] != 320:
            continue
        key = (item["label"], item["epochs"])
        grouped.setdefault(key, []).append(item["base_draft/joint_whitebox_transcript"]["auc"])
    lines.extend(["", "## Seed summary for the 320/class matrix", ""])
    lines.extend(
        [
            "| Epoch | Joint AUC mean | Joint AUC SD | Min | Max |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for epochs in EPOCHS:
        values = grouped[("matrix", epochs)]
        lines.append(
            f"| {epochs} | {mean(values):.3f} | "
            f"{(stdev(values) if len(values) > 1 else 0.0):.3f} | "
            f"{min(values):.3f} | {max(values):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The 320/class matrix estimates training-seed stability but is not sized for low-FPR claims.",
            "The 1,048/class conditions provide at least 1,000 held-out nonmembers after calibration and are the conditions to use for TPR@1%FPR.",
            "The verifier remains a local semantic simulation; passive L2 is reported separately in P1.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for epochs in EPOCHS:
            path = run_condition(seed, epochs, 320, "matrix")
            rows.append(row(path, seed, epochs, "matrix"))

    for epochs in (1, 4):
        path = run_condition(20260824, epochs, 1048, "large_audit")
        rows.append(row(path, 20260824, epochs, "large_audit"))

    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT_ROOT / "SUMMARY.md").write_text(render_summary(rows), encoding="utf-8")
    print(f"[P0] wrote {OUTPUT_ROOT / 'SUMMARY.md'}", flush=True)


if __name__ == "__main__":
    main()
