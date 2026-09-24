"""Fixed-candidate audit coordinator; preserved legacy task identities are migrated explicitly."""
import argparse
import json
import os
from pathlib import Path
import subprocess

from experiments.sd_membership_sft.audit import qwen_audit_matrix as matrix


def check_gpus(gpus):
    if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        raise ValueError("choose distinct nonnegative physical GPUs")
    limit = int(os.environ.get("SD_AUDIT_GPU_MAX_USED_MIB", "1024"))
    if limit < 0:
        raise ValueError("SD_AUDIT_GPU_MAX_USED_MIB must be nonnegative")
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    used = {}
    for line in inventory.splitlines():
        if line.strip():
            index, memory = line.split(",")
            used[int(index)] = int(memory.strip())
    for gpu in gpus:
        if gpu not in used:
            raise RuntimeError(f"GPU {gpu} is absent")
        if used[gpu] > limit:
            raise RuntimeError(
                f"GPU {gpu} uses {used[gpu]} MiB, above the background allowance of {limit} MiB "
                "(SD_AUDIT_GPU_MAX_USED_MIB)"
            )


def fixed_tasks(*args):
    # Filter whole original task dictionaries: do not alter cache identities.
    return [task for task in matrix.make_tasks(*args) if task.get("protocol") != "natural"]


def summarize_fixed(tasks, output_root):
    # Preserve existing whole-matrix summaries and all per-method artifacts.
    destination = output_root / "fixed_only_summary"
    original_methods = matrix.ALL_METHODS
    matrix.ALL_METHODS = (*matrix.MAIN_METHODS["fixed"], *matrix.METHODS)
    try:
        result = matrix.summarize(tasks, destination)
    finally:
        matrix.ALL_METHODS = original_methods
    selected = {task["id"] for task in tasks}
    attempts = [json.loads(path.read_text()) for path in
                (output_root / "executions").glob("*/STATUS.json")]
    result["attempted_worker_wall_seconds_sum"] = sum(
        row.get("worker_wall_seconds", 0.) for row in attempts if row.get("task") in selected)
    result["scope"] = "fixed_candidate_and_baselines_only; natural SD artifacts excluded and preserved"
    matrix._write_json(destination / "SUMMARY.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Qwen fixed-candidate + 11 baselines; resume original outputs, skip natural SD.")
    parser.add_argument("command", choices=("dry-run", "run", "status", "summarize"))
    parser.add_argument("--model-root", type=Path, default=matrix.DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=matrix.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benchmarks", nargs="+", choices=("wikitection", "newstection", "arxivtection"), default=["wikitection", "newstection", "arxivtection"])
    parser.add_argument("--epochs", nargs="+", type=int, choices=(1, 3), default=[1, 3])
    parser.add_argument("--seeds", nargs="+", type=int, choices=(1919, 1949, 1978), default=[1919, 1949, 1978])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    # Retained exactly for compatibility with pre-existing task signatures.
    parser.add_argument("--starts", nargs="+", default=["suffix64"], help="legacy cache identity; no natural SD is scheduled")
    parser.add_argument("--rounds-per-start", type=int, default=32, help="legacy cache identity; no natural SD is scheduled")
    parser.add_argument("--audit-seed", type=int, default=20260914)
    parser.add_argument("--detector-epochs", type=int, default=30)
    args = parser.parse_args()
    if args.rounds_per_start < 1 or args.detector_epochs < 1:
        parser.error("round and detector epoch budgets must be positive")
    if args.output_root.resolve().is_relative_to(matrix.LEGACY_OUTPUT_ROOT.resolve()):
        parser.error("the historical audit output root is read-only; choose a separate --output-root")
    for values in (args.benchmarks, args.epochs, args.seeds, args.starts):
        if len(set(values)) != len(values):
            parser.error("duplicate matrix entries are not allowed")
    from experiments.sd_membership_sft.protocols.sd_protocol import resolve_starts
    resolve_starts(2048, args.starts)
    settings = dict(starts=args.starts, rounds_per_start=args.rounds_per_start, audit_seed=args.audit_seed,
                    detector_epochs=args.detector_epochs, baseline=matrix.BASELINE_DEFAULTS,
                    baseline_execution="shared_robustness_reference_v1")
    tasks = fixed_tasks(args.model_root.resolve(), args.output_root.resolve(), args.benchmarks,
                        args.epochs, args.seeds, settings)
    if args.command in ("dry-run", "status"):
        conditions = sum(task["kind"] == "baseline" for task in tasks)
        print(json.dumps(dict(scope="fixed_candidate_and_baselines_only",
                              audit_configurations=conditions * len(matrix.ROLES), worker_tasks=len(tasks),
                              expected_method_rows=conditions * len(matrix.ROLES) * (len(matrix.METHODS) + 1),
                              summary_directory=str(args.output_root.resolve() / "fixed_only_summary"),
                              settings=settings, states={task["id"]: matrix.inspect_task(task) for task in tasks}), indent=2))
        return
    # Both initial preflight and between-task dispatch use the relaxed check.
    # Worker commands, task identities and experiment source hashes stay intact.
    matrix.check_gpus = check_gpus
    completed = matrix.run_tasks(tasks, args.output_root, args.gpus) if args.command == "run" else True
    result = summarize_fixed(tasks, args.output_root)
    print(json.dumps({key: result[key] for key in ("complete", "expected_rows", "completed_rows", "errors")}))
    if not completed or not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
