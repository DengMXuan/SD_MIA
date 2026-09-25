"""Fixed-candidate audit coordinator; preserved legacy task identities are migrated explicitly."""
from experiments.shared.audit.config import audit_settings
import argparse
import json
from pathlib import Path

from experiments.paths import audit_reports, audit_executions
from experiments.sd_membership_sft.audit import qwen_audit_matrix as matrix


from experiments.shared.audit.devices import check_gpus


def fixed_tasks(*args):
    # Preserve the established public helper and task identities.
    return matrix.make_tasks(*args)


def summarize_fixed(tasks, output_root):
    # Preserve existing whole-matrix summaries and all per-method artifacts.
    destination = audit_reports(output_root)
    result = matrix.summarize(tasks, destination,
                              methods=(*matrix.MAIN_METHODS["fixed"], *matrix.METHODS),
                              execution_root=audit_executions(output_root))
    result["scope"] = "fixed_candidate_and_baselines_only"
    matrix._write_json(destination / "SUMMARY.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Qwen fixed-candidate + 11 baselines; resume checked outputs.")
    parser.add_argument("command", choices=("dry-run", "run", "status", "summarize"))
    parser.add_argument("--model-root", type=Path, default=matrix.DEFAULT_MODEL_ROOT)
    parser.add_argument("--output-root", type=Path, default=matrix.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benchmarks", nargs="+", choices=("wikitection", "newstection", "arxivtection"), default=["wikitection", "newstection", "arxivtection"])
    parser.add_argument("--epochs", nargs="+", type=int, choices=(1, 3), default=[1, 3])
    parser.add_argument("--seeds", nargs="+", type=int, choices=(1919, 1949, 1978), default=[1919, 1949, 1978])
    parser.add_argument("--gpus", nargs="+", type=int, default=[0])
    # Retained exactly for compatibility with pre-existing task signatures.
    parser.add_argument("--starts", nargs="+", default=["suffix64"], help="legacy request identity only; does not affect observations")
    parser.add_argument("--rounds-per-start", type=int, default=32, help="legacy request identity only; does not affect observations")
    parser.add_argument("--audit-seed", type=int, default=None, help="must match the condition seed; defaults to it")
    parser.add_argument("--detector-epochs", type=int, default=30)
    args = parser.parse_args()
    if args.rounds_per_start < 1 or args.detector_epochs < 1:
        parser.error("round and detector epoch budgets must be positive")
    if args.output_root.resolve().is_relative_to(matrix.LEGACY_OUTPUT_ROOT.resolve()):
        parser.error("the historical audit output root is read-only; choose a separate --output-root")
    for values in (args.benchmarks, args.epochs, args.seeds, args.starts):
        if len(set(values)) != len(values):
            parser.error("duplicate matrix entries are not allowed")
    settings = audit_settings(audit_seed=args.audit_seed, detector_epochs=args.detector_epochs, starts=args.starts, rounds_per_start=args.rounds_per_start)
    tasks = fixed_tasks(args.model_root.resolve(), args.output_root.resolve(), args.benchmarks,
                        args.epochs, args.seeds, settings)
    if args.command in ("dry-run", "status"):
        conditions = sum(task["kind"] == "baseline" for task in tasks)
        print(json.dumps(dict(scope="fixed_candidate_and_baselines_only",
                              audit_configurations=conditions * len(matrix.ROLES), worker_tasks=len(tasks),
                              expected_method_rows=conditions * len(matrix.ROLES) * (len(matrix.METHODS) + 1),
                              summary_directory=str(audit_reports(args.output_root)),
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
