"""Run the frozen Qwen3 and Gemma 4 Wiki temporal proxy comparison."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from experiments.paths import ROOT
from experiments.pretraining.data import sha256
from experiments.shared.core import gpu_pool
from standalone.qwen_gemma_temporal import prepare as data

OUTPUT_ROOT = ROOT / "artifacts/audits/qwen_gemma_temporal_v1/tasks"


def _fresh_seed(folder: Path) -> None:
    """Check completed data and every pinned input before scheduling a model."""
    data.validate_seed(folder)


def _tasks(args) -> list[dict]:
    tasks = []
    for seed in args.seeds:
        _fresh_seed(args.data_root / f"seed{seed}")
    for pair in args.models:
        for variant in args.variants:
            for seed in args.seeds:
                folder = args.data_root / f"seed{seed}"
                manifest = folder / pair / variant / "manifest.json"
                output = args.output_root / pair / variant / f"seed{seed}"
                data.require(not (output == args.data_root or output.is_relative_to(args.data_root)
                                  or args.data_root.is_relative_to(output)),
                             "audit output overlaps frozen temporal data")
                tasks.append(dict(pair=pair, variant=variant, seed=seed,
                                  manifest=str(manifest), manifest_sha256=sha256(manifest),
                                  output=str(output), detector_epochs=args.detector_epochs))
    return tasks


def _worker(task: dict) -> None:
    manifest = Path(task["manifest"])
    _fresh_seed(manifest.parents[2])
    data.require(sha256(manifest) == task["manifest_sha256"], "worker manifest changed")
    from experiments.pretraining.evaluation import evaluate_main
    report = evaluate_main(manifest, task["output"], seed=task["seed"],
                           device="cuda:0", detector_epochs=task["detector_epochs"])
    print(json.dumps(dict(pair=task["pair"], variant=task["variant"], seed=task["seed"],
                          metrics=report["metrics"]), ensure_ascii=False), flush=True)


def _summarize(tasks: list[dict]) -> dict:
    from experiments.shared.audit.artifacts import read_result
    rows = []
    for task in tasks:
        folder = Path(task["output"]) / "main_fixed_sparse_positive"
        row = dict(pair=task["pair"], variant=task["variant"], seed=task["seed"],
                   report=str(folder / "REPORT.json"), state="missing")
        if (folder / "REPORT.json").exists():
            try:
                report = read_result(folder)
                if (report["settings"]["audit_seed"] != task["seed"] or
                        report["settings"]["detector_epochs"] != task["detector_epochs"] or
                        report["evaluation_context"]["data_manifest"] != task["manifest"] or
                        report["evaluation_context"]["membership_verified"] is not False):
                    raise ValueError("report belongs to another temporal condition")
                row.update(state="complete", metrics=report["metrics"])
            except (OSError, ValueError, KeyError) as error:
                row.update(state="invalid", error=str(error))
        rows.append(row)
    return dict(conditions=len(rows), complete=sum(r["state"] == "complete" for r in rows),
                rows=rows)


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_worker":
        if len(argv) != 2:
            raise ValueError("worker needs exactly one task JSON")
        _worker(json.loads(argv[1]))
        return
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("prepare", "dry-run", "run", "summarize"))
    parser.add_argument("--models", nargs="+", choices=data.PAIRS, default=list(data.PAIRS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=data.SEEDS, default=list(data.SEEDS))
    parser.add_argument("--variants", nargs="+", choices=data.VARIANTS,
                        default=["length_matched"])
    parser.add_argument("--data-root", type=Path, default=data.DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--detector-epochs", type=int, default=30)
    gpu_pool.add_arguments(parser)
    args = parser.parse_args(argv)
    for name in ("models", "seeds", "variants"):
        chosen = getattr(args, name)
        if not chosen or len(set(chosen)) != len(chosen):
            parser.error(f"--{name} must contain unique choices")
    if args.detector_epochs < 1:
        parser.error("--detector-epochs must be positive")
    args.data_root, args.output_root = args.data_root.resolve(), args.output_root.resolve()
    try:
        data.require(args.data_root != args.output_root and
                     not args.data_root.is_relative_to(args.output_root) and
                     not args.output_root.is_relative_to(args.data_root),
                     "audit output overlaps frozen temporal data")
        scheduling = gpu_pool.configuration(args)
        if args.command == "prepare":
            # Both pairs and both variants share the same raw article selection.
            for seed in args.seeds:
                folder = data.prepare_seed(seed, args.data_root)
                _fresh_seed(folder)
            print(json.dumps(dict(data_root=str(args.data_root), seeds=args.seeds,
                                  models=list(data.PAIRS), variants=list(data.VARIANTS)),
                             ensure_ascii=False, indent=2))
            return
        tasks = _tasks(args)
        if args.command == "dry-run":
            print(json.dumps(dict(conditions=len(tasks), scheduling=scheduling, tasks=tasks),
                             ensure_ascii=False, indent=2))
            return
        if args.command == "summarize":
            result = _summarize(tasks)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return
        jobs = [gpu_pool.Job(f'{t["pair"]}/{t["variant"]}/seed{t["seed"]}',
                             [sys.executable, "-B", "-u", "-m", __spec__.name,
                              "_worker", json.dumps(t)], t["seed"]) for t in tasks]
        rows = gpu_pool.run_jobs(jobs, scheduling=scheduling, cwd=ROOT,
                                 log_root=args.log_root or args.output_root.parent / "executions",
                                 use_cuda=True)
        if any(row["state"] != "complete" for row in rows):
            raise SystemExit(2)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    main()
