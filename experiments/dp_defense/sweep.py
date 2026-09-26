"""DP matrix with independent GPU workers; dry-run never loads a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from experiments.paths import ROOT, TRAINING_ROOT, AUDITS, audit_executions
from experiments.shared.core import gpu_pool
from experiments.shared.models.registry import MODEL_PAIRS
from experiments.dp_defense.conditions import variants, stage_roles, accumulator_settings


def commands(args):
    result = []
    pairs = getattr(args, "model_pairs", ["qwen3"])
    selected = variants(getattr(args, 'draft_variants', None))
    execution = accumulator_settings(getattr(args, 'accumulator_device', 'cpu'))
    if args.reference_root is not None and len(pairs) != 1:
        raise ValueError("--reference-root requires exactly one model pair")
    for pair in pairs:
        reference_root = args.reference_root or MODEL_PAIRS[pair].run_root
        for benchmark in args.benchmarks:
            for epoch in args.epochs:
                for seed in args.seeds:
                    condition = Path(benchmark) / f"epoch{epoch}" / f"seed{seed}"
                    for epsilon in args.epsilons:
                        relative = Path(pair) / f"epsilon{epsilon:g}" / condition
                        model = args.model_root / relative
                        train = [sys.executable, "-m", "experiments.dp_defense.train", "run",
                                 "--reference-run", str(reference_root / condition), "--output-dir", str(model),
                                 "--epsilon", str(epsilon), "--gpu", str(args.gpu)]
                        audit = [sys.executable, "-m", "experiments.dp_defense.audit", "run",
                                 "--run-dir", str(model), "--output-dir", str(args.audit_root / relative),
                                 "--device", f"cuda:{args.gpu}"]
                        if args.include_baselines:
                            audit.append("--include-baselines")
                        train += ['--accumulator-device', execution['accumulator_device'], '--draft-variants', *selected]
                        audit += ['--draft-variants', *selected]
                        result.append(dict(model_pair=pair, condition=str(condition), seed=seed, epsilon=epsilon,
                                           draft_variants=list(selected), stages=list(stage_roles(selected)),
                                           train=train, audit=audit))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("command", choices=("dry-run", "train", "audit"))
    parser.add_argument("--model-pairs", nargs="+", choices=tuple(MODEL_PAIRS), default=["qwen3"])
    parser.add_argument("--reference-root", type=Path, help="training root override; single model pair only")
    parser.add_argument("--model-root", type=Path, default=TRAINING_ROOT / "dp_defense_v1/runs")
    parser.add_argument("--audit-root", type=Path, default=AUDITS / "dp_defense_v1/tasks")
    parser.add_argument("--benchmarks", nargs="+", choices=("wikitection", "newstection", "arxivtection"),
                        default=["wikitection", "newstection", "arxivtection"])
    parser.add_argument("--epochs", nargs="+", type=int, choices=(1, 3), default=[1, 3])
    parser.add_argument("--seeds", nargs="+", type=int, choices=(1919, 1949, 1978), default=[1919, 1949, 1978])
    parser.add_argument("--epsilons", nargs="+", type=float, choices=(1., 4., 8.), default=[1., 4., 8.])
    gpu_pool.add_arguments(parser)
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--accumulator-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument('--draft-variants', nargs='+', choices=('kd', 'member'), default=['kd', 'member'])
    args = parser.parse_args(argv)
    if any(len(v) != len(set(v)) for v in (args.model_pairs, args.benchmarks, args.epochs, args.seeds, args.epsilons)):
        parser.error("choose a nonnegative GPU and unique matrix values")
    try:
        scheduling = gpu_pool.configuration(args)
        args.gpu = 0  # Stable worker-local device, independent of assigned GPU.
        tasks = commands(args)
    except ValueError as error:
        parser.error(str(error))
    if args.command == "dry-run":
        print(json.dumps({"conditions": len(tasks), "artifacts": sum(len(t['stages']) for t in tasks),
                          "draft_audits": sum(len(t['draft_variants']) for t in tasks),
                          "scheduling": scheduling, "tasks": tasks}, indent=2))
        return
    jobs = [gpu_pool.Job(f"{t['model_pair']}/epsilon{t['epsilon']:g}/{t['condition']}",
                         t[args.command], t['seed']) for t in tasks]
    logs = (args.model_root.parent / 'executions/train' if args.command == 'train'
            else audit_executions(args.audit_root) / 'audit')
    try:
        rows = gpu_pool.run_jobs(jobs, scheduling=scheduling, log_root=args.log_root or logs, cwd=ROOT)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except ValueError as error:
        parser.error(str(error))
    failures = [row for row in rows if row['state'] != 'complete']
    print(json.dumps({"failures": failures}))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
