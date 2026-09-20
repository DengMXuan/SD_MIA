"""Explicit sequential DP matrix launcher; dry-run never loads a model."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from experiments.sd_membership_sft.audit_runtime import ROOT


def commands(args):
    result = []
    for benchmark in args.benchmarks:
        for epoch in args.epochs:
            for seed in args.seeds:
                condition = Path(benchmark) / f"epoch{epoch}" / f"seed{seed}"
                for epsilon in args.epsilons:
                    relative = Path(f"epsilon{epsilon:g}") / condition
                    model = args.model_root / relative
                    train = [sys.executable, "-m", "experiments.dp_defense.train", "run",
                             "--reference-run", str(args.reference_root / condition), "--output-dir", str(model),
                             "--epsilon", str(epsilon), "--gpu", str(args.gpu)]
                    audit = [sys.executable, "-m", "experiments.dp_defense.audit", "run",
                             "--run-dir", str(model), "--output-dir", str(args.audit_root / relative),
                             "--device", f"cuda:{args.gpu}"]
                    if args.include_baselines:
                        audit.append("--include-baselines")
                    result.append(dict(condition=str(condition), epsilon=epsilon, train=train, audit=audit))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("dry-run", "train", "audit"))
    parser.add_argument("--reference-root", type=Path, default=ROOT / "experiments/results/sft_runs/unified_matrix_audit600_v2/model_pairs/qwen3")
    parser.add_argument("--model-root", type=Path, default=ROOT / "experiments/results/sft_runs/dp_defense_v1/models")
    parser.add_argument("--audit-root", type=Path, default=ROOT / "experiments/results/sft_runs/dp_defense_v1/audits")
    parser.add_argument("--benchmarks", nargs="+", choices=("wikitection", "newstection", "arxivtection"),
                        default=["wikitection", "newstection", "arxivtection"])
    parser.add_argument("--epochs", nargs="+", type=int, choices=(1, 3), default=[1, 3])
    parser.add_argument("--seeds", nargs="+", type=int, choices=(1919, 1949, 1978), default=[1919, 1949, 1978])
    parser.add_argument("--epsilons", nargs="+", type=float, choices=(1., 4., 8.), default=[1., 4., 8.])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--include-baselines", action="store_true")
    args = parser.parse_args()
    if args.gpu < 0 or any(len(v) != len(set(v)) for v in (args.benchmarks, args.epochs, args.seeds, args.epsilons)):
        parser.error("choose a nonnegative GPU and unique matrix values")
    tasks = commands(args)
    if args.command == "dry-run":
        print(json.dumps({"conditions": len(tasks), "artifacts": 3 * len(tasks), "tasks": tasks}, indent=2))
        return
    failures = []
    for task in tasks:
        # subprocess.run waits and propagates Ctrl-C; there are no detached workers.
        code = subprocess.run(task[args.command], cwd=ROOT).returncode
        if code:
            failures.append({"condition": task["condition"], "epsilon": task["epsilon"], "exit_code": code})
    print(json.dumps({"failures": failures}))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
