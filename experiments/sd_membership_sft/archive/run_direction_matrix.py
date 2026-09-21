"""Resume the registered nonmember-only detector experiment matrix.

This runner never fine-tunes target/draft models and never collects new GPU
observations. Collect frozen-model archives first; --ready-only permits an
explicit partial paired batch while additional GPU collection is running.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from experiments.sd_membership_sft.core.audit_runtime import (ROOT)


SEEDS = (20260914, 20260915, 20260916)
CONDITIONS = tuple((b, e) for b in ("wikitection", "newstection", "arxivtection") for e in (1, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("conditional", "paired", "active"))
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--ready-only", action="store_true")
    args = parser.parse_args()
    root = ROOT / "experiments/results/sft_runs"
    logs = root / "directions_validation/logs"
    logs.mkdir(parents=True, exist_ok=True)
    jobs, unavailable = [], []
    for benchmark, epoch in CONDITIONS:
        condition = f"{benchmark}_epoch{epoch}"
        for seed in SEEDS:
            if args.phase == "paired":
                source = root / f"counterfactual_observations/{condition}_seed{seed}.npz"
                if not source.with_suffix(".json").exists():
                    unavailable.append(str(source))
                    continue
            for budget in ((2, 8) if args.phase == "paired" else (2,)):
                if args.phase == "active":
                    output = root / f"directions_validation/active/{condition}/seed{seed}"
                    command = [sys.executable, "-m", "experiments.sd_membership_sft.active_protocol_design",
                               "--benchmark", benchmark, "--epoch", str(epoch), "--seed", str(seed)]
                else:
                    command = [sys.executable, "-m", "experiments.sd_membership_sft.conditional_accept_only",
                               "--seed", str(seed), "--budget", str(budget), "--threads", "2"]
                    if args.phase == "paired":
                        output = root / f"directions_validation/paired/{condition}/b{budget}_seed{seed}"
                        command += ["--observations", str(source), "--output-dir", str(output)]
                    else:
                        output = root / f"conditional_accept_only/{condition}/b{budget}_seed{seed}"
                        command += ["--benchmark", benchmark, "--epoch", str(epoch)]
                complete = (output / "REPORT.json").exists()
                if args.phase != "active":
                    complete = complete and (output / "SOURCE.json").exists()
                if not complete:
                    jobs.append((command, logs / f"{args.phase}_{condition}_b{budget}_seed{seed}.log"))
    if unavailable and not args.ready_only:
        raise FileNotFoundError(f"paired archives are incomplete: {unavailable}")
    print(json.dumps({"phase": args.phase, "pending_jobs": len(jobs), "unavailable_archives": unavailable}), flush=True)

    def run(job):
        command, path = job
        with path.open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        print(json.dumps({"completed": str(path)}), flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(run, jobs))


if __name__ == "__main__":
    main()
