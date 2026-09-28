"""Historical Qwen four-role SFT recipe, now a dynamic condition queue."""
import argparse
import json
import os
from pathlib import Path
import sys

from experiments.paths import ROOT
from experiments.shared.core import gpu_pool
from experiments.launchers.devices import run_jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('command', nargs='?', default='dry-run', choices=('dry-run', 'run', 'status'))
    parser.add_argument('--benchmarks', nargs='+', choices=('wikitection', 'newstection', 'arxivtection'),
                        default=['wikitection', 'newstection', 'arxivtection'])
    parser.add_argument('--epochs', type=int, nargs='+', choices=(1, 3), default=[1, 3])
    parser.add_argument('--output-root', type=Path, default=Path(os.environ.get(
        'RESULTS', ROOT / 'artifacts/training/four_role_v1/runs')))
    gpu_pool.add_arguments(parser)
    args = parser.parse_args(argv)
    if len(set(args.benchmarks)) != len(args.benchmarks) or len(set(args.epochs)) != len(args.epochs):
        parser.error('duplicate conditions')
    jobs, tasks = [], []
    for benchmark in args.benchmarks:
        for epoch in args.epochs:
            output = args.output_root.resolve() / f'{benchmark}_qwen3_8b_epoch{epoch}'
            complete = (output / 'results.json').is_file()
            tasks.append(dict(benchmark=benchmark, epoch=epoch, output=str(output), complete=complete))
            if complete:
                continue
            command = [sys.executable, '-B', '-u', '-m', 'experiments.shared.drafts.plain',
                '--gpu', '0', '--benchmark', benchmark, '--target-epochs', str(epoch),
                '--trainer', 'full', '--optimizer', 'adamw8bit', '--target-lr', '2e-5', '--draft-lr', '2e-5',
                '--target-batch-size', '2', '--target-grad-accum', '8', '--draft-batch-size', '2',
                '--draft-grad-accum', '8', '--n-per-class', '2000', '--n-aux', '2000', '--n-audit-aux', '600',
                '--seed', '20260824', '--data-seed', '20260824', '--distill-steps', '384',
                '--output-dir', str(output)]
            if (output / 'checkpoints/target/config.json').exists():
                command.append('--resume')
            jobs.append(gpu_pool.Job(f'{benchmark}/epoch{epoch}', command, 20260824))
    scheduling = gpu_pool.configuration(args)
    if args.command != 'run':
        print(json.dumps(dict(tasks=tasks, scheduling=scheduling,
                              commands=[job.command for job in jobs]), indent=2))
        return
    try:
        rows = run_jobs(jobs, scheduling=scheduling, cwd=ROOT,
                        log_root=args.log_root or args.output_root.resolve().parent / 'executions')
        if any(row['state'] != 'complete' for row in rows):
            raise SystemExit(2)
    except KeyboardInterrupt:
        raise SystemExit(130)


if __name__ == '__main__':
    main()
