"""Schedule TSV shell conditions with the shared dynamic GPU process queue."""
import argparse
import json
from pathlib import Path
import sys

from experiments.paths import ROOT
from experiments.shared.core import gpu_pool
from experiments.launchers.devices import run_jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--script', type=Path, required=True)
    gpu_pool.add_arguments(parser)
    args = parser.parse_args(argv)
    if not args.log_root:
        parser.error('--log-root is required')
    conditions = [line.strip().split('\t') for line in sys.stdin if line.strip()]
    jobs = []
    for fields in conditions:
        # pair, dataset, epoch, seed, optionally phase. Source conversion uses
        # the same shape, keeping the immutable MTP prerequisite a single job.
        if len(fields) not in (4, 5):
            parser.error('expected pair, benchmark, epoch, seed, optional phase')
        seed = int(fields[3])
        jobs.append(gpu_pool.Job('/'.join(fields),
            ['bash', str(args.script.resolve()), '--_condition', *fields], seed))
    try:
        rows = run_jobs(jobs, scheduling=gpu_pool.configuration(args), cwd=ROOT, log_root=args.log_root)
    except KeyboardInterrupt:
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(dict(conditions=len(rows), failed=sum(row['state'] != 'complete' for row in rows))))
    if any(row['state'] != 'complete' for row in rows):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
