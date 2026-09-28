"""Freeze only the low-overlap MIMIR benchmark; no GPU evaluation."""
import argparse
import json
import os


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--source', choices=('github', 'arxiv'), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', choices=(1919, 1949, 1978), default=[1919, 1949, 1978])
    parser.add_argument('--allow-download', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('duplicate seeds')
    # Hub reads offline mode at import time.
    os.environ['HF_HUB_OFFLINE'] = '0' if args.allow_download else '1'
    from experiments.pretraining.benchmarks.prepare import prepare_mimir
    for seed in args.seeds:
        print(json.dumps(dict(manifest=str(prepare_mimir(args.source, seed, allow_download=args.allow_download)))))


if __name__ == '__main__':
    main()
