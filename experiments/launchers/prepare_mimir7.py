"""Freeze only the low-overlap MIMIR benchmark; no GPU evaluation."""
import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--source', choices=('github', 'arxiv'), required=True)
    parser.add_argument('--seeds', type=int, nargs='+', choices=(1919, 1949, 1978), default=[1919, 1949, 1978])
    parser.add_argument('--data-root', type=Path, help='output root for frozen 7-gram manifests')
    parser.add_argument('--official-root', type=Path, help='directory for official 7-gram cache files')
    parser.add_argument('--auxiliary-root', type=Path,
                        help='root containing prepared 13-gram manifests and official cache files')
    parser.add_argument('--allow-download', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('duplicate seeds')
    # Hub reads offline mode at import time.
    os.environ['HF_HUB_OFFLINE'] = '0' if args.allow_download else '1'
    from experiments.pretraining.benchmarks.prepare import prepare_mimir
    from experiments.pretraining.benchmarks.prepare import DATA_ROOT
    for seed in args.seeds:
        print(json.dumps(dict(manifest=str(prepare_mimir(
            args.source, seed, data_root=args.data_root or DATA_ROOT,
            official_root=args.official_root, auxiliary_root=args.auxiliary_root,
            allow_download=args.allow_download)))))


if __name__ == '__main__':
    main()
