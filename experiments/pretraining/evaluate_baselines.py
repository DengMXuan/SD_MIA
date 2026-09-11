"""Evaluate baseline scores on exactly the frozen M1 calibration/test records."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..sd_membership_sft.m1_fit import make_partitions, evaluate_score_vector, _jsonable
from .data import load_evaluation


def evaluate(manifest_path, baseline_dir, partition_path, output):
    evaluation = load_evaluation(Path(manifest_path))
    expected = evaluation.members + evaluation.nonmembers
    baseline_dir = Path(baseline_dir)
    protocol = json.loads((baseline_dir / 'baseline_metrics.json').read_text())['protocol']
    if (protocol.get('training_regime') != 'pretraining' or
        protocol.get('split_metadata') != evaluation.manifest):
        raise ValueError('baseline provenance differs from requested frozen pretraining evaluation')
    with np.load(baseline_dir / 'baseline_scores.npz', allow_pickle=False) as archive:
        ids, labels = archive['record_ids'], archive['labels']
        if not np.array_equal(ids, [r.record_id for r in expected]):
            raise ValueError('baseline must contain the complete frozen record order')
        if not np.array_equal(labels, [1] * len(evaluation.members) + [0] * len(evaluation.nonmembers)):
            raise ValueError('baseline labels changed')
        partitions = make_partitions(labels, ids, frozen_manifest_path=Path(partition_path))
        methods = {name: evaluate_score_vector(archive[name], SimpleNamespace(labels=labels), partitions, bootstrap=None, bootstrap_repeats=0)
                   for name in protocol['methods']}
    report = dict(protocol=protocol, partition_manifest=str(Path(partition_path).resolve()), methods=methods)
    Path(output).write_text(json.dumps(_jsonable(report), indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--baseline-dir', type=Path, required=True)
    parser.add_argument('--partition-manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.manifest, args.baseline_dir, args.partition_manifest, args.output)


if __name__ == '__main__':
    main()
