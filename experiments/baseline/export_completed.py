"""Export completed-method NPZs from one execution, without model inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import METHODS
from .run import _render_report
from .costs import write_cost_report


def export_completed(execution_dir: Path, output_dir: Path) -> None:
    scores, costs = {}, {}
    protocol = labels = record_ids = None
    for name in METHODS:
        path = execution_dir / f"{name}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as artifact:
            candidate = json.loads(str(artifact['protocol_json']))
            candidate.pop('methods')
            if protocol is None:
                protocol = candidate
                labels = artifact['labels'].copy()
                record_ids = artifact['record_ids'].copy()
            elif (protocol != candidate or not np.array_equal(labels, artifact['labels'])
                  or not np.array_equal(record_ids, artifact['record_ids'])):
                raise ValueError(f"incompatible artifact: {path}")
            values = artifact[name]
            if values.shape != labels.shape:
                raise ValueError(f"score/label shape mismatch: {path}")
            scores[name] = values.tolist()
            if "cost_json" in artifact:
                costs[name] = json.loads(str(artifact["cost_json"]))
    if not scores:
        raise ValueError(f"no completed methods in {execution_dir}")
    # Explicitly prevent a recovery export from silently overwriting a run.
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ('baseline_metrics.json', 'baseline_scores.npz', 'BASELINE_RESULTS.md', 'baseline_costs.json', 'BASELINE_COSTS.md'):
        if (output_dir / name).exists():
            raise FileExistsError(output_dir / name)
    protocol = {**protocol, 'methods': list(scores), 'recovered_from': str(execution_dir)}
    if np.unique(labels).size >= 2:
        _render_report(output_dir, protocol, scores, labels, costs=costs or None)
    else:
        (output_dir / 'baseline_metrics.json').write_text(json.dumps(
            dict(protocol=protocol, metrics={}, scores=scores, costs=costs), indent=2), encoding='utf-8')
    if costs:
        write_cost_report(output_dir, protocol, costs)
    np.savez_compressed(output_dir / 'baseline_scores.npz', labels=labels,
                        record_ids=record_ids, **{k: np.asarray(v, dtype=np.float32) for k, v in scores.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    export_completed(args.execution_dir, args.output_dir)


if __name__ == '__main__':
    main()
