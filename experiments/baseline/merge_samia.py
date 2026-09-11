"""Merge raw SaMIA scores produced by parallel audit-record shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .run import _render_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    part_dirs = sorted(
        path
        for path in args.parts_dir.glob("part*")
        if (path / "baseline_metrics.json").exists()
        and (path / "baseline_scores.npz").exists()
    )
    if not part_dirs:
        raise FileNotFoundError(f"no completed SaMIA shards found in {args.parts_dir}")

    shards: list[tuple[int, int, Path, dict[str, object], np.lib.npyio.NpzFile]] = []
    for part_dir in part_dirs:
        protocol = json.loads(
            (part_dir / "baseline_metrics.json").read_text(encoding="utf-8")
        )["protocol"]
        start = int(protocol["record_start"])
        end = int(protocol["record_end"])
        scores = np.load(part_dir / "baseline_scores.npz")
        if set(scores.files) != {"labels", "record_ids", "samia"}:
            raise ValueError(f"unexpected arrays in {part_dir}: {scores.files}")
        if len(scores["labels"]) != end - start:
            raise ValueError(f"length/range mismatch in {part_dir}")
        shards.append((start, end, part_dir, protocol, scores))

    shards.sort(key=lambda item: item[0])
    expected_start = 0
    for start, end, part_dir, _, _ in shards:
        if start != expected_start:
            raise ValueError(
                f"SaMIA shard gap/overlap before {part_dir}: "
                f"expected {expected_start}, found {start}"
            )
        expected_start = end
    if expected_start != 4000:
        raise ValueError(f"expected 4,000 audit records, found {expected_start}")

    labels = np.concatenate([scores["labels"] for _, _, _, _, scores in shards])
    record_ids = np.concatenate(
        [scores["record_ids"] for _, _, _, _, scores in shards]
    )
    samia = np.concatenate([scores["samia"] for _, _, _, _, scores in shards])
    if len(np.unique(record_ids)) != len(record_ids):
        raise ValueError("duplicate record IDs in SaMIA shards")
    if tuple(np.unique(labels, return_counts=True)[1]) != (2000, 2000):
        raise ValueError("merged SaMIA labels are not 2,000/2,000")

    protocol = dict(shards[0][3])
    protocol.update(
        {
            "record_start": 0,
            "record_end": int(len(labels)),
            "n_member": int(np.sum(labels == 1)),
            "n_nonmember": int(np.sum(labels == 0)),
            "parallel_shards": len(shards),
            "shard_ranges": [[start, end] for start, end, *_ in shards],
        }
    )
    scores = {"samia": samia.astype(np.float64).tolist()}
    _render_report(args.output_dir, protocol, scores, labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "baseline_scores.npz",
        labels=labels,
        record_ids=record_ids,
        samia=samia.astype(np.float32),
    )
    print(json.dumps({"output_dir": str(args.output_dir), "n": len(labels)}))


if __name__ == "__main__":
    main()
