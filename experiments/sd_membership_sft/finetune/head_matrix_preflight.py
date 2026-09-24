"""Offline cache, GPU, and shared-split preflight for the head matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

from transformers import AutoConfig, AutoTokenizer

from experiments.shared.core.data_contract import DEFAULT_DATA_CONTRACT
from experiments.shared.drafts.common import PAIR_MODELS, ROOT, cached_snapshot, tokenizer_source_for
from experiments.shared.data.splits import CONTROLLED_SPLIT_SCHEMA_VERSION, build_controlled_split_from_shared_manifest, pool_path, prepare_shared_split_manifest


BENCHMARKS = ("wikitection", "newstection", "arxivtection")
SEEDS = (1919, 1949, 1978)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path(
            "artifacts/training/controlled_sft_v2/splits"
        ),
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=[3, 4, 5, 6])
    parser.add_argument("--skip-gpu-check", action="store_true")
    parser.add_argument("--skip-gpu-busy-check", action="store_true")
    return parser.parse_args()


def _run_nvidia_smi(query: str) -> list[list[str]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-{query}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        [field.strip() for field in line.split(",")]
        for line in result.stdout.splitlines()
        if line.strip()
    ]


def validate_gpus(
    gpu_indices: list[int], *, skip_check: bool, skip_busy_check: bool
) -> None:
    if len(gpu_indices) != 4 or len(set(gpu_indices)) != 4:
        raise RuntimeError("Exactly four distinct physical GPU indices are required")
    if skip_check:
        print("[gpu-check-skipped]", flush=True)
        return
    gpu_rows = _run_nvidia_smi("gpu=index,uuid,name,memory.total")
    by_index = {int(row[0]): row for row in gpu_rows}
    missing = [index for index in gpu_indices if index not in by_index]
    if missing:
        raise RuntimeError(f"Configured physical GPUs are unavailable: {missing}")
    selected_uuids = {by_index[index][1]: index for index in gpu_indices}
    if not skip_busy_check:
        app_rows = _run_nvidia_smi(
            "compute-apps=gpu_uuid,pid,process_name,used_memory"
        )
        busy = [
            {
                "gpu": selected_uuids[row[0]],
                "pid": row[1],
                "process": row[2],
                "used_memory_mib": row[3],
            }
            for row in app_rows
            if row[0] in selected_uuids
        ]
        if busy:
            raise RuntimeError(
                "Configured GPUs already have compute processes: "
                + json.dumps(busy, ensure_ascii=False)
            )
    for index in gpu_indices:
        row = by_index[index]
        print(
            f"[gpu-ok] index={index} uuid={row[1]} name={row[2]} "
            f"memory_mib={row[3]}",
            flush=True,
        )


def _has_weights(path: Path) -> bool:
    return any(path.glob("*.safetensors")) or any(
        path.glob("pytorch_model*.bin")
    )


def validate_cached_models() -> dict[str, Any]:
    tokenizers: dict[str, Any] = {}
    checked_snapshots: set[tuple[str, str]] = set()
    for pair, model in PAIR_MODELS.items():
        for role in ("target", "speculator"):
            model_id = model[role]
            revision = model[f"{role}_revision"]
            key = (model_id, revision)
            if key in checked_snapshots:
                continue
            snapshot = cached_snapshot(model_id, revision)
            if not (snapshot / "config.json").is_file() or not _has_weights(snapshot):
                raise RuntimeError(
                    f"Cached snapshot is incomplete: {model_id}@{revision}"
                )
            if model["kind"] == "eagle3" and role == "speculator":
                if not (snapshot / "eagle3.py").is_file():
                    raise RuntimeError(f"EAGLE-3 snapshot lacks eagle3.py: {snapshot}")
                model_type = "eagle3-speculator"
            else:
                config = AutoConfig.from_pretrained(
                    snapshot, local_files_only=True, trust_remote_code=True
                )
                model_type = str(getattr(config, "model_type", type(config).__name__))
            print(
                f"[model-ok] {model_id}@{revision} role={role} type={model_type}",
                flush=True,
            )
            checked_snapshots.add(key)

        source = tokenizer_source_for(pair)
        if source not in tokenizers:
            tokenizer = AutoTokenizer.from_pretrained(
                model["target"],
                revision=model["target_revision"],
                local_files_only=True,
            )
            tokenizers[source] = tokenizer
            print(
                f"[tokenizer-ok] {source} vocab={len(tokenizer)}",
                flush=True,
            )
    return tokenizers


def prepare_shared_splits(tokenizers: dict[str, Any], split_root: Path) -> None:
    if not split_root.is_absolute():
        split_root = ROOT / split_root
    eligibility_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    pool_hashes: dict[str, str] = {}
    for benchmark in BENCHMARKS:
        source_pool = ROOT / pool_path(benchmark)
        pool_hashes[benchmark] = hashlib.sha256(source_pool.read_bytes()).hexdigest()
        for seed in SEEDS:
            destination = split_root / benchmark / f"seed{seed}.json"
            audit_path = destination.with_suffix(".audit.json")
            if destination.is_file() and audit_path.is_file():
                manifest_bytes = destination.read_bytes()
                existing_manifest = json.loads(manifest_bytes)
                existing_audit = json.loads(audit_path.read_text(encoding="utf-8"))
                manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
                expected_sources = set(tokenizers)
                audited_sources = set(existing_audit.get("tokenizers", {}))
                reusable = (
                    existing_manifest.get("schema_version")
                    == CONTROLLED_SPLIT_SCHEMA_VERSION
                    and existing_manifest.get("benchmark") == benchmark
                    and existing_manifest.get("seed") == seed
                    and existing_manifest.get("pool_sha256")
                    == pool_hashes[benchmark]
                    and set(existing_manifest.get("tokenizer_sources", []))
                    == expected_sources
                    and existing_manifest.get("counts")
                    == DEFAULT_DATA_CONTRACT.shared_manifest_counts
                    and existing_audit.get("benchmark") == benchmark
                    and existing_audit.get("seed") == seed
                    and existing_audit.get("shared_split_schema_version")
                    == CONTROLLED_SPLIT_SCHEMA_VERSION
                    and existing_audit.get("selection")
                    == existing_manifest.get("selection")
                    and audited_sources == expected_sources
                    and all(
                        item.get("shared_split_sha256") == manifest_sha
                        and item.get("cross_split_ngram_audit", {}).get("gate")
                        == "PASS"
                        for item in existing_audit["tokenizers"].values()
                    )
                )
                if not reusable:
                    raise RuntimeError(
                        f"Existing shared split or audit is stale: {destination}"
                    )
                print(
                    f"[split-reused] benchmark={benchmark} seed={seed} "
                    f"tokenizer_audits={len(audited_sources)} path={destination}",
                    flush=True,
                )
                continue
            print(
                f"[split-start] benchmark={benchmark} seed={seed} "
                f"tokenizers={len(tokenizers)}",
                flush=True,
            )
            artifact = prepare_shared_split_manifest(
                benchmark,
                source_pool,
                tokenizers,
                n_per_class=DEFAULT_DATA_CONTRACT.members,
                n_aux=DEFAULT_DATA_CONTRACT.draft_auxiliary,
                n_audit_aux=DEFAULT_DATA_CONTRACT.audit_auxiliary,
                seed=seed,
                output_path=destination,
                eligibility_cache=eligibility_cache,
            )
            print(
                f"[split-selected] benchmark={benchmark} seed={seed} "
                f"survivors="
                f"{artifact['selection']['survivors_after_all_tokenizers']} "
                f"selected={artifact['selection']['selected_documents']}",
                flush=True,
            )
            tokenizer_audits: dict[str, Any] = {}
            for source, tokenizer in tokenizers.items():
                print(
                    f"[split-audit-start] benchmark={benchmark} seed={seed} "
                    f"tokenizer={source}",
                    flush=True,
                )
                controlled_split = build_controlled_split_from_shared_manifest(
                    benchmark,
                    source_pool,
                    tokenizer,
                    destination,
                    source,
                )
                metadata = controlled_split.metadata
                tokenizer_audits[source] = {
                    "shared_split_sha256": metadata["shared_split_sha256"],
                    "counts": metadata["counts"],
                    "cross_split_ngram_audit": metadata[
                        "cross_split_ngram_audit"
                    ],
                    "exact_token_deduplication": metadata[
                        "exact_token_deduplication"
                    ],
                }
                print(
                    f"[split-audit-ok] benchmark={benchmark} seed={seed} "
                    f"tokenizer={source} maximum_overlap="
                    f"{metadata['cross_split_ngram_audit']['maximum_pair_overlap_fraction']:.6f}",
                    flush=True,
                )
            audit_artifact = {
                "shared_split_schema_version": CONTROLLED_SPLIT_SCHEMA_VERSION,
                "benchmark": benchmark,
                "seed": seed,
                "manifest": str(destination),
                "selection": artifact["selection"],
                "tokenizers": tokenizer_audits,
            }
            # Normalize tuples from the n-gram audit to their JSON list form so
            # an identical second preflight compares equal to the saved file.
            audit_artifact = json.loads(
                json.dumps(audit_artifact, ensure_ascii=False)
            )
            if audit_path.exists():
                current = json.loads(audit_path.read_text(encoding="utf-8"))
                if current != audit_artifact:
                    raise RuntimeError(
                        f"Refusing to replace a different split audit: {audit_path}"
                    )
            else:
                temporary = audit_path.with_name(
                    f".{audit_path.name}.tmp.{os.getpid()}"
                )
                temporary.write_text(
                    json.dumps(audit_artifact, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, audit_path)
            print(
                f"[split-ok] benchmark={benchmark} seed={seed} "
                f"common_eligible={artifact['common_eligible_documents']} "
                f"deduplicated_survivors="
                f"{artifact['selection']['survivors_after_all_tokenizers']} "
                f"near_duplicates_rejected="
                f"{artifact['selection']['rejected_near_duplicate']} "
                f"tokenizer_audits={len(tokenizer_audits)} path={destination}",
                flush=True,
            )


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    validate_gpus(
        args.gpus,
        skip_check=args.skip_gpu_check,
        skip_busy_check=args.skip_gpu_busy_check,
    )
    tokenizers = validate_cached_models()
    prepare_shared_splits(tokenizers, args.split_root)
    print("[preflight-ok] pinned offline models and shared splits are ready", flush=True)


if __name__ == "__main__":
    main()
