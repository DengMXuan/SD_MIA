"""One offline model/GPU/data preflight for every controlled-SFT matrix."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .drafts.common import ROOT, cached_snapshot
from .head_matrix_preflight import (
    _has_weights,
    prepare_shared_splits,
    validate_cached_models,
    validate_gpus,
)


PLAIN_PAIR_MODELS = {
    "qwen3": {
        "target": "Qwen/Qwen3-8B-Base",
        "draft": "Qwen/Qwen3-1.7B-Base",
        "target_revision_key": "QWEN_TARGET_REVISION",
        "draft_revision_key": "QWEN_DRAFT_REVISION",
    },
    "gemma4": {
        "target": "google/gemma-4-12B",
        "draft": "google/gemma-4-E2B",
        "target_revision_key": "GEMMA_TARGET_REVISION",
        "draft_revision_key": "GEMMA_DRAFT_REVISION",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("experiments/results/sft_runs/unified_matrix_v1/shared_splits"),
    )
    parser.add_argument(
        "--model-revisions-env",
        type=Path,
        default=Path(
            "experiments/sd_membership_sft/model_pair_revisions.env"
        ),
    )
    parser.add_argument("--gpus", nargs="+", type=int, default=[3, 4, 5, 6])
    parser.add_argument("--skip-gpu-check", action="store_true")
    parser.add_argument("--skip-gpu-busy-check", action="store_true")
    return parser.parse_args()


def _read_revisions(path: Path) -> dict[str, str]:
    if not path.is_absolute():
        path = ROOT / path
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator:
            raise RuntimeError(f"Invalid revision line in {path}: {raw_line!r}")
        values[key.strip()] = value.strip()
    required = {
        str(spec[key])
        for spec in PLAIN_PAIR_MODELS.values()
        for key in ("target_revision_key", "draft_revision_key")
    }
    missing = sorted(required - values.keys())
    if missing:
        raise RuntimeError(f"Missing model revisions in {path}: {missing}")
    invalid = {
        key: values[key]
        for key in required
        if len(values[key]) != 40
        or any(character not in "0123456789abcdef" for character in values[key])
    }
    if invalid:
        raise RuntimeError(
            f"Model revisions must be 40-character commits: {invalid}"
        )
    return values


def _special_token_ids(tokenizer: Any) -> tuple[int | None, int | None, int | None]:
    return (
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
    )


def validate_plain_models(revision_env: Path) -> dict[str, Any]:
    """Validate both full-model pairs and return their actual training tokenizers."""
    revisions = _read_revisions(revision_env)
    tokenizers: dict[str, Any] = {}
    for pair, spec in PLAIN_PAIR_MODELS.items():
        loaded: dict[str, Any] = {}
        for role in ("target", "draft"):
            model_id = str(spec[role])
            revision = revisions[str(spec[f"{role}_revision_key"])]
            snapshot = cached_snapshot(model_id, revision)
            if not (snapshot / "config.json").is_file() or not _has_weights(snapshot):
                raise RuntimeError(
                    f"Cached snapshot is incomplete: {model_id}@{revision}"
                )
            config = AutoConfig.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=True,
            )
            model_class = AutoModelForCausalLM._model_mapping[type(config)]
            tokenizer = AutoTokenizer.from_pretrained(
                snapshot,
                local_files_only=True,
                trust_remote_code=True,
            )
            loaded[role] = tokenizer
            print(
                f"[model-ok] {model_id}@{revision} role={role} "
                f"type={model_class.__name__} vocab={len(tokenizer)}",
                flush=True,
            )
        target = loaded["target"]
        draft = loaded["draft"]
        if target.get_vocab() != draft.get_vocab() or _special_token_ids(
            target
        ) != _special_token_ids(draft):
            raise RuntimeError(f"Incompatible pair tokenizers: {pair}")
        draft_id = str(spec["draft"])
        draft_revision = revisions[str(spec["draft_revision_key"])]
        source = f"{draft_id}@{draft_revision}"
        tokenizers[source] = draft
        print(f"[pair-ok] {pair} tokenizer={source}", flush=True)
    return tokenizers


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    validate_gpus(
        args.gpus,
        skip_check=args.skip_gpu_check,
        skip_busy_check=args.skip_gpu_busy_check,
    )
    tokenizers = validate_cached_models()
    plain_tokenizers = validate_plain_models(args.model_revisions_env)
    overlap = set(tokenizers).intersection(plain_tokenizers)
    if overlap:
        raise RuntimeError(
            f"Duplicate tokenizer source registrations: {sorted(overlap)}"
        )
    tokenizers.update(plain_tokenizers)
    prepare_shared_splits(tokenizers, args.split_root)
    print(
        f"[unified-preflight-ok] tokenizer_sources={len(tokenizers)} "
        "shared_splits=9 conditions=90 artifacts=270",
        flush=True,
    )


if __name__ == "__main__":
    main()
