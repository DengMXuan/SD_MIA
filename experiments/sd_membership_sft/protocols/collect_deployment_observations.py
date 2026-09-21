"""Collect a four-role, fixed-candidate accept-only deployment archive.

The draft pass persists only local q-derived features. The verifier pass holds
target log probabilities in GPU/CPU memory just long enough to emit two binary
accept decisions per candidate token; target probabilities are never written.
Both already fine-tuned language models remain frozen.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import torch

from experiments.sd_membership_sft.core.audit_runtime import _record_uniforms, _write_json
from experiments.sd_membership_sft.datasets.data import collate_sft, make_sft_example
from experiments.sd_membership_sft.core.deployment_archive import (
    checkpoint_fingerprint,
    sha256_file,
    validate_deployment_archive,
    write_deployment_archive,
)
from experiments.sd_membership_sft.finetune.generalization import load_draft_model, load_finetuned_model, load_run_config
from experiments.sd_membership_sft.analysis.m1_features import Q_FEATURE_NAMES, q_features_from_logits
from experiments.sd_membership_sft.analysis.pq_gap_mia import selected_token_logprobs
from experiments.sd_membership_sft.core.scoring_common import (
    prepare_deployment_scoring_records,
    resolve_checkpoint_path,
    resolve_run_dir,
)
from experiments.sd_membership_sft.finetune.training import set_seed


from experiments.paths import ROOT


def _atomic_save(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp.npy")
    np.save(temporary, values)
    temporary.replace(path)


def _open_array(path: Path, shape: tuple[int, ...], dtype: Any) -> np.memmap:
    if path.exists():
        value = np.load(path, mmap_mode="r+")
        if value.shape != shape or value.dtype != np.dtype(dtype):
            raise ValueError(f"stale collection array: {path}")
        return value
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _batch(
    records: list[Any], tokenizer: Any, indices: np.ndarray, device: torch.device
) -> tuple[dict[str, torch.Tensor], np.ndarray]:
    examples = [
        make_sft_example(replace(records[int(index)], append_eos=False), tokenizer)
        for index in indices
    ]
    values = collate_sft(examples, int(tokenizer.pad_token_id))
    return {name: value.to(device) for name, value in values.items()}, indices


@torch.inference_mode()
def _collect_draft_features(
    model: Any,
    records: list[Any],
    tokenizer: Any,
    lengths: np.ndarray,
    offsets: np.ndarray,
    work_dir: Path,
    device: torch.device,
    batch_size: int,
) -> np.memmap:
    feature_path = work_dir / "q_features.npy"
    done_path = work_dir / "q_done.npy"
    features = _open_array(
        feature_path, (int(offsets[-1]), len(Q_FEATURE_NAMES)), np.float32
    )
    done = (
        np.load(done_path)
        if done_path.exists()
        else np.zeros(len(records), dtype=bool)
    )
    order = np.asarray(
        sorted(np.flatnonzero(~done), key=lambda index: int(lengths[index])),
        dtype=np.int64,
    )
    model.requires_grad_(False)
    model.eval()
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch, indices = _batch(records, tokenizer, indices, device)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits[:, :-1]
        token_ids = batch["labels"][:, 1:]
        valid = token_ids.ne(-100)
        positions = torch.cat(
            [
                torch.linspace(0.0, 1.0, int(lengths[index]), device=device)
                for index in indices
            ]
        )
        log_lengths = torch.cat(
            [
                torch.full(
                    (int(lengths[index]),),
                    float(np.log(lengths[index])),
                    device=device,
                )
                for index in indices
            ]
        )
        values = q_features_from_logits(
            logits[valid], token_ids[valid], positions, log_lengths
        ).cpu().numpy()
        cursor = 0
        for index in indices:
            length = int(lengths[index])
            left, right = int(offsets[index]), int(offsets[index + 1])
            features[left:right] = values[cursor : cursor + length]
            cursor += length
            done[index] = True
        features.flush()
        _atomic_save(done_path, done)
        print(
            json.dumps({"stage": "draft_features", "records": int(done.sum())}),
            flush=True,
        )
        del logits, values, batch, valid
    return features


@torch.inference_mode()
def _collect_accept_bits(
    model: Any,
    records: list[Any],
    tokenizer: Any,
    q_features: np.ndarray,
    lengths: np.ndarray,
    offsets: np.ndarray,
    work_dir: Path,
    device: torch.device,
    batch_size: int,
    acceptance_seed: int,
) -> np.memmap:
    bits_path = work_dir / "accept_bits.npy"
    done_path = work_dir / "bits_done.npy"
    bits = _open_array(bits_path, (int(offsets[-1]), 1, 2), np.uint8)
    done = (
        np.load(done_path)
        if done_path.exists()
        else np.zeros(len(records), dtype=bool)
    )
    order = np.asarray(
        sorted(np.flatnonzero(~done), key=lambda index: int(lengths[index])),
        dtype=np.int64,
    )
    model.requires_grad_(False)
    model.eval()
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        batch, indices = _batch(records, tokenizer, indices, device)
        logits = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits[:, :-1]
        token_ids = batch["labels"][:, 1:]
        valid = token_ids.ne(-100)
        # This tensor is the verifier-internal value. It is consumed in this
        # loop and never copied to a file or returned from the function.
        verifier_logp = selected_token_logprobs(logits, token_ids)[valid].cpu().numpy()
        cursor = 0
        for index in indices:
            length = int(lengths[index])
            left, right = int(offsets[index]), int(offsets[index + 1])
            local_logp = verifier_logp[cursor : cursor + length]
            local_logq = np.asarray(q_features[left:right, 0])
            alpha = np.exp(np.minimum(0.0, local_logp - local_logq))
            uniforms = _record_uniforms(
                acceptance_seed, int(index), length, 2
            )
            bits[left:right, 0, :] = uniforms < alpha[:, None]
            cursor += length
            done[index] = True
        bits.flush()
        _atomic_save(done_path, done)
        print(
            json.dumps({"stage": "accept_bits", "records": int(done.sum())}),
            flush=True,
        )
        del logits, verifier_logp, batch, valid
    return bits


def _collection_contract(
    run_dir: Path,
    run_manifest_sha: str,
    record_ids: np.ndarray,
    lengths: np.ndarray,
    acceptance_seed: int,
) -> dict[str, Any]:
    record_digest = hashlib.sha256()
    for record_id in record_ids.astype(str):
        record_digest.update(record_id.encode("utf-8"))
        record_digest.update(b"\0")
    return {
        "schema": "deployment_collection_work_v1",
        "run_dir": str(run_dir),
        "run_manifest_sha256": run_manifest_sha,
        "record_ids_sha256": record_digest.hexdigest(),
        "records": len(record_ids),
        "tokens": int(lengths.sum()),
        "acceptance_seed": acceptance_seed,
        "budget": 2,
    }


def collect(args: argparse.Namespace) -> None:
    run_dir = resolve_run_dir(args.run_dir).resolve()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        validate_deployment_archive(output)
        print(json.dumps({"already_complete": str(output)}), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to collect language-model observations")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    set_seed(args.acceptance_seed)

    cfg = load_run_config(run_dir)
    cfg, data = prepare_deployment_scoring_records(run_dir, cfg)
    records = data.records
    lengths = np.asarray([len(record.response_ids) for record in records], dtype=np.int64)
    offsets = np.r_[0, lengths.cumsum()]
    run_manifest = run_dir / "results.json"
    run_manifest_sha = sha256_file(run_manifest)
    work_dir = output.with_name(f".{output.name}.collection")
    work_dir.mkdir(parents=True, exist_ok=True)
    contract = _collection_contract(
        run_dir,
        run_manifest_sha,
        data.record_ids,
        lengths,
        args.acceptance_seed,
    )
    contract_path = work_dir / "COLLECTION.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise RuntimeError("existing deployment collection has a different contract")
    else:
        _write_json(contract_path, contract)

    draft = load_draft_model(
        run_dir,
        cfg.draft_model,
        "draft_auxiliary_distilled",
        device,
        attn_implementation=args.attn_implementation,
    )
    try:
        q_features = _collect_draft_features(
            draft,
            records,
            data.tokenizer,
            lengths,
            offsets,
            work_dir,
            device,
            args.draft_batch_size,
        )
    finally:
        del draft
        gc.collect()
        torch.cuda.empty_cache()

    target = load_finetuned_model(
        run_dir,
        cfg.target_model,
        device,
        attn_implementation=args.attn_implementation,
    )
    try:
        bits = _collect_accept_bits(
            target,
            records,
            data.tokenizer,
            q_features,
            lengths,
            offsets,
            work_dir,
            device,
            args.target_batch_size,
            args.acceptance_seed,
        )
    finally:
        del target
        gc.collect()
        torch.cuda.empty_cache()

    draft_checkpoint = resolve_checkpoint_path(
        run_dir, "draft_auxiliary_distilled"
    )
    target_checkpoint = resolve_checkpoint_path(run_dir, "target")
    provenance = {
        "run_dir": str(run_dir),
        "run_manifest_sha256": run_manifest_sha,
        "benchmark": cfg.benchmark,
        "target_epochs": cfg.target_epochs,
        "pool_sha256": str(
            json.loads(run_manifest.read_text(encoding="utf-8"))["data"]["pool_sha256"]
        ),
        "split_seed": cfg.data_seed,
        "acceptance_seed": args.acceptance_seed,
        "language_models_frozen": True,
        "query_budget": 2,
        "draft_checkpoint": {
            "role": "draft_auxiliary_distilled",
            "path": str(draft_checkpoint),
            "fingerprint": checkpoint_fingerprint(draft_checkpoint),
        },
        "target_checkpoint": {
            "role": "target_verifier",
            "path": str(target_checkpoint),
            "fingerprint": checkpoint_fingerprint(target_checkpoint),
        },
    }
    write_deployment_archive(
        output,
        logq=np.asarray(q_features[:, :1]),
        bits=np.asarray(bits),
        lengths=lengths,
        labels=data.labels,
        record_ids=data.record_ids,
        record_roles=data.record_roles,
        draft_features=np.asarray(q_features[:, 1:4]),
        provenance=provenance,
    )
    validate_deployment_archive(output)
    del q_features, bits
    shutil.rmtree(work_dir)
    print(json.dumps({"completed": str(output)}), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--draft-batch-size", type=int, default=4)
    parser.add_argument("--target-batch-size", type=int, default=1)
    parser.add_argument("--acceptance-seed", type=int, default=20260914)
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="sdpa"
    )
    return parser.parse_args()


def main() -> None:
    collect(parse_args())


if __name__ == "__main__":
    main()
