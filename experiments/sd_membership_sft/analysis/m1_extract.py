"""Extract the Qwen3-1.7B Q/H feature cache used by M1.

Only the draft model is loaded here.  The saved p/q probability archive is a
read-only input: the newly extracted candidate log-probability is checked
against it with a preregistered tolerance before any feature cache is
released.  Selected decoder blocks are summarized inside their forward hook,
so raw hidden states are not accumulated on the GPU.

Example (one physical GPU):

    CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m \
      experiments.sd_membership_sft.m1_extract \
      --run-dir experiments/results/sft_runs/wikitection_qwen3_8b_epoch3 \
      --role draft_auxiliary_distilled --gpu 0 \
      --output-dir experiments/results/sft_runs/m1_conditional/wikitection_epoch3/draft_auxiliary_distilled
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from experiments.sd_membership_sft.datasets.data import SFTRecord, collate_sft, make_sft_example, prompt_prefix_ids
from experiments.sd_membership_sft.finetune.generalization import load_draft_model, load_run_config
from experiments.sd_membership_sft.analysis.m1_features import (
    ACTIVATION_FEATURE_NAMES,
    ACTIVATION_STAT_NAMES,
    M1_FEATURE_NAMES,
    Q_FEATURE_NAMES,
    SELECTED_BLOCKS,
    activation_statistics,
    q_features_from_logits,
)
from experiments.sd_membership_sft.analysis.pq_gap_mia import LOGSUMEXP_SEQUENCE_CHUNK, selected_token_logprobs
from experiments.sd_membership_sft.core.scoring_common import (
    prepare_scoring_records,
    resolve_checkpoint_path,
    resolve_run_dir,
    role_provenance,
)
from experiments.sd_membership_sft.finetune.training import set_seed


from experiments.paths import ROOT
EXTRACTOR_VERSION = "m1-extract-v2-provenance-chain"
ALIGN_ATOL = 1e-4
ALIGN_RTOL = 1e-5
ROLE_CHOICES = ("draft_auxiliary_distilled", "draft_member_sft")


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _path_fingerprint(path: Path) -> dict[str, Any]:
    """Hash a checkpoint or file without following files outside its root."""

    path = path.resolve()
    if path.is_file():
        return {
            "kind": "file",
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
    if not path.is_dir():
        raise FileNotFoundError(path)
    files: list[dict[str, Any]] = []
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        relative = child.relative_to(path).as_posix()
        sha = _sha256_file(child)
        size = child.stat().st_size
        files.append({"path": relative, "size": size, "sha256": sha})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha.encode("ascii"))
        digest.update(b"\n")
    return {
        "kind": "directory",
        "path": str(path),
        "files": files,
        "manifest_sha256": digest.hexdigest(),
    }


def _tokenizer_fingerprint(tokenizer: Any) -> dict[str, Any]:
    vocabulary = tokenizer.get_vocab()
    vocabulary_hash = hashlib.sha256(
        json.dumps(vocabulary, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "")),
        "class": type(tokenizer).__name__,
        "vocab_size": int(tokenizer.vocab_size),
        "eos_token_id": (
            int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None
        ),
        "pad_token_id": (
            int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None
        ),
        "vocabulary_sha256": vocabulary_hash,
    }


def _records_fingerprint(records: list[SFTRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.record_id.encode("utf-8"))
        digest.update(b"\0")
        for token_id in tuple(record.prompt_ids or ()) + tuple(record.response_ids):
            digest.update(int(token_id).to_bytes(4, "little", signed=False))
        digest.update(b"\xff")
    return digest.hexdigest()


def _find_decoder_layers(model: torch.nn.Module) -> tuple[torch.nn.ModuleList, str]:
    """Find Qwen/GPT-NeoX decoder blocks through PEFT or LM wrappers."""

    expected = int(getattr(getattr(model, "config", None), "num_hidden_layers", -1))
    candidates: list[tuple[str, torch.nn.ModuleList]] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.ModuleList) or len(module) != expected:
            continue
        if not module:
            continue
        first = module[0]
        if (hasattr(first, "self_attn") or hasattr(first, "attention")) and hasattr(first, "input_layernorm"):
            candidates.append((name, module))
    if len(candidates) != 1:
        names = [name for name, _ in candidates]
        raise RuntimeError(
            "Could not uniquely identify decoder layers; "
            f"expected {expected} layers, candidates={names}"
        )
    return candidates[0][1], candidates[0][0]


def selected_blocks_for_model(model):
    return (5, 11, 17, 23) if getattr(model.config, "model_type", None) == "gpt_neox" else SELECTED_BLOCKS


def _validate_checkpoint(model: torch.nn.Module) -> tuple[torch.nn.ModuleList, str]:
    config = getattr(model, "config", None)
    layers, layer_path = _find_decoder_layers(model)
    actual_layers = int(getattr(config, "num_hidden_layers", -1))
    hidden_size = int(getattr(config, "hidden_size", -1))
    expected_layers = 24 if getattr(config, "model_type", None) == "gpt_neox" else 28
    if actual_layers != expected_layers or hidden_size != 2048:
        raise RuntimeError(
            "M1 supports Qwen3-1.7B (28 layers) or Pythia-1.4B (24 layers), "
            f"hidden size 2048, got layers={actual_layers}, hidden_size={hidden_size}"
        )
    if max(selected_blocks_for_model(model)) >= len(layers):
        raise RuntimeError(f"Selected block exceeds checkpoint depth: {len(layers)}")
    return layers, layer_path


def _example_without_eos(record: SFTRecord, tokenizer: Any) -> dict[str, list[int]]:
    prefix_ids = prompt_prefix_ids(record, tokenizer)
    response_ids = list(record.response_ids)
    return {
        "input_ids": prefix_ids + response_ids,
        "labels": [-100] * len(prefix_ids) + response_ids,
    }


def _build_position_metadata(
    records: list[SFTRecord], tokenizer: Any, include_eos: bool
) -> tuple[
    list[dict[str, list[int]]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    examples = [
        make_sft_example(record, tokenizer)
        if include_eos
        else _example_without_eos(record, tokenizer)
        for record in records
    ]
    lengths = np.asarray(
        [len(record.response_ids) + int(include_eos) for record in records], dtype=np.int64
    )
    offsets = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    token_ids = np.empty(int(offsets[-1]), dtype=np.int64)
    prediction_positions = np.empty(int(offsets[-1]), dtype=np.int64)
    input_positions = np.empty(int(offsets[-1]), dtype=np.int64)
    eos_mask = np.zeros(int(offsets[-1]), dtype=bool)
    for index, (record, example, start, end) in enumerate(
        zip(records, examples, offsets[:-1], offsets[1:])
    ):
        prefix_length = len(prompt_prefix_ids(record, tokenizer))
        response = list(record.response_ids)
        if include_eos:
            response.append(int(tokenizer.eos_token_id))
        if len(response) != int(end - start):
            raise AssertionError("response metadata does not match constructed example")
        token_ids[int(start) : int(end)] = response
        input_positions[int(start) : int(end)] = prefix_length + np.arange(len(response))
        prediction_positions[int(start) : int(end)] = prefix_length - 1 + np.arange(
            len(response)
        )
        if include_eos:
            eos_mask[int(end) - 1] = True
        # The assertion catches accidental changes to make_sft_example's mask.
        labels = np.asarray(example["labels"], dtype=np.int64)
        if not np.array_equal(labels[prefix_length:], np.asarray(response, dtype=np.int64)):
            raise AssertionError("response labels are not aligned with input positions")
    return examples, lengths, offsets, token_ids, prediction_positions, input_positions, eos_mask


def _load_probability_reference(
    probability_dir: Path,
    role: str,
    records: list[SFTRecord],
    labels: np.ndarray,
    record_ids: np.ndarray,
    include_eos: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    token_path = probability_dir / "pq_gap_token_logps.npz"
    score_path = probability_dir / "pq_gap_scores.npz"
    if not token_path.exists() or not score_path.exists():
        raise FileNotFoundError(
            f"Missing probability cache under {probability_dir}: {token_path}, {score_path}"
        )
    token_data = np.load(token_path, allow_pickle=False)
    score_data = np.load(score_path, allow_pickle=False)
    for key in ("lengths", role):
        if key not in token_data.files:
            raise RuntimeError(f"Probability cache is missing {key}: {token_path}")
    for key in ("labels", "record_ids"):
        if key not in score_data.files:
            raise RuntimeError(f"Score cache is missing {key}: {score_path}")
    cached_labels_all = np.asarray(score_data["labels"], dtype=np.int64)
    cached_ids_all = np.asarray(score_data["record_ids"])
    if len(cached_labels_all) < len(labels) or not np.array_equal(cached_labels_all[: len(labels)], labels):
        raise RuntimeError("Probability-cache labels disagree with the rebuilt scoring split")
    if len(cached_ids_all) < len(record_ids) or not np.array_equal(cached_ids_all[: len(record_ids)], record_ids):
        raise RuntimeError("Probability-cache record_ids disagree with the rebuilt scoring split")
    cached_lengths_all = np.asarray(token_data["lengths"], dtype=np.int64)
    if len(cached_lengths_all) < len(records):
        raise RuntimeError("Probability-cache record count is smaller than the requested scoring subset")
    cached_lengths = cached_lengths_all[: len(records)]
    full_lengths = np.asarray([len(record.response_ids) + 1 for record in records], dtype=np.int64)
    if not np.array_equal(cached_lengths, full_lengths):
        raise RuntimeError("Probability-cache lengths disagree with response+EOS lengths")
    cached_values_all = np.asarray(token_data[role], dtype=np.float32)
    prefix_tokens = int(cached_lengths.sum())
    if len(cached_values_all) < prefix_tokens:
        raise RuntimeError("Probability-cache token values are shorter than the requested subset")
    cached_values = cached_values_all[:prefix_tokens]
    if include_eos:
        expected_lengths = cached_lengths
        expected_values = cached_values
    else:
        expected_lengths = cached_lengths - 1
        pieces: list[np.ndarray] = []
        offset = 0
        for length in cached_lengths:
            pieces.append(cached_values[offset : offset + int(length) - 1])
            offset += int(length)
        expected_values = np.concatenate(pieces)
    return expected_values, expected_lengths, {
        "path": str(token_path.resolve()),
        "sha256": _sha256_file(token_path),
        "scores_path": str(score_path.resolve()),
        "scores_sha256": _sha256_file(score_path),
    }


@torch.inference_mode()
def extract_qh_features(
    model: torch.nn.Module,
    examples: list[dict[str, list[int]]],
    lengths: np.ndarray,
    offsets: np.ndarray,
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
    q_output: np.ndarray,
    h_output: np.ndarray,
    row_chunk: int = 128,
    empty_cache_each_batch: bool = False,
    selected_blocks: tuple[int, ...] | None = None,
    progress: Any = None,
) -> None:
    """Run the draft forward pass and fill flat FP32 Q/H arrays."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]))
    if selected_blocks is None:
        layers, _ = _validate_checkpoint(model)
        selected_blocks = selected_blocks_for_model(model)
    else:
        layers, _ = _find_decoder_layers(model)
    if len(selected_blocks) != 4 or len(set(selected_blocks)) != 4 or min(selected_blocks) < 0 or max(selected_blocks) >= len(layers):
        raise ValueError("M1 requires four distinct in-range decoder blocks")
    handles: list[Any] = []
    capture: dict[int, torch.Tensor] = {}
    valid_mask: torch.Tensor | None = None

    def make_hook(block: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            if valid_mask is None:
                raise RuntimeError("activation hook ran before the batch mask was set")
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
                raise RuntimeError(
                    f"decoder block {block + 1} did not return [batch, sequence, hidden]"
                )
            selected = hidden[:, :-1, :][valid_mask]
            # Move only the ten-value summary to CPU; do not retain raw states.
            capture[block] = activation_statistics(selected).cpu()

        return hook

    try:
        for block in selected_blocks:
            handles.append(layers[block].register_forward_hook(make_hook(block)))
        starts = list(range(0, len(order), batch_size))
        if progress is not None:
            starts = progress.track(starts, "draft Q/H extraction", unit="batches")
        for start in starts:
            indices = order[start : start + batch_size]
            rows = [examples[index] for index in indices]
            batch = {
                key: value.to(device)
                for key, value in collate_sft(rows, int(tokenizer.pad_token_id)).items()
            }
            valid_mask = batch["labels"][:, 1:].ne(-100)
            capture.clear()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            logits = outputs.logits[:, :-1]
            if len(capture) != len(selected_blocks):
                raise RuntimeError(
                    f"Expected {len(selected_blocks)} activation hooks, got {len(capture)}"
                )
            flat_ids = batch["labels"][:, 1:][valid_mask]
            relative_parts: list[torch.Tensor] = []
            length_parts: list[torch.Tensor] = []
            for index in indices:
                count = int(lengths[index])
                relative_parts.append(
                    torch.arange(count, device=device, dtype=torch.float32)
                    / float(max(count - 1, 1))
                )
                length_parts.append(
                    torch.full(
                        (count,), float(np.log(count)), device=device, dtype=torch.float32
                    )
                )
            relative = torch.cat(relative_parts)
            log_length = torch.cat(length_parts)
            if len(flat_ids) != len(relative):
                raise RuntimeError("response mask and saved position metadata disagree")
            q_features = q_features_from_logits(
                logits[valid_mask],
                flat_ids,
                relative,
                log_length,
                row_chunk=row_chunk,
            ).cpu().numpy()
            h_features = torch.cat([capture[block] for block in selected_blocks], dim=-1).numpy()
            cursor = 0
            for index in indices:
                count = int(lengths[index])
                destination = slice(int(offsets[index]), int(offsets[index + 1]))
                q_output[destination] = q_features[cursor : cursor + count]
                h_output[destination] = h_features[cursor : cursor + count]
                cursor += count
            if cursor != len(q_features):
                raise RuntimeError("failed to consume the complete batch feature output")
            del outputs, logits, q_features, h_features, batch, valid_mask
            valid_mask = None
            if empty_cache_each_batch and device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--role", choices=ROLE_CHOICES, required=True)
    parser.add_argument("--probability-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--row-chunk", type=int, default=128)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--without-eos", action="store_true")
    parser.add_argument(
        "--rebuild-q-cache",
        action="store_true",
        help="On a failed historical-q check, write a new p+fresh-q cache with explicit provenance (full run only).",
    )
    parser.add_argument("--empty-cache-each-batch", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="eager"
    )
    parser.add_argument("--seed", type=int, default=20260909)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; M1 extraction requires an A100 host")
    run_dir = resolve_run_dir(args.run_dir).resolve()
    cfg, prepared = prepare_scoring_records(run_dir)
    records = prepared.records
    labels = prepared.labels
    record_ids = prepared.record_ids
    if args.max_records is not None:
        if args.max_records <= 0:
            raise ValueError("--max-records must be positive")
        records = records[: args.max_records]
        labels = labels[: args.max_records]
        record_ids = record_ids[: args.max_records]
    default_output_dir = (
        ROOT
        / "experiments/results/sft_runs/m1_conditional"
        / f"{cfg.benchmark}_epoch{cfg.target_epochs}"
        / args.role
    )
    if args.without_eos:
        default_output_dir = default_output_dir / "features_without_eos"
    output_dir = (
        _resolve(args.output_dir) if args.output_dir is not None else default_output_dir
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "feature_manifest.json"
    if manifest_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite existing M1 cache: {manifest_path}")

    probability_dir = (
        _resolve(args.probability_dir)
        if args.probability_dir is not None
        else ROOT
        / "experiments/results/sft_runs/pq_directional"
        / f"{cfg.benchmark}_epoch{cfg.target_epochs}"
    ).resolve()
    include_eos = not args.without_eos
    examples, lengths, offsets, token_ids, prediction_positions, input_positions, eos_mask = (
        _build_position_metadata(records, prepared.tokenizer, include_eos)
    )
    cached_q, cached_lengths, probability_provenance = _load_probability_reference(
        probability_dir,
        args.role,
        records,
        labels,
        record_ids,
        include_eos,
    )
    if not np.array_equal(lengths, cached_lengths):
        raise RuntimeError("constructed feature lengths disagree with probability cache")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    set_seed(args.seed)
    model = load_draft_model(
        run_dir,
        cfg.draft_model,
        args.role,
        device,
        attn_implementation=args.attn_implementation,
    )
    layers, layer_path = _validate_checkpoint(model)
    selected_blocks = selected_blocks_for_model(model)
    activation_names = tuple(f"block{block + 1}_{stat}" for block in selected_blocks for stat in ACTIVATION_STAT_NAMES)
    model_config = getattr(model, "config", None)
    model_num_layers = int(getattr(model_config, "num_hidden_layers", len(layers)))
    model_hidden_size = int(getattr(model_config, "hidden_size", 2048))
    del layers
    total_tokens = int(offsets[-1])
    q_tmp = output_dir / "q.npy.partial"
    h_tmp = output_dir / "h.npy.partial"
    for path in (q_tmp, h_tmp):
        if path.exists():
            path.unlink()
    q_memmap = np.lib.format.open_memmap(
        q_tmp, mode="w+", dtype=np.float32, shape=(total_tokens, len(Q_FEATURE_NAMES))
    )
    h_memmap = np.lib.format.open_memmap(
        h_tmp,
        mode="w+",
        dtype=np.float32,
        shape=(total_tokens, len(ACTIVATION_FEATURE_NAMES)),
    )
    started = time.perf_counter()
    try:
        extract_qh_features(
            model,
            examples,
            lengths,
            offsets,
            prepared.tokenizer,
            device,
            args.batch_size,
            q_memmap,
            h_memmap,
            row_chunk=args.row_chunk,
            empty_cache_each_batch=args.empty_cache_each_batch,
        )
    finally:
        del model
        q_memmap.flush()
        h_memmap.flush()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    # Keep the memmap handles alive for the final flush/rename, while using
    # ordinary ndarray views for the numerical checks and provenance writes.
    q_array = np.asarray(q_memmap)
    h_array = np.asarray(h_memmap)
    q_error = np.asarray(q_array[:, 0], dtype=np.float64) - np.asarray(
        cached_q, dtype=np.float64
    )
    abs_error = np.abs(q_error)
    max_error = float(abs_error.max()) if len(abs_error) else 0.0
    quantile_error = {
        str(q): float(np.quantile(abs_error, q)) for q in (0.50, 0.90, 0.99, 1.0)
    }
    alignment_passed = bool(
        np.allclose(q_array[:, 0], cached_q, atol=ALIGN_ATOL, rtol=ALIGN_RTOL)
    )
    if not alignment_passed and not args.rebuild_q_cache:
        raise RuntimeError(
            "Fresh log q does not match the saved probability cache: "
            f"max_abs_error={max_error:.6g}, quantiles={quantile_error}, "
            f"atol={ALIGN_ATOL}, rtol={ALIGN_RTOL}"
        )
    q_memmap.flush()
    h_memmap.flush()
    (output_dir / "q.npy.partial").replace(output_dir / "q.npy")
    (output_dir / "h.npy.partial").replace(output_dir / "h.npy")
    np.save(output_dir / "record_ids.npy", np.asarray(record_ids))
    np.save(output_dir / "labels.npy", np.asarray(labels, dtype=np.int64))
    np.save(output_dir / "lengths.npy", lengths)
    np.save(output_dir / "offsets.npy", offsets)
    np.save(output_dir / "token_ids.npy", token_ids)
    np.save(output_dir / "input_positions.npy", input_positions)
    np.save(output_dir / "prediction_positions.npy", prediction_positions)
    np.save(output_dir / "eos_mask.npy", eos_mask)

    reconciled_probability_dir: Path | None = None
    if not alignment_passed:
        if args.max_records is not None:
            raise RuntimeError("--rebuild-q-cache requires a complete 4000-record extraction, not --max-records")
        source_token_data = np.load(probability_dir / "pq_gap_token_logps.npz", allow_pickle=False)
        source_scores_data = np.load(probability_dir / "pq_gap_scores.npz", allow_pickle=False)
        source_target = np.asarray(source_token_data["target"], dtype=np.float32)
        source_lengths = np.asarray(source_token_data["lengths"], dtype=np.int64)
        if include_eos:
            reconciled_target = source_target
            reconciled_lengths = source_lengths
        else:
            if not np.array_equal(source_lengths - 1, lengths):
                raise RuntimeError("cannot rebuild q cache because the source p cache is not complete")
            target_parts: list[np.ndarray] = []
            source_offset = 0
            for source_length in source_lengths:
                count = int(source_length)
                if count <= 1:
                    raise RuntimeError("cannot remove EOS from a source probability record with no response token")
                target_parts.append(source_target[source_offset : source_offset + count - 1])
                source_offset += count
            reconciled_target = np.concatenate(target_parts)
            reconciled_lengths = lengths
        if len(reconciled_target) != total_tokens:
            raise RuntimeError("cannot rebuild q cache because the source p cache is not complete")
        reconciled_probability_dir = output_dir / "reconciled_probability"
        reconciled_probability_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            reconciled_probability_dir / "pq_gap_token_logps.npz",
            lengths=reconciled_lengths,
            target=reconciled_target,
            **{args.role: np.asarray(q_array[:, 0], dtype=np.float32)},
        )
        np.savez_compressed(
            reconciled_probability_dir / "pq_gap_scores.npz",
            labels=np.asarray(source_scores_data["labels"], dtype=np.int64),
            record_ids=np.asarray(source_scores_data["record_ids"]),
        )
        reconciled_token_sha256 = _sha256_file(
            reconciled_probability_dir / "pq_gap_token_logps.npz"
        )
        reconciled_scores_sha256 = _sha256_file(
            reconciled_probability_dir / "pq_gap_scores.npz"
        )
        (reconciled_probability_dir / "pq_gap_provenance.json").write_text(
            json.dumps(
                {
                    "kind": "reconciled_q_cache",
                    "source_probability_dir": str(probability_dir),
                    "source_probability_sha256": probability_provenance["sha256"],
                    "source_scores_sha256": probability_provenance["scores_sha256"],
                    "reconciled_token_logps_sha256": reconciled_token_sha256,
                    "reconciled_scores_sha256": reconciled_scores_sha256,
                    "source_q_alignment": {
                        "passed": False,
                        "atol": ALIGN_ATOL,
                        "rtol": ALIGN_RTOL,
                        "max_abs_error": max_error,
                        "absolute_error_quantiles": quantile_error,
                    },
                    "q_source": "fresh draft checkpoint forward with FP32 logsumexp",
                    "q_role": args.role,
                    "target_source": (
                        "copied unchanged from historical probability cache"
                        if include_eos
                        else "copied from historical probability cache with appended EOS removed"
                    ),
                    "eos_included": include_eos,
                    "records": len(records),
                    "tokens": total_tokens,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    pool_path = cfg.pool_path if cfg.pool_path is not None else ROOT / "experiments/data/pools" / cfg.benchmark / "pool.jsonl"
    pool_path = _resolve(pool_path)
    pool_manifest_path = pool_path.with_suffix(".manifest.json")
    checkpoint_path = resolve_checkpoint_path(run_dir, args.role)
    manifest = {
        "extractor_version": EXTRACTOR_VERSION,
        "run_dir": str(run_dir),
        "benchmark": str(cfg.benchmark),
        "epoch": int(cfg.target_epochs),
        "role": args.role,
        "base_model": str(cfg.draft_model),
        "checkpoint": _path_fingerprint(checkpoint_path),
        "checkpoint_provenance": role_provenance(cfg, run_dir, args.role),
        "probability_cache": probability_provenance,
        "probability_cache_alignment": {
            "atol": ALIGN_ATOL,
            "rtol": ALIGN_RTOL,
            "max_abs_error": max_error,
            "absolute_error_quantiles": quantile_error,
            "passed": alignment_passed,
            "action": "reused_historical_q" if alignment_passed else "rebuild_q_cache",
            "reconciled_probability_dir": (
                str(reconciled_probability_dir.resolve())
                if reconciled_probability_dir is not None
                else None
            ),
        },
        "pool_manifest": {
            "path": str(pool_manifest_path.resolve()),
            "sha256": _sha256_file(pool_manifest_path),
        },
        "tokenizer": _tokenizer_fingerprint(prepared.tokenizer),
        "records": len(records),
        "record_ids_sha256": hashlib.sha256(
            "\n".join(str(value) for value in record_ids).encode("utf-8")
        ).hexdigest(),
        "records_token_fingerprint": _records_fingerprint(records),
        "labels": {"members": int(np.sum(labels == 1)), "nonmembers": int(np.sum(labels == 0))},
        "total_tokens": total_tokens,
        "lengths": {
            "min": int(lengths.min()),
            "max": int(lengths.max()),
            "sum": int(lengths.sum()),
        },
        "eos_included": include_eos,
        "candidate_token_definition": "labels[:, 1:] != -100; appended EOS included" if include_eos else "labels[:, 1:] != -100; appended EOS removed",
        "prediction_position_definition": "input token position j predicts with decoder output at j-1",
        "feature_names": {
            "q": list(Q_FEATURE_NAMES),
            "h": list(activation_names),
            "combined": list(Q_FEATURE_NAMES + activation_names),
        },
        "selected_blocks_zero_based": list(selected_blocks),
        "selected_blocks_one_based": [block + 1 for block in selected_blocks],
        "activation_definition": "decoder block output residual stream before final model normalization",
        "hook_module": layer_path,
        "model_config": {
            "num_hidden_layers": model_num_layers,
            "hidden_size": model_hidden_size,
            "vocab_size": int(prepared.tokenizer.vocab_size),
        },
        "dtype": "float32 feature arrays; model checkpoint loaded in bfloat16",
        "attention_backend": args.attn_implementation,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "host": platform.node(),
        "pid": os.getpid(),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "row_chunk": int(args.row_chunk),
        "max_records": args.max_records,
        "seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        ),
        "arrays": {
            "q": "q.npy [total_tokens, 6] float32",
            "h": "h.npy [total_tokens, 40] float32",
            "record_ids": "record_ids.npy",
            "labels": "labels.npy",
            "lengths": "lengths.npy",
            "offsets": "offsets.npy",
            "token_ids": "token_ids.npy",
            "input_positions": "input_positions.npy",
            "prediction_positions": "prediction_positions.npy",
            "eos_mask": "eos_mask.npy",
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "records": len(records),
                "tokens": total_tokens,
                "max_abs_logq_alignment_error": max_error,
                "seconds": manifest["seconds"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
