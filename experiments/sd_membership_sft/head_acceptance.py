"""Draft-acceptance comparison for protocol (EAGLE-3 / MTP) drafter heads.

For one protocol-track run directory this replays the newstection split,
samples a fixed number of records per class, and measures the exact
speculative-decoding acceptance ``sum_v min(p(v), q(v)) = 1 - TV(p, q)``
plus greedy top-1 agreement for every (head, verifier) cell, all
teacher-forced with oracle-prefix conditioning (the head sees the true
continuation prefix, which upper-bounds deployed acceptance):

- EAGLE pairs (draft-vocab head): ``frozen x {base, tuned}`` plus
  ``aux_kd x tuned`` and ``member_kd x tuned``. Draft distributions are
  embedded into the target vocabulary through ``speculator.d2t``; positions
  whose ground-truth token lies outside the draft vocabulary never accept,
  and the covered fraction is reported alongside.
- MTP pairs (full-vocab head): ``pre_head x {base, tuned}``,
  ``aux_kd x tuned``, ``joint x tuned``. The head's position ``t`` predicts
  ``x_{t+2}`` and is compared with the verifier distribution at ``t+1``.

Per-record values go to ``acceptance_heads.npz``; means with bootstrap CIs
and paired deltas go to ``acceptance_heads.json``.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .acceptance_comparison import bootstrap_mean_ci, sample_records
from .nart_data import build_nart_split, pool_path as nart_pool_path
from .protocol_heads import (
    eagle3_target_layer_ids,
    load_eagle3_speculator,
    load_mtp_speculator,
)
from .protocol_ft import PAIR_MODELS
from .data import collate_sft, make_sft_example
from .generalization import load_finetuned_model
from .training import _autocast, load_causal_lm

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pair", required=True, choices=list(PAIR_MODELS))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--per-class", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# per-position acceptance
# ---------------------------------------------------------------------------


@torch.no_grad()
def eagle_head_acceptance(
    speculator: Any,
    verifier: Any,
    records: list[Any],
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Exact acceptance + top-1 agreement + coverage for one EAGLE cell.

    The checkpoint's forward maps draft-vocab scores into the target
    vocabulary (d2t scatter); if a variant returns raw draft-vocab logits
    instead, they are embedded through ``speculator.d2t`` here.
    """
    base = speculator.get_base_model() if hasattr(speculator, "get_base_model") else speculator
    d2t = base.d2t.long()
    layers = eagle3_target_layer_ids(verifier)
    rows: dict[str, list[np.ndarray]] = {name: [] for name in (
        "exact_acceptance", "top1_agreement", "coverage",
    )}
    examples = [make_sft_example(record, tokenizer) for record in records]
    for start in range(0, len(examples), batch_size):
        batch = collate_sft(examples[start : start + batch_size], int(tokenizer.pad_token_id))
        batch = {k: v.to(device) for k, v in batch.items()}
        with _autocast(device):
            verifier_output = verifier(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            teacher_logits = verifier_output.logits[:, :-1].float()
            connector = torch.cat(
                [verifier_output.hidden_states[index] for index in layers], dim=-1
            )
            draft_logits = speculator(
                input_ids=batch["input_ids"],
                hidden_states=connector,
                attention_mask=batch["attention_mask"],
                return_dict=True,
            ).logits[:, :-1].float()
        labels = batch["labels"][:, 1:]
        valid = labels.ne(-100)
        p = F.softmax(teacher_logits, dim=-1)
        if draft_logits.shape[-1] == p.shape[-1]:
            q_full = F.softmax(draft_logits, dim=-1)
        else:
            q_draft = F.softmax(draft_logits, dim=-1)
            q_full = torch.zeros_like(p)
            q_full.index_copy_(2, d2t, q_draft)
        acceptance = torch.minimum(p, q_full).sum(dim=-1)
        agreement = q_full.argmax(dim=-1).eq(p.argmax(dim=-1)).float()
        truth = labels.clamp_min(0)
        covered = (q_full.gather(-1, truth.unsqueeze(-1)).squeeze(-1) > 0).float()
        for row in range(len(labels)):
            mask = valid[row]
            rows["exact_acceptance"].append(acceptance[row, mask].mean().cpu().numpy())
            rows["top1_agreement"].append(agreement[row, mask].mean().cpu().numpy())
            rows["coverage"].append(covered[row, mask].mean().cpu().numpy())
    return {name: np.asarray(values, dtype=np.float32) for name, values in rows.items()}


@torch.no_grad()
def native_mtp_acceptance(
    speculator: Any,
    verifier: Any,
    records: list[Any],
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Acceptance for a converted native-MTP speculator (batch-1 head API).

    The verifier runs batched; each row is trimmed to its true length and fed
    through the speculator (its query length is seq_len - steps - 1, so a
    padding mask would be incompatible). Step-0 logits predict x_{t+2}.
    """
    rows: dict[str, list[np.ndarray]] = {
        name: [] for name in ("exact_acceptance", "top1_agreement")
    }
    examples = [make_sft_example(record, tokenizer) for record in records]
    for start in range(0, len(examples), batch_size):
        batch = collate_sft(examples[start : start + batch_size], int(tokenizer.pad_token_id))
        batch = {k: v.to(device) for k, v in batch.items()}
        with _autocast(device):
            verifier_output = verifier(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        teacher_logits_full = verifier_output.logits.float()
        for row in range(len(examples[start : start + batch_size])):
            length = int(batch["attention_mask"][row].sum())
            with _autocast(device):
                logits_list, _loss, _metrics = speculator(
                    input_ids=batch["input_ids"][row : row + 1, :length],
                    hidden_states=verifier_output.hidden_states[-1][row : row + 1, :length],
                    attention_mask=None,
                    return_dict=True,
                )
            head_logits = logits_list[0].float()[0]  # [L-2, V], position t -> x_{t+2}
            teacher = teacher_logits_full[row, 1 : 1 + head_logits.shape[0]]
            labels = batch["labels"][row, 2 : 2 + head_logits.shape[0]]
            valid = labels.ne(-100)
            p = F.softmax(teacher, dim=-1)
            q = F.softmax(head_logits, dim=-1)
            acceptance = torch.minimum(p, q).sum(dim=-1)
            agreement = q.argmax(dim=-1).eq(p.argmax(dim=-1)).float()
            rows["exact_acceptance"].append(acceptance[valid].mean().cpu().numpy())
            rows["top1_agreement"].append(agreement[valid].mean().cpu().numpy())
    return {name: np.asarray(values, dtype=np.float32) for name, values in rows.items()}


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def _summarize(values: dict[str, np.ndarray], labels: np.ndarray, repeats: int) -> dict:
    summary = {}
    for metric, array in values.items():
        rows = {"overall": bootstrap_mean_ci(array, 500, 17)}
        for index, name in enumerate(("member", "nonmember", "auxiliary")):
            rows[name] = bootstrap_mean_ci(array[labels == index], 500, 18 + index)
        summary[metric] = rows
    return summary


def _paired_delta(left: np.ndarray, right: np.ndarray, repeats: int) -> dict:
    return bootstrap_mean_ci(left - right, repeats, 19)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir if args.run_dir.is_absolute() else ROOT / args.run_dir
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    tokenizer = AutoTokenizer.from_pretrained(PAIR_MODELS[args.pair]["target"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    seed = args.seed if args.seed is not None else 20260824 + 13
    run_config = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))["config"]
    benchmark = run_config.get("benchmark", "newstection")
    members, nonmembers, auxiliary, _meta = build_nart_split(
        benchmark,
        ROOT / Path(run_config["pool_path"]) if Path(run_config["pool_path"]).is_absolute() else ROOT / Path(run_config["pool_path"]),
        tokenizer,
        int(run_config.get("n_per_class", 2000)),
        int(run_config.get("n_aux", 2000)),
        int(run_config.get("data_seed", 20260824)),
    )
    records, labels = sample_records(
        {"member": members, "nonmember": nonmembers, "auxiliary": auxiliary},
        args.per_class,
        seed,
    )

    base_target = load_causal_lm(PAIR_MODELS[args.pair]["target"], device)
    tuned_target = load_finetuned_model(
        run_dir, PAIR_MODELS[args.pair]["target"], device
    )

    is_eagle = args.pair.endswith("eagle3")
    cells: dict[str, dict[str, np.ndarray]] = {}
    if is_eagle:
        published = PAIR_MODELS[args.pair]["speculator"]
        for head_name, head_source in (
            ("frozen", published),
            ("aux_kd", run_dir / "heads" / "aux_kd_head"),
            ("member_kd", run_dir / "heads" / "member_kd_head"),
        ):
            if head_name != "frozen" and not head_source.exists():
                print(f"skip missing head {head_source}", flush=True)
                continue
            head = load_eagle3_speculator(str(head_source), device)
            for verifier_name, verifier in (
                ("base", base_target),
                ("tuned", tuned_target),
            ):
                if head_name != "frozen" and verifier_name == "base":
                    continue  # adapted heads are only paired with their tuned target
                key = f"{head_name}__vs__{verifier_name}"
                started = time.time()
                cells[key] = eagle_head_acceptance(
                    head, verifier, records, tokenizer, device, args.batch_size
                )
                print(
                    f"{key}: acceptance={cells[key]['exact_acceptance'].mean():.4f} "
                    f"({time.time() - started:.0f}s)",
                    flush=True,
                )
            del head
            gc.collect()
            torch.cuda.empty_cache()
    else:
        for head_name, head_dir in (
            ("pre", run_dir / "heads" / "pre_head"),
            ("joint", run_dir / "heads" / "joint_head"),
            ("aux_kd", run_dir / "heads" / "aux_kd_head"),
        ):
            if not head_dir.exists():
                print(f"skip missing head {head_dir}", flush=True)
                continue
            for verifier_name, verifier in (
                ("base", base_target),
                ("tuned", tuned_target),
            ):
                if head_name != "pre" and verifier_name == "base":
                    continue
                key = f"{head_name}__vs__{verifier_name}"
                started = time.time()
                head = load_mtp_speculator(head_dir, device)
                cells[key] = native_mtp_acceptance(
                    head, verifier, records, tokenizer, device, args.batch_size
                )
                print(
                    f"{key}: acceptance={cells[key]['exact_acceptance'].mean():.4f} "
                    f"({time.time() - started:.0f}s)",
                    flush=True,
                )
                del head
            gc.collect()
            torch.cuda.empty_cache()

    results = {key: _summarize(values, labels, args.bootstrap_repeats) for key, values in cells.items()}
    deltas = {}
    frozen_base = cells.get("frozen__vs__base") or cells.get("pre__vs__base")
    frozen_tuned = cells.get("frozen__vs__tuned") or cells.get("pre__vs__tuned")
    if frozen_base is not None and frozen_tuned is not None:
        deltas["frozen_tuned_minus_base"] = _paired_delta(
            frozen_tuned["exact_acceptance"], frozen_base["exact_acceptance"], args.bootstrap_repeats
        )
    for adapted in ("aux_kd", "member_kd", "joint"):
        cell = cells.get(f"{adapted}__vs__tuned")
        if cell is not None and frozen_tuned is not None:
            deltas[f"{adapted}_minus_frozen_on_tuned"] = _paired_delta(
                cell["exact_acceptance"],
                frozen_tuned["exact_acceptance"],
                args.bootstrap_repeats,
            )

    artifact = {
        "run_dir": str(run_dir),
        "pair": args.pair,
        "protocol": {
            "per_class": args.per_class,
            "records": len(records),
            "batch_size": args.batch_size,
            "seed": seed,
            "conditioning": "teacher-forced oracle prefix",
            "acceptance_definition": "sum_v min(p, q) = 1 - TV(p, q)",
        },
        "results": results,
        "paired_deltas": deltas,
    }
    (run_dir / "acceptance_heads.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    np.savez_compressed(
        run_dir / "acceptance_heads.npz",
        labels=labels,
        **{f"{key}__{metric}": values[metric] for key, values in cells.items() for metric in values},
    )
    headline = {
        key: float(values["exact_acceptance"].mean()) for key, values in cells.items()
    }
    print(json.dumps({"cells": headline, "deltas": {
        k: v["mean"] for k, v in deltas.items()
    }}, indent=2), flush=True)


if __name__ == "__main__":
    main()
