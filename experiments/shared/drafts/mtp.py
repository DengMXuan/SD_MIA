"""Frozen-target adaptation of Qwen3.5's native MTP head.

Each condition has four restartable stages:

- ``mtp-source`` exports the original, pinned native MTP layer.
- ``mtp-target`` performs full-parameter member SFT and saves the target.
- ``mtp-head --variant aux`` initializes from the original head and uses KD
  on auxiliary documents against the frozen SFT target.
- ``mtp-head --variant member`` independently initializes from the same
  original head and uses native MTP cross-entropy on member documents.

The old joint trunk/head path is intentionally absent: both comparison heads
must observe exactly the same frozen membership-bearing target checkpoint.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.shared.drafts.heads import ensure_mtp_conversion, load_mtp_speculator
from experiments.shared.data.data import collate_sft, make_sft_example
from experiments.shared.training.training import _autocast, load_causal_lm, set_seed, sft_train
from experiments.shared.drafts.common import PAIR_MODELS, build_parser, cached_snapshot, checkpoint_complete, device_for, load_split, partial_checkpoint_path, promote_checkpoint, resolve_output_dir, run_command, run_dir_for, save_pretrained_atomically, tokenizer_for, write_run_config


def parse_args() -> argparse.Namespace:
    parser = build_parser(__doc__, ["qwen35_9b_mtp"])
    parser.add_argument(
        "--source-head",
        type=Path,
        required=True,
        help="matrix-global immutable native-MTP export",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("mtp-source", help="export the pinned checkpoint's native MTP head")
    sub.add_parser("mtp-target", help="full-parameter member SFT of the target")
    head = sub.add_parser("mtp-head", help="adapt a fresh native MTP head")
    head.add_argument("--variant", choices=["aux", "member"], required=True)
    args = parser.parse_args()
    resolve_output_dir(args)
    return args


def _source_head_path(args: argparse.Namespace) -> Path:
    return args.source_head if args.source_head.is_absolute() else Path.cwd() / args.source_head


def cmd_mtp_source(args: argparse.Namespace) -> None:
    """Convert the original native layer without involving the SFT target."""
    device_for(args)
    source_dir = _source_head_path(args)
    if checkpoint_complete(source_dir):
        print(f"[skip] complete native MTP source: {source_dir}", flush=True)
        return
    model = PAIR_MODELS[args.pair]
    snapshot = cached_snapshot(model["target"], model["target_revision"])
    temporary = partial_checkpoint_path(source_dir)
    converted = ensure_mtp_conversion(
        str(snapshot), temporary, num_speculative_steps=1
    )
    promote_checkpoint(
        converted,
        source_dir,
        {
            "stage": "source_head",
            "pair": args.pair,
            "initialized_from": model["target"],
            "initialized_from_revision": model["target_revision"],
            "native_mtp_export": True,
            "training_updates": 0,
        },
    )
    write_run_config(
        args,
        {
            "stage": "source_head",
            "source_head": str(source_dir),
            "init": "pinned-native-export",
        },
    )
    print(json.dumps({"command": "mtp-source", "converted": str(source_dir)}), flush=True)


def cmd_mtp_target(args: argparse.Namespace) -> None:
    device = device_for(args)
    run_dir = run_dir_for(args)
    checkpoint = run_dir / "checkpoints" / "target"
    if checkpoint_complete(checkpoint):
        print(f"[skip] complete target checkpoint: {checkpoint}", flush=True)
        return
    tokenizer = tokenizer_for(args.pair)
    members, _nonmembers, _auxiliary, metadata = load_split(tokenizer, args)
    model = PAIR_MODELS[args.pair]
    target = load_causal_lm(
        model["target"],
        device,
        revision=model["target_revision"],
        local_files_only=True,
    )
    started = time.time()
    losses = sft_train(
        target,
        members,
        tokenizer,
        device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        seed=args.seed,
        label="MTP-line target SFT",
        optimizer_name="adamw8bit",
    )

    def writer(path: Path) -> None:
        target.save_pretrained(path)
        tokenizer.save_pretrained(path)

    save_pretrained_atomically(
        checkpoint,
        writer,
        {
            "stage": "target",
            "pair": args.pair,
            "base_model": model["target"],
            "base_revision": model["target_revision"],
            "seed": args.seed,
            "data_seed": args.data_seed,
            "epochs": args.epochs,
            "full_parameter_sft": True,
        },
    )
    write_run_config(
        args,
        {
            "stage": "target",
            "target_sft_loss": losses,
            "seconds": time.time() - started,
            "data": metadata,
        },
    )
    print(json.dumps({"command": "mtp-target", "losses": losses}, indent=2), flush=True)


def _load_frozen_target_and_fresh_head(
    args: argparse.Namespace, device: torch.device
) -> tuple[Any, Any]:
    run_dir = run_dir_for(args)
    source_dir = _source_head_path(args)
    if not checkpoint_complete(source_dir):
        raise RuntimeError(f"Native MTP source is incomplete: {source_dir}")
    from experiments.shared.training.generalization import load_finetuned_model

    target = load_finetuned_model(
        run_dir, PAIR_MODELS[args.pair]["target"], device
    )
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    target.eval()
    target.config.use_cache = False
    # Each stage is a separate process and always reloads this immutable source;
    # the auxiliary and member branches can therefore never inherit each other.
    speculator = load_mtp_speculator(
        source_dir,
        device,
        verifier_checkpoint=run_dir / "checkpoints" / "target",
    )
    speculator.train()
    return target, speculator


def _mtp_auxiliary_kd(
    args: argparse.Namespace,
    target: Any,
    speculator: Any,
    examples: list[dict[str, list[int]]],
    tokenizer: Any,
    device: torch.device,
) -> list[float]:
    rng = np.random.default_rng(args.seed)
    trainable = [parameter for parameter in speculator.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.head_lr)
    history: list[float] = []
    for update in range(args.head_updates):
        optimizer.zero_grad(set_to_none=True)
        update_losses: list[float] = []
        for _micro in range(args.head_grad_accum):
            indices = rng.integers(0, len(examples), size=args.head_batch_size)
            batch = collate_sft(
                [examples[int(index)] for index in indices],
                int(tokenizer.pad_token_id),
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad(), _autocast(device):
                trunk_output = target(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
            row_losses: list[torch.Tensor] = []
            for row in range(args.head_batch_size):
                length = int(batch["attention_mask"][row].sum())
                with _autocast(device):
                    logits_list, _native_loss, _metrics = speculator(
                        input_ids=batch["input_ids"][row : row + 1, :length],
                        hidden_states=trunk_output.hidden_states[-1][
                            row : row + 1, :length
                        ],
                        attention_mask=None,
                        return_dict=True,
                    )
                    # Depth-1 step-0 logits[:, t] predict x_{t+2}; the teacher
                    # distribution aligned to that event is trunk position t+1.
                    student = logits_list[0].float()
                    teacher = trunk_output.logits[
                        row : row + 1, 1 : 1 + student.shape[1]
                    ].float()
                    labels = batch["labels"][
                        row : row + 1, 2 : 2 + student.shape[1]
                    ]
                    valid = labels.ne(-100)
                    if not bool(valid.any()):
                        raise ValueError("MTP KD micro-batch has no response tokens")
                    kd = (
                        F.kl_div(
                            F.log_softmax(
                                student[valid] / args.kd_temperature, dim=-1
                            ),
                            F.softmax(
                                teacher[:, : student.shape[1]][valid]
                                / args.kd_temperature,
                                dim=-1,
                            ),
                            reduction="batchmean",
                        )
                        * args.kd_temperature**2
                    )
                row_losses.append(kd)
            micro_loss = torch.stack(row_losses).mean()
            (micro_loss / args.head_grad_accum).backward()
            update_losses.append(float(micro_loss.detach().cpu()))
            del trunk_output
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        history.append(float(np.mean(update_losses)))
        if (update + 1) % max(1, args.head_updates // 4) == 0:
            print(
                f"MTP auxiliary KD update {update + 1}/{args.head_updates}: "
                f"loss={history[-1]:.5f}",
                flush=True,
            )
    return history


def _mtp_member_ce(
    args: argparse.Namespace,
    target: Any,
    speculator: Any,
    examples: list[dict[str, list[int]]],
    tokenizer: Any,
    device: torch.device,
) -> list[float]:
    rng = np.random.default_rng(args.seed)
    trainable = [parameter for parameter in speculator.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.head_lr)
    history: list[float] = []
    for update in range(args.head_updates):
        optimizer.zero_grad(set_to_none=True)
        update_losses: list[float] = []
        for _micro in range(args.head_grad_accum):
            indices = rng.integers(0, len(examples), size=args.head_batch_size)
            batch = collate_sft(
                [examples[int(index)] for index in indices],
                int(tokenizer.pad_token_id),
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad(), _autocast(device):
                trunk_output = target(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
            row_losses: list[torch.Tensor] = []
            for row in range(args.head_batch_size):
                length = int(batch["attention_mask"][row].sum())
                with _autocast(device):
                    _logits, native_loss, _metrics = speculator(
                        input_ids=batch["input_ids"][row : row + 1, :length],
                        hidden_states=trunk_output.hidden_states[-1][
                            row : row + 1, :length
                        ],
                        attention_mask=None,
                        loss_mask=batch["labels"][
                            row : row + 1, :length
                        ].ne(-100),
                        return_dict=True,
                    )
                row_losses.append(native_loss)
            micro_loss = torch.stack(row_losses).mean()
            (micro_loss / args.head_grad_accum).backward()
            update_losses.append(float(micro_loss.detach().cpu()))
            del trunk_output
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        history.append(float(np.mean(update_losses)))
        if (update + 1) % max(1, args.head_updates // 4) == 0:
            print(
                f"MTP member CE update {update + 1}/{args.head_updates}: "
                f"loss={history[-1]:.5f}",
                flush=True,
            )
    return history


def cmd_mtp_head(args: argparse.Namespace) -> None:
    assert args.variant in ("aux", "member")
    device = device_for(args)
    run_dir = run_dir_for(args)
    head_dir = run_dir / "heads" / (
        "auxiliary_head" if args.variant == "aux" else "member_head"
    )
    if checkpoint_complete(head_dir):
        print(f"[skip] complete head checkpoint: {head_dir}", flush=True)
        return
    tokenizer = tokenizer_for(args.pair)
    members, _nonmembers, auxiliary, metadata = load_split(tokenizer, args)
    records = auxiliary if args.variant == "aux" else members
    examples = [make_sft_example(record, tokenizer) for record in records]
    target, speculator = _load_frozen_target_and_fresh_head(args, device)
    set_seed(args.seed)
    if args.variant == "aux":
        history = _mtp_auxiliary_kd(
            args, target, speculator, examples, tokenizer, device
        )
        objective = "temperature-kl"
        temperature: float | None = args.kd_temperature
    else:
        history = _mtp_member_ce(
            args, target, speculator, examples, tokenizer, device
        )
        objective = "native-mtp-cross-entropy"
        temperature = None
    speculator.eval()
    model = PAIR_MODELS[args.pair]

    def writer(path: Path) -> None:
        speculator.save_pretrained(path)

    save_pretrained_atomically(
        head_dir,
        writer,
        {
            "stage": f"{args.variant}_head",
            "variant": args.variant,
            "objective": objective,
            "initialized_from": model["target"],
            "initialized_from_revision": model["target_revision"],
            "source_head": str(_source_head_path(args)),
            "target_checkpoint": str(run_dir / "checkpoints" / "target"),
            "verifier_owned_weights_from_target": True,
            "target_frozen": True,
            "optimizer_updates": args.head_updates,
            "effective_batch_size": args.head_batch_size * args.head_grad_accum,
            "learning_rate": args.head_lr,
            "temperature": temperature,
            "seed": args.seed,
        },
    )
    write_run_config(
        args,
        {
            "stage": f"{args.variant}_head",
            "variant": args.variant,
            "objective": objective,
            "temperature": temperature,
            "loss": history,
            "data": metadata,
        },
    )
    print(
        json.dumps(
            {
                "command": "mtp-head",
                "variant": args.variant,
                "objective": objective,
                "final_loss": history[-1],
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    run_command(
        args,
        {
            "mtp-source": cmd_mtp_source,
            "mtp-target": cmd_mtp_target,
            "mtp-head": cmd_mtp_head,
        },
    )


if __name__ == "__main__":
    main()
