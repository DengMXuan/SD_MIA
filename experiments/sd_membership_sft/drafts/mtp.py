"""Native-MTP-head draft: joint trunk+MTP SFT plus head KD on post-cutoff pool data.

Draft approach 3 of 3 (see :mod:`drafts`): the deployment adapts the
checkpoint's own native MTP head instead of a plain draft LM or an
EAGLE-3 speculator. Pair: ``qwen35_9b_mtp`` (Qwen/Qwen3.5-9B-Base).
Commands, run in order:

- ``mtp-prehead``: convert the checkpoint's native depth-1 MTP layer into
  a speculators model (the "pre-head"; no extra training).
- ``mtp-joint``: joint fine-tune trunk + native MTP head on member
  documents with ``LM CE + lambda_mtp * MTP CE`` (the "MTP trains
  together" condition).
- ``mtp-adapt``: member-blind KD adaptation of the joint head to the
  joint target on auxiliary documents.

Each run directory receives a ``results.json`` whose ``config`` block
matches :class:`config.Config` so ``generalization.py`` can replay it
unchanged.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .heads import ensure_mtp_conversion, load_mtp_speculator
from ..data import collate_sft, make_sft_example
from ..training import _autocast, _enable_checkpointing, load_causal_lm
from .common import (
    KD_TEMPERATURE,
    LAMBDA_MTP,
    PAIR_MODELS,
    build_parser,
    device_for,
    load_split,
    make_optimizer_from_params,
    resolve_output_dir,
    run_command,
    run_dir_for,
    tokenizer_for,
    write_run_config,
)


def parse_args() -> argparse.Namespace:
    parser = build_parser(__doc__, ["qwen35_9b_mtp"])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("mtp-prehead", help="convert the checkpoint's native MTP layer")
    sub.add_parser("mtp-joint", help="joint trunk+native-MTP member SFT")
    sub.add_parser("mtp-adapt", help="member-blind KD adaptation of the joint MTP head")
    args = parser.parse_args()
    resolve_output_dir(args)
    return args


def cmd_mtp_prehead(args: argparse.Namespace) -> None:
    """Convert the native depth-1 MTP layer into the run's pre-head."""
    device_for(args)
    run_dir = run_dir_for(args)
    converted = ensure_mtp_conversion(
        PAIR_MODELS[args.pair]["target"],
        run_dir / "heads" / "pre_head",
        num_speculative_steps=1,
    )
    write_run_config(args, {"pre_head": str(converted), "init": "native-export"})
    print(json.dumps({"command": "mtp-prehead", "converted": str(converted)}), flush=True)


def cmd_mtp_joint(args: argparse.Namespace) -> None:
    """Joint trunk + native-MTP speculator SFT (speculators batch-1 head API)."""
    device = device_for(args)
    run_dir = run_dir_for(args)
    tokenizer = tokenizer_for(args.pair)
    members, _nonmembers, _aux, _meta = load_split(tokenizer, args)
    target = load_causal_lm(PAIR_MODELS[args.pair]["target"], device)
    _enable_checkpointing(target)
    speculator = load_mtp_speculator(run_dir / "heads" / "pre_head", device)
    speculator.train()
    target.train()
    params = (
        [p for p in target.parameters() if p.requires_grad]
        + [p for p in speculator.parameters() if p.requires_grad]
    )
    optimizer = make_optimizer_from_params(params, args.lr, "adamw8bit")

    examples = [make_sft_example(record, tokenizer) for record in members]
    rng = np.random.default_rng(args.seed + 33)
    history_lm: list[float] = []
    history_mtp: list[float] = []
    steps_per_epoch = max(1, len(examples) // (args.batch_size * args.grad_accum))
    for epoch in range(args.epochs):
        order = rng.permutation(len(examples))
        for micro in range(steps_per_epoch * args.grad_accum):
            rows = [
                examples[int(order[(micro * args.batch_size + i) % len(examples)])]
                for i in range(args.batch_size)
            ]
            batch = collate_sft(rows, int(tokenizer.pad_token_id))
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device):
                trunk_output = target(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True,
                    use_cache=False,
                )
                shift_logits = trunk_output.logits[:, :-1].float()
                shift_labels = batch["labels"][:, 1:]
                valid = shift_labels.ne(-100)
                lm_ce = F.cross_entropy(
                    shift_logits[valid], shift_labels.clamp_min(0)[valid]
                )
            (lm_ce * len(rows)).backward()
            mtp_losses: list[float] = []
            # the speculator internally detaches its hidden-state input, so
            # MTP gradients flow into the head only; that is the deployment
            # semantics (the head adapts to the trunk, not vice versa)
            with torch.no_grad():
                hidden = trunk_output.hidden_states[-1].detach()
            with _autocast(device):
                for row in range(len(rows)):
                    length = int(batch["attention_mask"][row].sum())
                    _, mtp_loss, _ = speculator(
                        input_ids=batch["input_ids"][row : row + 1, :length],
                        hidden_states=hidden[row : row + 1, :length],
                        attention_mask=None,
                        loss_mask=batch["labels"][row : row + 1, :length].ne(-100),
                        return_dict=True,
                    )
                    (LAMBDA_MTP * mtp_loss / len(rows)).backward()
                    mtp_losses.append(float(mtp_loss.detach().cpu()))
            history_lm.append(float(lm_ce.detach().cpu()))
            history_mtp.append(float(np.mean(mtp_losses)))
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            del trunk_output, hidden
        print(
            f"mtp-joint(native) epoch {epoch + 1}/{args.epochs}: "
            f"lm={np.mean(history_lm[-steps_per_epoch * args.grad_accum:]):.5f} "
            f"mtp={np.mean(history_mtp[-steps_per_epoch * args.grad_accum:]):.5f}",
            flush=True,
        )
    (run_dir / "checkpoints" / "target").mkdir(parents=True, exist_ok=True)
    target.save_pretrained(run_dir / "checkpoints" / "target")
    tokenizer.save_pretrained(run_dir / "checkpoints" / "target")
    speculator.eval()
    (run_dir / "heads" / "joint_head").mkdir(parents=True, exist_ok=True)
    speculator.save_pretrained(run_dir / "heads" / "joint_head")
    write_run_config(
        args,
        {
            "lm_loss": float(np.mean(history_lm)),
            "mtp_loss": float(np.mean(history_mtp)),
            "init": "native-export",
        },
    )
    print(json.dumps({"command": "mtp-joint", "native": True, "done": True}), flush=True)


def cmd_mtp_adapt(args: argparse.Namespace) -> None:
    """Member-blind KD adaptation of the joint native MTP head.

    Objective matches the self-trained line: temperature KL toward the joint
    target's distribution at aligned positions. A pure CE-to-ground-truth
    objective (the speculator's built-in loss) destroys the head's
    calibration to the verifier distribution and measurably lowers
    acceptance, so it is deliberately not used here.
    """
    device = device_for(args)
    run_dir = run_dir_for(args)
    tokenizer = tokenizer_for(args.pair)
    _members, _nonmembers, auxiliary, _meta = load_split(tokenizer, args)

    from ..generalization import load_finetuned_model

    target = load_finetuned_model(run_dir, PAIR_MODELS[args.pair]["target"], device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    speculator = load_mtp_speculator(run_dir / "heads" / "joint_head", device)
    speculator.train()
    optimizer = torch.optim.AdamW(
        [p for p in speculator.parameters() if p.requires_grad], lr=args.kd_lr
    )

    examples = [make_sft_example(record, tokenizer) for record in auxiliary]
    rng = np.random.default_rng(args.seed + 34)
    losses: list[float] = []
    for step in range(args.kd_steps):
        example = examples[int(rng.integers(0, len(examples)))]
        batch = collate_sft([example], int(tokenizer.pad_token_id))
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad(), _autocast(device):
            trunk_output = target(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        with _autocast(device):
            length = int(batch["attention_mask"][0].sum())
            logits_list, _native_loss, _metrics = speculator(
                input_ids=batch["input_ids"][:, :length],
                hidden_states=trunk_output.hidden_states[-1][:, :length],
                attention_mask=None,
                return_dict=True,
            )
            # step-0 logits[:, t] predicts x_{t+2}; teacher at trunk position t+1
            student = logits_list[0].float()
            teacher = trunk_output.logits[:, 1 : 1 + student.shape[1]].float()
            labels = batch["labels"][:, 2 : 2 + student.shape[1]]
            valid = labels.ne(-100)
            kd = (
                F.kl_div(
                    F.log_softmax(student[valid] / KD_TEMPERATURE, dim=-1),
                    F.softmax(teacher[:, : student.shape[1]][valid] / KD_TEMPERATURE, dim=-1),
                    reduction="batchmean",
                )
                * KD_TEMPERATURE**2
            )
        optimizer.zero_grad(set_to_none=True)
        kd.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in speculator.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        losses.append(float(kd.detach().cpu()))
        if (step + 1) % max(1, args.kd_steps // 4) == 0:
            print(f"mtp adapt(native) step {step + 1}/{args.kd_steps}: loss={losses[-1]:.5f}", flush=True)
        del trunk_output, student, teacher, kd
    speculator.eval()
    (run_dir / "heads" / "aux_kd_head").mkdir(parents=True, exist_ok=True)
    speculator.save_pretrained(run_dir / "heads" / "aux_kd_head")
    write_run_config(args, {"final_loss": losses[-1], "objective": "kd-kl"})
    print(json.dumps({"command": "mtp-adapt", "native": True, "final_loss": losses[-1]}), flush=True)


def main() -> None:
    args = parse_args()
    run_command(
        args,
        {
            "mtp-prehead": cmd_mtp_prehead,
            "mtp-joint": cmd_mtp_joint,
            "mtp-adapt": cmd_mtp_adapt,
        },
    )


if __name__ == "__main__":
    main()
