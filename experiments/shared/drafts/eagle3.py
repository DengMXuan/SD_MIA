"""Frozen-target EAGLE-3 adaptation on shared controlled-SFT splits.

Draft approach 2 of 3 (see :mod:`drafts`): the deployment adapts the
published EAGLE-3 speculator head instead of a plain draft LM. Pairs:
``qwen3_8b_eagle3`` (Qwen/Qwen3-8B + RedHatAI/Qwen3-8B-speculator.eagle3)
and ``llama31_8b_eagle3`` (unsloth/Meta-Llama-3.1-8B-Instruct + its
EAGLE-3 speculator). Commands:

- ``eagle-target``: full-parameter member SFT of the EAGLE-line target.
- ``eagle-head``: independently train a fresh published EAGLE-3 head against the
  fine-tuned target with KD; ``--variant aux`` uses auxiliary documents
  (member-blind, deployment-aligned), ``--variant member`` uses member
  documents (boundary condition). Only the data differs between variants.

Targets load through :func:`training.load_causal_lm`. Trunk SFT
hyperparameters mirror the mainline SFT settings: lr 2e-5, effective batch 16,
PagedAdamW8bit, bf16. Each run directory receives a ``results.json``
whose ``config`` block matches :class:`config.Config` so
``generalization.py`` can replay it unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from experiments.shared.drafts.heads import eagle3_target_layer_ids, load_eagle3_speculator
from experiments.shared.data.data import collate_sft, make_sft_example
from experiments.shared.training.training import _autocast, _make_optimizer, load_causal_lm, set_seed, sft_train
from experiments.shared.drafts.common import PAIR_MODELS, build_parser, cached_snapshot, checkpoint_complete, device_for, load_split, resolve_output_dir, run_command, run_dir_for, save_pretrained_atomically, tokenizer_for, write_run_config


def parse_args() -> argparse.Namespace:
    parser = build_parser(__doc__, ["qwen3_8b_eagle3", "llama31_8b_eagle3"])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("eagle-target", help="member SFT of the EAGLE-line target")
    head = sub.add_parser("eagle-head", help="KD continue-training of the EAGLE-3 head")
    head.add_argument("--variant", choices=["aux", "member"], required=True)
    args = parser.parse_args()
    resolve_output_dir(args)
    return args


def cmd_eagle_target(args: argparse.Namespace) -> None:
    device = device_for(args)
    run_dir = run_dir_for(args)
    checkpoint = run_dir / "checkpoints" / "target"
    if checkpoint_complete(checkpoint):
        print(f"[skip] complete target checkpoint: {checkpoint}", flush=True)
        return
    tokenizer = tokenizer_for(args.pair)
    members, _nonmembers, _aux, metadata = load_split(tokenizer, args)
    model = PAIR_MODELS[args.pair]
    target = load_causal_lm(
        model["target"],
        device,
        revision=model["target_revision"],
        local_files_only=True,
    )
    started = time.time()
    losses = sft_train(
        target, members, tokenizer, device,
        epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, seed=args.seed, label="eagle target SFT",
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
    print(json.dumps({"command": "eagle-target", "losses": losses}, indent=2), flush=True)


def _eagle_kd_loss(
    speculator: Any,
    target: Any,
    batch: dict[str, torch.Tensor],
    device: torch.device,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One EAGLE-3 KD step: draft-vocab KL against the tuned target teacher.

    ``d2t`` holds *offsets* (target_idx = draft_idx + d2t[draft_idx], per the
    checkpoint's own ``map_draft_to_target_tokens``); the teacher logits must
    be gathered at those absolute positions, not at the offsets themselves.
    """
    base = speculator.get_base_model() if hasattr(speculator, "get_base_model") else speculator
    captured: list[torch.Tensor] = []
    handle = base.norm.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    try:
        with torch.no_grad(), _autocast(device):
            target_output = target(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            offsets = base.d2t.long()
            absolute = torch.arange(offsets.numel(), device=device) + offsets
            teacher = target_output.logits[:, :-1].float().index_select(-1, absolute)
            connector = torch.cat(
                [target_output.hidden_states[index] for index in eagle3_target_layer_ids(target)],
                dim=-1,
            )
        with _autocast(device):
            speculator(
                input_ids=batch["input_ids"],
                hidden_states=connector,
                attention_mask=batch["attention_mask"],
                return_dict=True,
            )
            student = base.lm_head(captured[-1])[:, :-1].float()
    finally:
        handle.remove()
    labels = batch["labels"][:, 1:]
    valid = labels.ne(-100)
    student_selected = student[valid]
    teacher_selected = teacher[valid]
    kd = (
        F.kl_div(
            F.log_softmax(student_selected / temperature, dim=-1),
            F.softmax(teacher_selected / temperature, dim=-1),
            reduction="batchmean",
        )
        * temperature**2
    )
    return kd, valid


def cmd_eagle_head(args: argparse.Namespace) -> None:
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

    from experiments.shared.training.generalization import load_finetuned_model

    target = load_finetuned_model(run_dir, PAIR_MODELS[args.pair]["target"], device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    target.eval()
    target.config.use_cache = False
    model = PAIR_MODELS[args.pair]
    speculator = load_eagle3_speculator(
        model["speculator"], device, revision=model["speculator_revision"]
    )
    speculator.train()
    set_seed(args.seed)
    optimizer = _make_optimizer(speculator, args.head_lr, "adamw")

    examples = [make_sft_example(record, tokenizer) for record in records]
    rng = np.random.default_rng(args.seed)
    losses: list[float] = []
    trainable = [p for p in speculator.parameters() if p.requires_grad]
    for update in range(args.head_updates):
        optimizer.zero_grad(set_to_none=True)
        micro_losses: list[float] = []
        for _micro in range(args.head_grad_accum):
            indices = rng.integers(0, len(examples), size=args.head_batch_size)
            batch = collate_sft(
                [examples[int(i)] for i in indices], int(tokenizer.pad_token_id)
            )
            batch = {k: v.to(device) for k, v in batch.items()}
            kd, _valid = _eagle_kd_loss(
                speculator, target, batch, device, args.kd_temperature
            )
            (kd / args.head_grad_accum).backward()
            micro_losses.append(float(kd.detach().cpu()))
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        losses.append(float(np.mean(micro_losses)))
        if (update + 1) % max(1, args.head_updates // 4) == 0:
            print(
                f"eagle head {args.variant} update {update + 1}/"
                f"{args.head_updates}: loss={losses[-1]:.5f}",
                flush=True,
            )
    speculator.eval()
    source = cached_snapshot(model["speculator"], model["speculator_revision"])

    def writer(path: Path) -> None:
        speculator.save_pretrained(path)
        implementation = source / "eagle3.py"
        if not implementation.is_file():
            raise FileNotFoundError(f"eagle3.py not found in {source}")
        shutil.copy(implementation, path / "eagle3.py")

    save_pretrained_atomically(
        head_dir,
        writer,
        {
            "stage": f"{args.variant}_head",
            "variant": args.variant,
            "objective": "temperature-kl",
            "initialized_from": model["speculator"],
            "initialized_from_revision": model["speculator_revision"],
            "target_checkpoint": str(run_dir / "checkpoints" / "target"),
            "target_frozen": True,
            "optimizer_updates": args.head_updates,
            "effective_batch_size": args.head_batch_size * args.head_grad_accum,
            "learning_rate": args.head_lr,
            "temperature": args.kd_temperature,
            "seed": args.seed,
        },
    )
    write_run_config(
        args,
        {
            "stage": f"{args.variant}_head",
            "variant": args.variant,
            "objective": "temperature-kl",
            "kd_loss": losses,
            "data": metadata,
        },
    )
    print(
        json.dumps(
            {
                "command": "eagle-head",
                "variant": args.variant,
                "final_loss": losses[-1],
            }
        ),
        flush=True,
    )


def main() -> None:
    args = parse_args()
    run_command(
        args,
        {
            "eagle-target": cmd_eagle_target,
            "eagle-head": cmd_eagle_head,
        },
    )


if __name__ == "__main__":
    main()
