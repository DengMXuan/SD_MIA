"""EAGLE-3 speculator-head draft: target SFT plus head KD on post-cutoff pool data.

Draft approach 2 of 3 (see :mod:`drafts`): the deployment adapts the
published EAGLE-3 speculator head instead of a plain draft LM. Pairs:
``qwen3_8b_eagle3`` (Qwen/Qwen3-8B + RedHatAI/Qwen3-8B-speculator.eagle3)
and ``llama31_8b_eagle3`` (unsloth/Meta-Llama-3.1-8B-Instruct + its
EAGLE-3 speculator). Commands:

- ``eagle-target``: full-parameter member SFT of the EAGLE-line target.
- ``eagle-head``: continue-train the published EAGLE-3 head against the
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

from .heads import eagle3_target_layer_ids, load_eagle3_speculator
from ..data import collate_sft, make_sft_example
from ..training import _autocast, _make_optimizer, load_causal_lm, sft_train
from .common import (
    KD_TEMPERATURE,
    PAIR_MODELS,
    build_parser,
    device_for,
    load_split,
    resolve_output_dir,
    run_command,
    run_dir_for,
    tokenizer_for,
    write_run_config,
)


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
    tokenizer = tokenizer_for(args.pair)
    members, _nonmembers, _aux, _meta = load_split(tokenizer, args)
    target = load_causal_lm(PAIR_MODELS[args.pair]["target"], device)
    started = time.time()
    losses = sft_train(
        target, members, tokenizer, device,
        epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum,
        lr=args.lr, seed=args.seed + 10, label="eagle target SFT",
        optimizer_name="adamw8bit",
    )
    (run_dir / "checkpoints" / "target").mkdir(parents=True, exist_ok=True)
    target.save_pretrained(run_dir / "checkpoints" / "target")
    tokenizer.save_pretrained(run_dir / "checkpoints" / "target")
    write_run_config(args, {"target_sft_loss": losses, "seconds": time.time() - started})
    print(json.dumps({"command": "eagle-target", "losses": losses}, indent=2), flush=True)


def _eagle_kd_loss(
    speculator: Any,
    target: Any,
    batch: dict[str, torch.Tensor],
    device: torch.device,
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
            F.log_softmax(student_selected / KD_TEMPERATURE, dim=-1),
            F.softmax(teacher_selected / KD_TEMPERATURE, dim=-1),
            reduction="batchmean",
        )
        * KD_TEMPERATURE**2
    )
    return kd, valid


def cmd_eagle_head(args: argparse.Namespace) -> None:
    assert args.variant in ("aux", "member")
    device = device_for(args)
    run_dir = run_dir_for(args)
    tokenizer = tokenizer_for(args.pair)
    members, _nonmembers, auxiliary, _meta = load_split(tokenizer, args)
    records = auxiliary if args.variant == "aux" else members

    from ..generalization import load_finetuned_model

    target = load_finetuned_model(run_dir, PAIR_MODELS[args.pair]["target"], device)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    speculator = load_eagle3_speculator(PAIR_MODELS[args.pair]["speculator"], device)
    speculator.train()
    optimizer = _make_optimizer(speculator, args.kd_lr, "adamw")

    examples = [make_sft_example(record, tokenizer) for record in records]
    rng = np.random.default_rng(args.seed + 31)
    losses: list[float] = []
    optimizer.zero_grad(set_to_none=True)
    pending = 0
    for step in range(args.kd_steps):
        indices = rng.integers(0, len(examples), size=args.kd_batch_size)
        batch = collate_sft(
            [examples[int(i)] for i in indices], int(tokenizer.pad_token_id)
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        kd, _valid = _eagle_kd_loss(speculator, target, batch, device)
        loss = kd / args.kd_grad_accum
        loss.backward()
        pending += 1
        losses.append(float(kd.detach().cpu()))
        if pending == args.kd_grad_accum:
            torch.nn.utils.clip_grad_norm_(
                [p for p in speculator.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            pending = 0
        if (step + 1) % max(1, args.kd_steps // 4) == 0:
            print(f"eagle head kd step {step + 1}/{args.kd_steps}: loss={losses[-1]:.5f}", flush=True)
    head_dir = run_dir / "heads" / f"{args.variant}_kd_head"
    head_dir.mkdir(parents=True, exist_ok=True)
    speculator.save_pretrained(head_dir)
    from huggingface_hub import snapshot_download

    source = Path(snapshot_download(repo_id=PAIR_MODELS[args.pair]["speculator"]))
    for name in ("eagle3.py",):
        if (source / name).is_file():
            shutil.copy(source / name, head_dir / name)
    write_run_config(args, {"variant": args.variant, "kd_loss_last": losses[-1]})
    print(json.dumps({"command": "eagle-head", "variant": args.variant, "final_loss": losses[-1]}), flush=True)


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
