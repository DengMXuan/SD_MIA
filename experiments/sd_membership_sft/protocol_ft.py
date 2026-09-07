"""Protocol-track fine-tuning: EAGLE-3 and MTP drafter pairs on NART data.

Subcommands cover the four protocol pairs (see the experiment plan of
2026-09-06; all use the newstection pool and replay the mainline split with
``data_seed=20260824``):

- ``eagle-target``: full-parameter member SFT of the EAGLE-line target
  (Qwen/Qwen3-8B instruct or the unsloth Llama-3.1-8B-Instruct mirror).
- ``eagle-head``: continue-train the published EAGLE-3 head against the
  fine-tuned target with KD; ``--variant aux`` uses auxiliary documents
  (member-blind, deployment-aligned), ``--variant member`` uses member
  documents (boundary condition). Only the data differs between variants.
- ``mtp-prehead``: convert the checkpoint's native depth-1 MTP layer into a
  speculators model (the "pre-head"; no extra training).
- ``mtp-joint``: joint fine-tune trunk + native MTP head on member documents
  with ``LM CE + lambda_mtp * MTP CE`` (the "MTP trains together" condition).
- ``mtp-adapt``: member-blind KD adaptation of the joint head to the joint
  target on auxiliary documents.

Targets load through :func:`training.load_causal_lm`, which falls back to
``AutoModelForImageTextToText`` for the Qwen3.5 wrapper architecture. Trunk
SFT hyperparameters mirror the NART mainline: lr 2e-5, effective batch 16,
PagedAdamW8bit, bf16. Each run directory receives a ``results.json`` whose
``config`` block matches :class:`config.Config` so ``generalization.py`` can
replay it unchanged.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from .config import Config
from .data import SFTRecord, collate_sft, make_sft_example
from .nart_data import build_nart_split, pool_path as nart_pool_path
from .protocol_heads import (
    eagle3_target_layer_ids,
    ensure_mtp_conversion,
    load_eagle3_speculator,
    load_mtp_speculator,
)
from .training import (
    _autocast,
    _enable_checkpointing,
    _make_optimizer,
    load_causal_lm,
    set_seed,
    sft_train,
)

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("experiments/data/nart_benchmarks")
RESULTS_ROOT = Path("experiments/results/protocol_ft")
LAMBDA_MTP = 0.3
KD_TEMPERATURE = 2.0
KD_STEPS = 384
KD_LR = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pair", required=True, choices=[
        "qwen3_8b_eagle3", "llama31_8b_eagle3", "qwen35_9b_mtp",
    ])
    parser.add_argument(
        "--benchmark",
        choices=["newstection", "wikitection", "arxivtection"],
        default="newstection",
    )
    parser.add_argument("--epochs", type=int, default=3, help="target SFT epochs")
    parser.add_argument("--data-seed", type=int, default=20260824)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--n-per-class", type=int, default=2000)
    parser.add_argument("--n-aux", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--kd-steps", type=int, default=KD_STEPS)
    parser.add_argument("--kd-lr", type=float, default=KD_LR)
    parser.add_argument("--kd-batch-size", type=int, default=2)
    parser.add_argument("--kd-grad-accum", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("eagle-target", help="member SFT of the EAGLE-line target")
    head = sub.add_parser("eagle-head", help="KD continue-training of the EAGLE-3 head")
    head.add_argument("--variant", choices=["aux", "member"], required=True)
    sub.add_parser("mtp-prehead", help="convert the checkpoint's native MTP layer")
    joint = sub.add_parser("mtp-joint", help="joint trunk+native-MTP member SFT")
    sub.add_parser("mtp-adapt", help="member-blind KD adaptation of the joint MTP head")
    args = parser.parse_args()
    if args.output_dir is None:
        tag = "" if args.lr == 2e-5 else f"_lr{args.lr:g}"
        bench = "" if args.benchmark == "newstection" else f"_{args.benchmark}"
        args.output_dir = (
            RESULTS_ROOT / args.pair / f"{args.pair}{bench}{tag}_epoch{args.epochs}"
        )
    return args


PAIR_MODELS: dict[str, dict[str, str]] = {
    "qwen3_8b_eagle3": {
        "target": "Qwen/Qwen3-8B",
        "speculator": "RedHatAI/Qwen3-8B-speculator.eagle3",
    },
    "llama31_8b_eagle3": {
        "target": "unsloth/Meta-Llama-3.1-8B-Instruct",
        "speculator": "RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3",
    },
    "qwen35_9b_mtp": {"target": "Qwen/Qwen3.5-9B-Base"},
}


def run_dir_for(args: argparse.Namespace) -> Path:
    return args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir


def load_split(tokenizer: Any, args: argparse.Namespace):
    pool = nart_pool_path(args.benchmark)
    return build_nart_split(
        args.benchmark, ROOT / pool, tokenizer, args.n_per_class, args.n_aux, args.data_seed
    )


def _loader(
    records: list[SFTRecord], tokenizer: Any, batch_size: int, seed: int | None
) -> DataLoader:
    examples = [make_sft_example(record, tokenizer) for record in records]
    generator = torch.Generator()
    generator.manual_seed(seed or 0)
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=seed is not None,
        generator=generator,
        num_workers=0,
        collate_fn=lambda rows: collate_sft(rows, int(tokenizer.pad_token_id)),
    )


def write_run_config(args: argparse.Namespace, extra: dict[str, Any]) -> None:
    """results.json whose config block matches Config for generalization.py."""
    run_dir = run_dir_for(args)
    cfg = Config().as_dict()
    cfg.update(
        {
            "gpu": args.gpu,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "target_model": PAIR_MODELS[args.pair]["target"],
            "draft_model": PAIR_MODELS[args.pair]["target"],  # no separate draft LM in this track
            "benchmark": args.benchmark,
            "pool_path": ROOT / DATA_ROOT / args.benchmark / "pool.jsonl",
            "n_per_class": args.n_per_class,
            "n_aux": args.n_aux,
            "target_epochs": args.epochs,
            "target_batch_size": args.batch_size,
            "target_grad_accum": args.grad_accum,
            "target_lr": args.lr,
            "output_dir": run_dir,
        }
    )
    artifact = {
        "protocol_track": {"pair": args.pair, "command": args.command, **extra},
        "config": cfg,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# EAGLE line
# ---------------------------------------------------------------------------


def cmd_eagle_target(args: argparse.Namespace) -> None:
    device = _device(args)
    run_dir = run_dir_for(args)
    tokenizer = _tokenizer_for(args.pair)
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
    device = _device(args)
    run_dir = run_dir_for(args)
    tokenizer = _tokenizer_for(args.pair)
    members, _nonmembers, auxiliary, _meta = load_split(tokenizer, args)
    records = auxiliary if args.variant == "aux" else members

    from .generalization import load_finetuned_model

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


# ---------------------------------------------------------------------------
# MTP line
# ---------------------------------------------------------------------------


def cmd_mtp_prehead(args: argparse.Namespace) -> None:
    """Convert the native depth-1 MTP layer into the run's pre-head."""
    _device(args)
    run_dir = run_dir_for(args)
    converted = ensure_mtp_conversion(
        PAIR_MODELS[args.pair]["target"],
        run_dir / "heads" / "pre_head",
        num_speculative_steps=1,
    )
    write_run_config(args, {"pre_head": str(converted), "init": "native-export"})
    print(json.dumps({"command": "mtp-prehead", "converted": str(converted)}), flush=True)


def cmd_mtp_joint(args: argparse.Namespace) -> None:
    _mtp_joint_native(args)


def _mtp_joint_native(args: argparse.Namespace) -> None:
    """Joint trunk + native-MTP speculator SFT (speculators batch-1 head API)."""
    device = _device(args)
    run_dir = run_dir_for(args)
    tokenizer = _tokenizer_for(args.pair)
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
    optimizer = _make_optimizer_from_params(params, args.lr, "adamw8bit")

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


def _make_optimizer_from_params(params: list[torch.nn.Parameter], lr: float, name: str):
    if name == "adamw8bit":
        import bitsandbytes as bnb

        try:
            return bnb.optim.PagedAdamW8bit(params, lr=lr)
        except (TypeError, RuntimeError):
            return bnb.optim.AdamW8bit(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr)


def _mtp_adapt_native(args: argparse.Namespace) -> None:
    """Member-blind KD adaptation of the joint native MTP head.

    Objective matches the self-trained line: temperature KL toward the joint
    target's distribution at aligned positions. A pure CE-to-ground-truth
    objective (the speculator's built-in loss) destroys the head's
    calibration to the verifier distribution and measurably lowers
    acceptance, so it is deliberately not used here.
    """
    device = _device(args)
    run_dir = run_dir_for(args)
    tokenizer = _tokenizer_for(args.pair)
    _members, _nonmembers, auxiliary, _meta = load_split(tokenizer, args)

    from .generalization import load_finetuned_model

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


def cmd_mtp_adapt(args: argparse.Namespace) -> None:
    _mtp_adapt_native(args)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _device(args: argparse.Namespace) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    set_seed(args.seed)
    return device


def _tokenizer_for(pair: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(PAIR_MODELS[pair]["target"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def main() -> None:
    args = parse_args()
    started = time.time()
    if args.command == "eagle-target":
        cmd_eagle_target(args)
    elif args.command == "eagle-head":
        cmd_eagle_head(args)
    elif args.command == "mtp-prehead":
        cmd_mtp_prehead(args)
    elif args.command == "mtp-joint":
        cmd_mtp_joint(args)
    elif args.command == "mtp-adapt":
        cmd_mtp_adapt(args)
    else:
        raise ValueError(args.command)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{args.command}] total {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
