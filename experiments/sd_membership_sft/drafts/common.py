"""Shared helpers for the head-based draft approaches (EAGLE-3 / MTP).

``drafts.plain`` is self-contained; the two head-based approaches share
the model-pair registry, the split loader, and the run-config writer
here. Results stay under ``experiments/results/protocol_ft`` for
continuity with the existing run artifacts.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
from transformers import AutoTokenizer

from ..config import Config
from ..splits import build_split, pool_path
from ..training import set_seed

ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path("experiments/data/pools")
RESULTS_ROOT = Path("experiments/results/protocol_ft")
LAMBDA_MTP = 0.3
KD_TEMPERATURE = 2.0
KD_STEPS = 384
KD_LR = 1e-4


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


def run_dir_for(args: Any) -> Path:
    return args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir


def load_split(tokenizer: Any, args: Any):
    pool = pool_path(args.benchmark)
    return build_split(
        args.benchmark, ROOT / pool, tokenizer, args.n_per_class, args.n_aux, args.data_seed
    )


def device_for(args: Any) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    set_seed(args.seed)
    return device


def tokenizer_for(pair: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(PAIR_MODELS[pair]["target"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def make_optimizer_from_params(params: list[torch.nn.Parameter], lr: float, name: str):
    if name == "adamw8bit":
        import bitsandbytes as bnb

        try:
            return bnb.optim.PagedAdamW8bit(params, lr=lr)
        except (TypeError, RuntimeError):
            return bnb.optim.AdamW8bit(params, lr=lr)
    return torch.optim.AdamW(params, lr=lr)


def build_parser(description: str, pairs: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--pair", required=True, choices=pairs)
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
    return parser


def resolve_output_dir(args: argparse.Namespace) -> None:
    if args.output_dir is None:
        tag = "" if args.lr == 2e-5 else f"_lr{args.lr:g}"
        bench = "" if args.benchmark == "newstection" else f"_{args.benchmark}"
        args.output_dir = (
            RESULTS_ROOT / args.pair / f"{args.pair}{bench}{tag}_epoch{args.epochs}"
        )


def run_command(args: argparse.Namespace, handlers: dict[str, Callable[[Any], None]]) -> None:
    started = time.time()
    handler = handlers.get(args.command)
    if handler is None:
        raise ValueError(args.command)
    handler(args)
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[{args.command}] total {time.time() - started:.0f}s", flush=True)


def write_run_config(args: Any, extra: dict[str, Any]) -> None:
    """results.json whose config block matches Config for generalization.py."""
    run_dir = run_dir_for(args)
    cfg = Config().as_dict()
    cfg.update(
        {
            "gpu": args.gpu,
            "seed": args.seed,
            "data_seed": args.data_seed,
            "target_model": PAIR_MODELS[args.pair]["target"],
            "draft_model": PAIR_MODELS[args.pair]["target"],  # no separate draft LM
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
