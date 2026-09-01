"""Re-score saved NART runs with the SD-Window attack family.

Loads each run's fine-tuned target and auxiliary-distilled draft from its
saved checkpoints (no training), extracts the token-aligned features needed
by the speculative-decoding transcript, and evaluates the windowed
aggregation families of ``sd_window`` on the run's shared audit split.
Writes ``sd_window.json`` and ``SD_WINDOW.md`` into each run directory;
existing artifacts (results.json, RESULTS.md, ...) are never modified, so
re-runs are idempotent and git-safe.

Usage:
  python -m experiments.sd_membership_sft.sd_window_reanalysis \
      --run-dir experiments/results/nart_sft/newstection_qwen3_8b_epoch3 \
      --run-dir ... --gpu 0
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from .audit import make_audit_split
from .draver_activation import (
    extract_draft_activation_outputs,
    extract_target_token_outputs,
    sample_acceptance_rates,
)
from .generalization import load_draft_model, load_finetuned_model
from .nart_data import build_nart_split, pool_path as nart_pool_path
from .sd_window import evaluate_window_audit
from .training import load_causal_lm, set_seed


ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU index when --gpus is not used",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default=None,
        help="comma-separated GPU ids or 'auto': distribute run-dirs across "
        "GPUs by spawning one worker subprocess each and wait",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--detector-seeds", type=int, default=3)
    return parser.parse_args()


def _idle_gpus() -> list[int]:
    """GPU ids with no significant memory in use right now."""
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    idle = []
    for line in output.splitlines():
        index, used = (part.strip() for part in line.split(","))
        if int(used) < 1024:
            idle.append(int(index))
    return idle


def _spawn_worker(gpu: int, run_dirs: list[Path], args: argparse.Namespace) -> subprocess.Popen:
    command = [
        "uv", "run", "--no-sync", "python", "-m",
        "experiments.sd_membership_sft.sd_window_reanalysis",
        "--gpu", str(gpu),
        "--batch-size", str(args.batch_size),
        "--bootstrap-repeats", str(args.bootstrap_repeats),
        "--detector-seeds", str(args.detector_seeds),
    ]
    for run_dir in run_dirs:
        command.extend(["--run-dir", str(run_dir)])
    return subprocess.Popen(command, cwd=ROOT)


def main() -> None:
    args = parse_args()
    run_dirs = [
        raw if Path(raw).is_absolute() else ROOT / raw for raw in args.run_dir
    ]
    if args.gpus:
        if args.gpus == "auto":
            gpus = _idle_gpus()
            if not gpus:
                raise RuntimeError("no idle GPU available for --gpus auto")
        else:
            gpus = [int(value) for value in args.gpus.split(",")]
        buckets: dict[int, list[Path]] = {gpu: [] for gpu in gpus}
        for index, run_dir in enumerate(run_dirs):
            buckets[gpus[index % len(gpus)]].append(run_dir)
        workers = [
            _spawn_worker(gpu, dirs, args) for gpu, dirs in buckets.items() if dirs
        ]
        failures = 0
        for worker in workers:
            if worker.wait() != 0:
                failures += 1
        if failures:
            raise RuntimeError(f"{failures} SD-Window worker(s) failed; see logs above")
        return

    for raw in args.run_dir:
        run_dir = Path(raw)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        output_json = run_dir / "sd_window.json"
        if output_json.exists():
            print(f"skip (already scored): {run_dir}", flush=True)
            continue
        started = time.time()
        result = score_run(run_dir, args)
        run_name = run_dir.name

        def dump(payload: dict[str, Any], path: Path, renderer) -> None:
            text = renderer(payload) if renderer else json.dumps(payload, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value))
            temporary = path.with_name(path.name + ".tmp")
            temporary.write_text(text, encoding="utf-8")
            os.replace(temporary, path)  # atomic: no partial artifacts, no TOCTOU window

        dump(result, run_dir / "sd_window.json", None)
        dump(result, run_dir / "SD_WINDOW.md", lambda payload: render_markdown(run_name, payload))
        headline = {
            name.split("/")[-1]: round(row["auc"], 4)
            for name, row in result["metrics"].items()
        }
        print(
            json.dumps({"run": run_name, "elapsed_s": round(time.time() - started, 1), **headline}),
            flush=True,
        )


def load_run_config(run_dir: Path) -> Any:
    from .config import Config

    artifact = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    values = dict(artifact["config"])
    if values.get("pool_path"):
        values["pool_path"] = Path(values["pool_path"])
    return Config(**values)


def score_run(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_run_config(run_dir)
    if cfg.benchmark == "legacy":
        raise RuntimeError(f"{run_dir}: window analysis requires an NART benchmark run")
    device = torch.device(f"cuda:{args.gpu}")
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(cfg.draft_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pool = (
        cfg.pool_path
        if cfg.pool_path is not None
        else nart_pool_path(cfg.benchmark)
    )
    if not pool.is_absolute():
        pool = ROOT / pool
    members, nonmembers, _auxiliary, split_metadata = build_nart_split(
        cfg.benchmark, pool, tokenizer, cfg.n_per_class, cfg.n_aux, cfg.data_seed
    )
    candidates = members + nonmembers
    labels = np.concatenate(
        [
            np.ones(len(members), dtype=np.int64),
            np.zeros(len(nonmembers), dtype=np.int64),
        ]
    )

    tuned_target = load_finetuned_model(run_dir, cfg.target_model, device)
    target_features = extract_target_token_outputs(
        tuned_target, candidates, tokenizer, device, args.batch_size
    )
    del tuned_target
    gc.collect()
    torch.cuda.empty_cache()

    aux_draft = load_draft_model(
        run_dir, cfg.draft_model, "draft_auxiliary_distilled", device
    )
    draft_features = extract_draft_activation_outputs(
        aux_draft, candidates, tokenizer, device, args.batch_size
    )
    del aux_draft
    gc.collect()
    torch.cuda.empty_cache()

    # Reproduce the run's probe selection: min-k fraction over draft logp,
    # capped at the run's selected_token_cap.
    draft_logp = draft_features["token_logp"]
    finite = np.isfinite(draft_logp)
    k = max(1, min(draft_logp.shape[1], int(np.ceil(draft_logp.shape[1] * cfg.min_k_fraction))))
    safe = np.where(finite, draft_logp, np.inf)
    selected = np.argpartition(safe, kth=k - 1, axis=1)[:, :k]
    if cfg.selected_token_cap and cfg.selected_token_cap < selected.shape[1]:
        gathered = np.take_along_axis(draft_logp, selected, axis=1)
        order = np.argsort(gathered, axis=1, kind="mergesort")[:, : cfg.selected_token_cap]
        selected = np.take_along_axis(selected, order, axis=1)

    acceptance, _exact_alpha = sample_acceptance_rates(
        target_features["token_logp"],
        draft_features["token_logp"],
        selected,
        cfg.transcript_repeats,
        cfg.audit_seed,
    )
    positions = selected
    visible = np.isfinite(acceptance)

    calibration, test = make_audit_split(
        len(members), len(nonmembers), cfg.audit_train_per_class, cfg.audit_seed
    )
    selected_activations = draft_features["activation_stats"][
        np.arange(len(candidates))[:, None], selected
    ]
    result = evaluate_window_audit(
        acceptance,
        positions,
        selected_activations=selected_activations,
        labels=labels,
        calibration=calibration,
        test=test,
        bootstrap_repeats=args.bootstrap_repeats,
        seed=cfg.audit_seed,
        visible=visible,
        detector_seeds=args.detector_seeds,
    )
    result["run_dir"] = str(run_dir)
    result["protocol"].update(
        {
            "benchmark": cfg.benchmark,
            "pool_sha256": split_metadata["pool_sha256"],
            "min_k_fraction": cfg.min_k_fraction,
            "selected_token_cap": cfg.selected_token_cap,
            "transcript_repeats": cfg.transcript_repeats,
            "probe_positions": int(selected.shape[1]),
        }
    )
    return result


def render_markdown(run_name: str, result: dict[str, Any]) -> str:
    lines = [
        f"# SD-Window reanalysis: `{run_name}`",
        "",
        f"- Benchmark: {result['protocol']['benchmark']} "
        f"(pool sha {result['protocol']['pool_sha256'][:16]})",
        f"- Probed positions per record: {result['protocol']['probe_positions']} "
        f"(min-k {result['protocol']['min_k_fraction']}, cap "
        f"{result['protocol']['selected_token_cap']}, "
        f"{result['protocol']['transcript_repeats']} repeats)",
        f"- Window definitions: {', '.join(result['protocol']['window_definitions'])}; "
        f"visibility mask honoured: {result['protocol']['visibility_masked']}",
        "",
        "| Signal | AUC (95% CI) | TPR@1%FPR | TPR@5%FPR |",
        "|---|---:|---:|---:|",
    ]
    for name, row in sorted(result["metrics"].items()):
        lines.append(
            f"| `{name}` | {row['auc']:.4f} [{row['auc_ci95_low']:.4f}, {row['auc_ci95_high']:.4f}] "
            f"| {row['tpr_at_1pct_fpr']:.4f} | {row['tpr_at_5pct_fpr']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Paired AUC deltas vs the unwindowed acceptance mean "
            "(positive = windowing helps):",
            "",
            "| Signal | Delta AUC (95% CI) |",
            "|---|---:|",
        ]
    )
    for name, delta in sorted(result["paired_auc_deltas"].items()):
        signal = name.replace(" minus unwindowed_mean", "")
        lines.append(
            f"| `{signal}` | {delta['delta_auc']:+.4f} "
            f"[{delta['ci95_low']:+.4f}, {delta['ci95_high']:+.4f}] |"
        )
    lines.append("")
    return "\n".join(lines)


