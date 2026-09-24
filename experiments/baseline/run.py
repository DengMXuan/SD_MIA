"""Run target-only MIA baselines on a completed controlled-SFT run.

Example::

    CUDA_VISIBLE_DEVICES=0 uv run --no-sync python -m experiments.baseline.run \
      --run-dir experiments/results/sft_runs/newstection_qwen3_8b_epoch1 \
      --gpu 0 --methods all --output-dir experiments/results/baseline/newstection_e1

The process loads exactly one model: the saved fine-tuned target checkpoint.
It never loads a draft, a reference model, or the pre-SFT target as an
independent scoring model.  LoRA loading necessarily instantiates the base
weights underneath the saved adapter, but every score is produced by the
resulting target checkpoint with the adapter active.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from ..sd_membership_sft.data import SFTRecord, prompt_prefix_ids
from ..sd_membership_sft.generalization import load_finetuned_model, load_run_config
from ..sd_membership_sft.logpq_distribution import verify_split_against_run
from ..sd_membership_sft.splits import build_split, pool_path
from ..sd_membership_sft.training import set_seed
from .runtime import RunProgress
from .costs import CostMeter, cost_protocol, cost_table, write_cost_report
from . import METHODS
from .methods import (
    icp_score,
    mean_log_likelihood,
    min_k_plus_plus_score,
    min_k_prob_score,
    petal_score,
    rank_auc,
    recall_score,
    rouge1_recall,
    sead_score,
    upper_tail_tpr,
)

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class AuditRecord:
    record: SFTRecord
    label: int


@dataclass
class TokenStats:
    token_ids: np.ndarray
    token_logp: np.ndarray
    expected_logp: np.ndarray
    variance_logp: np.ndarray
    log_similarity: np.ndarray | None = None
    sampled_token_ids: np.ndarray | None = None
    sead_log_density: float | None = None
    sead_lexical_density: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--pretraining-manifest", type=Path, help="frozen MIMIR evaluation manifest")
    parser.add_argument("--pool-path", type=Path)
    parser.add_argument("--cost-warmup-records", type=int, default=1, help="untimed forward warmup records before each method")
    parser.add_argument("--progress-interval", type=float, default=30.0, help="seconds between progress updates")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--methods",
        default="all",
        help="comma-separated method names, or 'all' (default: all)",
    )
    parser.add_argument("--k-percent", type=float, default=20.0)
    parser.add_argument("--recall-shots", type=int, default=4)
    parser.add_argument("--icp-top-k", type=int, default=5)
    parser.add_argument("--icp-aggregation", choices=("min", "mean", "max"), default="min")
    parser.add_argument("--sead-samples", type=int, default=50)
    parser.add_argument(
        "--sead-temperature",
        type=float,
        default=1.0,
        help="SEAD sampling temperature; 1.0 matches the official frequency estimator",
    )
    parser.add_argument("--samia-samples", type=int, default=10)
    parser.add_argument("--prefix-ratio", type=float, default=0.5)
    parser.add_argument("--perturbation-rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--batch-size", type=int, default=1, help="reserved for future batched forward passes")
    parser.add_argument(
        "--record-start",
        type=int,
        default=0,
        help="inclusive audit-record index; useful for parallel generation runs",
    )
    parser.add_argument(
        "--record-end",
        type=int,
        help="exclusive audit-record index; defaults to the end of the audit set",
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=8,
        help="batch size for generation-based baselines",
    )
    parser.add_argument(
        "--attn-implementation", choices=("eager", "sdpa"), default="eager"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/baseline"),
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _target_tokenizer(run_dir: Path, cfg: Any) -> Any:
    """Prefer tokenizer files saved with the target checkpoint when present."""

    for directory in ("checkpoints", "adapters"):
        target = run_dir / directory / "target"
        if target.exists() and (target / "tokenizer_config.json").exists():
            tokenizer = AutoTokenizer.from_pretrained(str(target))
            break
    else:
        # Full target checkpoints from ``drafts.plain`` do not copy tokenizer
        # files.  The configured target/draft pairs share a tokenizer, so the
        # target tokenizer reproduces the frozen split without touching any
        # draft artifact.  ``verify_split_against_run`` below fails loudly if
        # a custom run violates that repository invariant.
        tokenizer = AutoTokenizer.from_pretrained(cfg.target_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_audit_records(
    run_dir: Path, cfg: Any, tokenizer: Any, pool_override: Path | None
) -> tuple[list[AuditRecord], list[AuditRecord], list[SFTRecord], dict[str, Any]]:
    pool = pool_override if pool_override is not None else cfg.pool_path
    if pool is None:
        pool = pool_path(cfg.benchmark)
    pool = _resolve(Path(pool))
    members, nonmembers, auxiliary, metadata = build_split(
        cfg.benchmark,
        pool,
        tokenizer,
        cfg.n_per_class,
        cfg.n_aux,
        cfg.data_seed,
    )
    verify_split_against_run(members, nonmembers, run_dir)
    return (
        [AuditRecord(record, 1) for record in members],
        [AuditRecord(record, 0) for record in nonmembers],
        auxiliary,
        metadata,
    )


def _response_ids(record: SFTRecord, tokenizer: Any) -> list[int]:
    values = list(record.response_ids)
    if record.append_eos and tokenizer.eos_token_id is not None:
        values.append(int(tokenizer.eos_token_id))
    return values


class TargetScorer:
    """Teacher-forced and label-only probes against one fine-tuned target."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: torch.device,
        sead_samples: int,
        sead_temperature: float,
        seed: int,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.sead_samples = sead_samples
        self.sead_temperature = sead_temperature
        self.seed = seed
        self.cost_meter: CostMeter | None = None
        # Input embeddings can occupy >1 GB for an 8B model.  Keep them lazy:
        # only PETAL requests the target-only semantic proxy.
        self.embedding_weight: torch.Tensor | None = None

    def _target_embedding_weight(self) -> torch.Tensor | None:
        if self.embedding_weight is None:
            try:
                self.embedding_weight = self.model.get_input_embeddings().weight.detach()
            except (AttributeError, RuntimeError):
                return None
        return self.embedding_weight

    def _input_ids(self, record: SFTRecord, context_ids: Iterable[int] = ()) -> tuple[list[int], int]:
        prompt_ids = list(context_ids) + list(prompt_prefix_ids(record, self.tokenizer))
        return prompt_ids + _response_ids(record, self.tokenizer), len(prompt_ids)

    def _forward_response(
        self,
        record: SFTRecord,
        context_ids: Iterable[int] = (),
        need_similarity: bool = False,
        need_sead: bool = False,
    ) -> TokenStats:
        input_values, response_start = self._input_ids(record, context_ids)
        if len(input_values) < response_start + 1:
            raise ValueError(f"record {record.record_id} has no response tokens")
        if self.cost_meter is not None:
            self.cost_meter.forward(len(input_values))
        input_ids = torch.tensor([input_values], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).logits[0, :-1]
        labels = input_ids[0, 1:]
        # The first response token at input index response_start is predicted by
        # logits[response_start - 1].
        response_mask = torch.arange(labels.numel(), device=self.device) >= response_start - 1
        # Select response positions before converting to FP32.  ArXivTection
        # uses 2,048-token responses; converting the complete [sequence,
        # vocabulary] projection first needlessly creates a multi-GB FP32
        # copy, especially for ICP contexts containing an auxiliary probe.
        selected_logits = logits[response_mask]
        del logits
        selected_labels = labels[response_mask]
        token_logp_parts = []
        expected_parts = []
        variance_parts = []
        for chunk_logits, chunk_labels in zip(
            selected_logits.split(256), selected_labels.split(256)
        ):
            chunk_logits = chunk_logits.float()
            log_probs = F.log_softmax(chunk_logits, dim=-1)
            probs = log_probs.exp()
            chunk_logp = log_probs.gather(-1, chunk_labels[:, None]).squeeze(-1)
            chunk_expected = (probs * log_probs).sum(-1)
            chunk_variance = (
                (probs * log_probs.square()).sum(-1) - chunk_expected.square()
            )
            token_logp_parts.append(chunk_logp.cpu().numpy())
            expected_parts.append(chunk_expected.cpu().numpy())
            variance_parts.append(chunk_variance.cpu().numpy())
            del chunk_logits, log_probs, probs, chunk_logp
        token_logp = np.concatenate(token_logp_parts)
        expected = np.concatenate(expected_parts)
        variance = np.concatenate(variance_parts)

        log_similarity = None
        if need_similarity:
            predicted = selected_logits.argmax(-1)
            embedding_weight = self._target_embedding_weight()
            if embedding_weight is None:
                similarity = (predicted == selected_labels).float()
            else:
                target_vectors = embedding_weight[selected_labels].float()
                predicted_vectors = embedding_weight[predicted].float()
                similarity = F.cosine_similarity(target_vectors, predicted_vectors, dim=-1)
                similarity = ((similarity + 1.0) / 2.0).clamp_min(1e-12)
                similarity = torch.maximum(
                    similarity, (predicted == selected_labels).float()
                )
            log_similarity = similarity.clamp_min(1e-12).log().cpu().numpy()

        sampled_token_ids = None
        sead_log_density = None
        sead_lexical_density = None
        if need_sead:
            temperature = max(float(self.sead_temperature), 1e-4)
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.seed + int(record.record_id.encode().hex()[:8], 16) % 1_000_000)
            # Vectorize over token rows in bounded chunks.  Calling
            # ``multinomial`` once per token launches millions of tiny GPU
            # kernels for the 4,000-record audit, while chunking preserves the
            # same Monte Carlo estimator and keeps the vocabulary matrix
            # bounded for long ArXivTection responses.
            sampled_chunks = []
            for chunk in selected_logits.split(256):
                chunk_probs = F.softmax(chunk.float() / temperature, dim=-1)
                sampled_chunks.append(
                    torch.multinomial(
                        chunk_probs,
                        self.sead_samples,
                        replacement=True,
                        generator=generator,
                    ).cpu().numpy()
                )
                del chunk_probs
            sampled_token_ids = np.concatenate(sampled_chunks, axis=0)
            # This is SEAD's official surrogate-free frequency estimator.  The
            # NLI/semantic variant is intentionally not enabled here because
            # the user requested a single target-model implementation.
            sead_log_density, sead_lexical_density = sead_score(
                sampled_token_ids, selected_labels.cpu().numpy()
            )

        del selected_logits, input_ids
        return TokenStats(
            token_ids=selected_labels.cpu().numpy(),
            token_logp=token_logp,
            expected_logp=expected,
            variance_logp=variance,
            log_similarity=log_similarity,
            sampled_token_ids=sampled_token_ids,
            sead_log_density=sead_log_density,
            sead_lexical_density=sead_lexical_density,
        )

    def stats(
        self, record: SFTRecord, need_similarity: bool = False, need_sead: bool = False
    ) -> TokenStats:
        return self._forward_response(
            record, need_similarity=need_similarity, need_sead=need_sead
        )

    def mean_ll(self, record: SFTRecord, context_ids: Iterable[int] = ()) -> float:
        stats = self._forward_response(record, context_ids=context_ids)
        return mean_log_likelihood(stats.token_logp)

    @lru_cache(maxsize=None)
    def _probe_vector(self, record: SFTRecord) -> torch.Tensor | None:
        weight = self._target_embedding_weight()
        if weight is None:
            return None
        ids = list(prompt_prefix_ids(record, self.tokenizer)) + list(record.response_ids)
        vectors = weight[torch.tensor(ids, dtype=torch.long, device=weight.device)].float()
        return F.normalize(vectors.mean(dim=0), dim=0)

    def probe_similarity(self, left: SFTRecord, right: SFTRecord) -> float:
        """Rank ICP auxiliary probes with the target's own input embeddings.

        ICP-MIA's reference-data path retrieves semantically similar
        demonstrations before measuring the optimization gap.  The official
        implementation uses an external sentence encoder; this target-only
        version uses the fine-tuned target's input-embedding space instead.
        It falls back to token Jaccard only for architectures without input
        embeddings.
        """

        left_vector = self._probe_vector(left)
        right_vector = self._probe_vector(right)
        if left_vector is None or right_vector is None:
            return _jaccard(left.response_ids, right.response_ids)
        return float(torch.dot(left_vector, right_vector).detach().cpu())

    def build_probe_matrix(
        self, records: list[SFTRecord]
    ) -> torch.Tensor | None:
        """Precompute normalized target-embedding probes for ICP retrieval.

        The original per-candidate implementation synchronizes one GPU dot
        product to the CPU for every auxiliary record.  ICP-MIA ranks the same
        auxiliary pool repeatedly, so retaining the target-only vectors and
        using one matrix product per audit record is mathematically identical
        and avoids millions of tiny synchronization points.
        """

        if self._target_embedding_weight() is None:
            return None
        with torch.inference_mode():
            return torch.stack([self._probe_vector(record) for record in records])

    def top_probe_records(
        self,
        record: SFTRecord,
        candidates: list[SFTRecord],
        probe_matrix: torch.Tensor | None,
        top_k: int,
    ) -> list[SFTRecord]:
        """Return the target-only nearest auxiliary records for ICP-MIA."""

        limit = min(max(1, int(top_k)), len(candidates))
        if probe_matrix is None:
            return sorted(
                candidates,
                key=lambda candidate: self.probe_similarity(record, candidate),
                reverse=True,
            )[:limit]
        with torch.inference_mode():
            scores = torch.mv(probe_matrix, self._probe_vector(record))
            indices = torch.topk(scores, k=limit, largest=True).indices
        return [candidates[int(index)] for index in indices.detach().cpu().tolist()]

    def _max_context(self) -> int:
        value = getattr(self.model.config, "max_position_embeddings", None)
        if value is None or value > 1_000_000:
            value = 32768
        return int(value)

    def fit_context(self, context_ids: Iterable[int], record: SFTRecord) -> list[int]:
        context = list(context_ids)
        prompt = list(prompt_prefix_ids(record, self.tokenizer))
        response = _response_ids(record, self.tokenizer)
        budget = max(0, self._max_context() - len(prompt) - len(response))
        return context[-budget:] if budget else []

    def generate(
        self,
        input_ids: Iterable[int],
        max_new_tokens: int,
        seed: int,
        sample: bool,
        num_return_sequences: int = 1,
    ) -> list[list[int]]:
        values = list(input_ids)
        if max_new_tokens <= 0:
            return [[] for _ in range(num_return_sequences)]
        tensor = torch.tensor([values], dtype=torch.long, device=self.device)
        attention = torch.ones_like(tensor)
        kwargs: dict[str, Any] = {
            "input_ids": tensor,
            "attention_mask": attention,
            "max_new_tokens": max_new_tokens,
            "do_sample": sample,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": int(self.tokenizer.pad_token_id),
        }
        if sample:
            kwargs.update({"temperature": 1.0, "top_k": 50, "top_p": 1.0})
        # Transformers 5.16 routes sampling through torch's global RNG and
        # does not accept ``generator=`` as a generate() kwarg.  Forking the
        # relevant RNG makes each probe reproducible without perturbing the
        # caller's global random state.
        cuda_devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            with torch.inference_mode():
                output = self.model.generate(**kwargs)
        width = tensor.shape[1]
        sequences = [row[width:].detach().cpu().tolist() for row in output]
        if self.cost_meter is not None:
            self.cost_meter.generation([values], [sequences], self._generation_eos_ids())
        return sequences

    def generate_batch(
        self,
        input_ids: list[list[int]],
        max_new_tokens: int,
        seed: int,
        sample: bool,
        num_return_sequences: int = 1,
    ) -> list[list[list[int]]]:
        """Generate for left-padded inputs in one target-only batch.

        The caller may trim each returned continuation to its own requested
        length.  Greedy generation is unchanged by that trimming, while the
        shared batch seed keeps sampled probes reproducible as a group.
        """

        if not input_ids:
            return []
        if max_new_tokens <= 0:
            return [[[] for _ in range(num_return_sequences)] for _ in input_ids]
        width = max(len(values) for values in input_ids)
        pad_id = int(self.tokenizer.pad_token_id)
        tensor = torch.full(
            (len(input_ids), width), pad_id, dtype=torch.long, device=self.device
        )
        attention = torch.zeros_like(tensor)
        for row, values in enumerate(input_ids):
            length = len(values)
            tensor[row, width - length :] = torch.tensor(
                values, dtype=torch.long, device=self.device
            )
            attention[row, width - length :] = 1
        kwargs: dict[str, Any] = {
            "input_ids": tensor,
            "attention_mask": attention,
            "max_new_tokens": max_new_tokens,
            "do_sample": sample,
            "num_return_sequences": num_return_sequences,
            "pad_token_id": pad_id,
        }
        if sample:
            kwargs.update({"temperature": 1.0, "top_k": 50, "top_p": 1.0})
        cuda_devices = [self.device.index] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            with torch.inference_mode():
                output = self.model.generate(**kwargs)
        continuations = output[:, width:].detach().cpu().tolist()
        grouped = []
        for row in range(len(input_ids)):
            start = row * num_return_sequences
            grouped.append(
                continuations[start : start + num_return_sequences]
            )
        if self.cost_meter is not None:
            self.cost_meter.generation(input_ids, grouped, self._generation_eos_ids())
        return grouped

    def _generation_eos_ids(self):
        config = getattr(self.model, "generation_config", None)
        if config is None:
            config = getattr(getattr(self.model, "model", None), "generation_config", None)
        ids = getattr(config, "eos_token_id", None)
        if ids is None:
            ids = self.tokenizer.eos_token_id
        return set(ids if isinstance(ids, (list, tuple)) else [ids]) if ids is not None else set()


def _aux_prefix(auxiliary: list[SFTRecord], tokenizer: Any, shots: int) -> list[int]:
    result: list[int] = []
    for record in auxiliary[: max(0, shots)]:
        result.extend(list(prompt_prefix_ids(record, tokenizer)))
        result.extend(_response_ids(record, tokenizer))
    return result


def _jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / max(1, len(a | b))


def _response_prefix_text(record: SFTRecord, tokenizer: Any, ratio: float) -> tuple[list[int], str, str]:
    """Split once for generation inputs, perturbation text, and reference suffix."""
    response = list(record.response_ids)
    cut = min(len(response), max(1, int(round(len(response) * ratio))))
    prefix_ids = list(prompt_prefix_ids(record, tokenizer)) + response[:cut]
    return (
        prefix_ids,
        tokenizer.decode(response[:cut], skip_special_tokens=True),
        tokenizer.decode(response[cut:], skip_special_tokens=True),
    )


def _perturb_words(text: str, kind: str, rate: float, rng: np.random.Generator) -> str:
    words = text.split()
    if len(words) < 2:
        return text
    count = max(1, int(round(len(words) * rate)))
    if kind == "rs":
        for _ in range(count):
            left = int(rng.integers(0, len(words) - 1))
            words[left], words[left + 1] = words[left + 1], words[left]
    else:
        for index in rng.choice(len(words), size=min(count, len(words)), replace=False):
            replacement = words[(int(index) + 1) % len(words)]
            words[int(index)] = replacement
    return " ".join(words)


def _render_report(
    output_dir: Path,
    protocol: dict[str, Any],
    scores: dict[str, list[float]],
    labels: np.ndarray,
    costs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    members = labels == 1
    nonmembers = labels == 0
    metrics: dict[str, Any] = {}
    for name, values in scores.items():
        values_array = np.asarray(values, dtype=np.float64)
        member = values_array[members]
        nonmember = values_array[nonmembers]
        tpr10, fpr10, threshold10 = upper_tail_tpr(member, nonmember, 0.10)
        tpr01, fpr01, threshold01 = upper_tail_tpr(member, nonmember, 0.01)
        metrics[name] = {
            "auc": rank_auc(member, nonmember),
            "tpr@10%fpr": tpr10,
            "actual_fpr@10%": fpr10,
            "threshold@10%": threshold10,
            "tpr@1%fpr": tpr01,
            "actual_fpr@1%": fpr01,
            "threshold@1%": threshold01,
            "member_mean": float(np.mean(member)),
            "nonmember_mean": float(np.mean(nonmember)),
        }
    artifact = {"protocol": protocol, "metrics": metrics, "scores": scores}
    if costs is not None:
        artifact["costs"] = costs
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_metrics.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Target-only membership-inference baselines",
        "",
        f"- Run: `{protocol['run_dir']}`",
        "- Model source: pretrained target checkpoint." if protocol.get("training_regime") == "pretraining" else "- Model source: saved fine-tuned target checkpoint only.",
        "- Reference model: none. Draft model: not loaded.",
        "",
        "| Method | AUC | TPR@10%FPR | actual FPR | TPR@1%FPR | actual FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in metrics.items():
        lines.append(
            f"| `{name}` | {result['auc']:.4f} | {result['tpr@10%fpr']:.4f} "
            f"| {result['actual_fpr@10%']:.4f} | {result['tpr@1%fpr']:.4f} "
            f"| {result['actual_fpr@1%']:.4f} |"
        )
    lines.extend(cost_table(costs))
    (output_dir / "BASELINE_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return artifact


def _normalise_methods(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return METHODS
    requested = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not requested:
        raise ValueError("at least one baseline method is required")
    unknown = sorted(set(requested) - set(METHODS))
    if unknown:
        raise ValueError(f"unknown methods {unknown}; choose from {', '.join(METHODS)}")
    return tuple(dict.fromkeys(requested))


def _score_methods(args, progress, scorer, all_records, auxiliary, tokenizer, methods,
                   *, reference_cache: dict):
    needs_aux = bool({"recall", "icp_mia", "petal"} & set(methods))
    if needs_aux and not auxiliary:
        raise RuntimeError("the requested target-only methods need non-member auxiliary records")
    icp_probe_matrix = (
        scorer.build_probe_matrix(auxiliary) if "icp_mia" in methods else None
    )
    recall_prefix = _aux_prefix(auxiliary, tokenizer, args.recall_shots) if "recall" in methods else []

    petal_slope = petal_intercept = None
    if "petal" in methods:
        calibration_x: list[float] = []
        calibration_y: list[float] = []
        for record in progress.track(auxiliary, "petal calibration"):
            stats = scorer.stats(record, need_similarity=True)
            if stats.log_similarity is not None:
                calibration_x.extend(stats.log_similarity.tolist())
                calibration_y.extend(stats.token_logp.tolist())
        _, petal_slope, petal_intercept = petal_score(
            calibration_x, calibration_y
        )

    scores = {name: [] for name in methods}
    teacher_forced_methods = {
        "loss",
        "min_k_prob",
        "min_k_pp",
        "petal",
        "sead",
        "recall",
        "icp_mia",
    } & set(methods)
    for row_index, audit_record in enumerate(progress.track(all_records if teacher_forced_methods else [], "teacher-forced scoring")):
        record = audit_record.record
        need_similarity = "petal" in methods
        need_sead = "sead" in methods
        stats = scorer.stats(record, need_similarity=need_similarity, need_sead=need_sead)
        base_ll = mean_log_likelihood(stats.token_logp)

        if "loss" in methods:
            scores["loss"].append(base_ll)
        if "min_k_prob" in methods:
            scores["min_k_prob"].append(min_k_prob_score(stats.token_logp, args.k_percent))
        if "min_k_pp" in methods:
            scores["min_k_pp"].append(
                min_k_plus_plus_score(
                    stats.token_logp,
                    stats.expected_logp,
                    stats.variance_logp,
                    args.k_percent,
                )
            )
        if "petal" in methods:
            assert stats.log_similarity is not None
            value, _, _ = petal_score(
                stats.log_similarity,
                stats.token_logp,
                slope=petal_slope,
                intercept=petal_intercept,
            )
            scores["petal"].append(value)
        if "sead" in methods:
            assert stats.sead_log_density is not None
            scores["sead"].append(stats.sead_log_density)
        if "recall" in methods:
            context = scorer.fit_context(recall_prefix, record)
            conditional_ll = scorer.mean_ll(record, context)
            scores["recall"].append(recall_score(base_ll, conditional_ll))
        if "icp_mia" in methods:
            candidates = scorer.top_probe_records(
                record, auxiliary, icp_probe_matrix, args.icp_top_k
            )
            candidate_scores = []
            for candidate in candidates:
                context = scorer.fit_context(_response_ids(candidate, tokenizer), record)
                candidate_scores.append(icp_score(base_ll, scorer.mean_ll(record, context)))
            if args.icp_aggregation == "min":
                scores["icp_mia"].append(float(np.min(candidate_scores)))
            elif args.icp_aggregation == "max":
                scores["icp_mia"].append(float(np.max(candidate_scores)))
            else:
                scores["icp_mia"].append(float(np.mean(candidate_scores)))

    generation_methods = {"ws", "rs", "bt", "samia"} & set(methods)
    if generation_methods:
        generation_inputs: list[list[int]] = []
        prefix_texts: list[str] = []
        suffixes: list[str] = []
        max_news: list[int] = []
        for audit_record in all_records:
            generation_input, prefix_text, suffix = _response_prefix_text(
                audit_record.record, tokenizer, args.prefix_ratio
            )
            generation_inputs.append(generation_input)
            prefix_texts.append(prefix_text)
            suffixes.append(suffix)
            max_news.append(
                max(1, len(tokenizer(suffix, add_special_tokens=False).input_ids))
            )

        def batches(values: list[Any], stage: str) -> Iterable[tuple[int, list[Any]]]:
            for start in progress.track(list(range(0, len(values), args.generation_batch_size)), stage, unit="batches"):
                yield start, values[start : start + max(1, args.generation_batch_size)]

        def trim(values: list[int], length: int) -> list[int]:
            return values[:length]

        if "samia" in generation_methods:
            for start, batch_inputs in batches(generation_inputs, "samia"):
                end = start + len(batch_inputs)
                generated = scorer.generate_batch(
                    batch_inputs,
                    max(max_news[start:end]),
                    args.seed + start,
                    sample=True,
                    num_return_sequences=args.samia_samples,
                )
                for offset, rows in enumerate(generated):
                    suffix = suffixes[start + offset]
                    generated_text = [
                        tokenizer.decode(
                            trim(ids, max_news[start + offset]),
                            skip_special_tokens=True,
                        )
                        for ids in rows
                    ]
                    scores["samia"].append(
                        float(
                            np.mean(
                                [rouge1_recall(text, suffix) for text in generated_text]
                            )
                        )
                    )

        if {"ws", "rs", "bt"} & generation_methods:
            if "texts" in reference_cache:
                baseline_texts = reference_cache["texts"]
                if len(baseline_texts) != len(all_records):
                    raise ValueError("robustness reference cache has the wrong record count")
            else:
                baseline_texts = [""] * len(all_records)
                for start, batch_inputs in batches(generation_inputs, "shared robustness reference generation"):
                    end = start + len(batch_inputs)
                    generated = scorer.generate_batch(
                        batch_inputs,
                        max(max_news[start:end]),
                        args.seed + 10_000 + start,
                        sample=False,
                    )
                    for offset, rows in enumerate(generated):
                        baseline_texts[start + offset] = tokenizer.decode(
                            trim(rows[0], max_news[start + offset]),
                            skip_special_tokens=True,
                        )
                reference_cache["texts"] = baseline_texts

        for kind in ("ws", "rs"):
            if kind not in generation_methods:
                continue
            perturbed_inputs: list[list[int]] = []
            for row_index, audit_record in enumerate(all_records):
                rng = np.random.default_rng(args.seed + 20_000 + row_index)
                perturbed_text = _perturb_words(
                    prefix_texts[row_index], kind, args.perturbation_rate, rng
                )
                perturbed_inputs.append(
                    list(prompt_prefix_ids(audit_record.record, tokenizer))
                    + list(tokenizer(perturbed_text, add_special_tokens=False).input_ids)
                )
            for start, batch_inputs in batches(perturbed_inputs, kind):
                end = start + len(batch_inputs)
                generated = scorer.generate_batch(
                    batch_inputs,
                    max(max_news[start:end]),
                    args.seed + 30_000 + start,
                    sample=False,
                )
                for offset, rows in enumerate(generated):
                    perturbed_text = tokenizer.decode(
                        trim(rows[0], max_news[start + offset]),
                        skip_special_tokens=True,
                    )
                    scores[kind].append(
                        rouge1_recall(perturbed_text, baseline_texts[start + offset])
                    )

        if "bt" in generation_methods:
            rewrite_inputs = [
                tokenizer(
                    "Rewrite the following passage in different words while preserving its meaning.\n"
                    f"Passage: {prefix_text}\nRewrite:",
                    return_tensors="pt",
                ).input_ids[0].tolist()
                for prefix_text in prefix_texts
            ]
            rewritten: list[list[int]] = [[] for _ in all_records]
            for start, batch_inputs in batches(rewrite_inputs, "bt rewrite"):
                end = start + len(batch_inputs)
                generated = scorer.generate_batch(
                    batch_inputs,
                    max(max_news[start:end]),
                    args.seed + 40_000 + start,
                    sample=False,
                )
                for offset, rows in enumerate(generated):
                    rewritten[start + offset] = trim(
                        rows[0], max_news[start + offset]
                    )
            bt_inputs = [
                list(prompt_prefix_ids(audit_record.record, tokenizer)) + rewritten[index]
                for index, audit_record in enumerate(all_records)
            ]
            for start, batch_inputs in batches(bt_inputs, "bt scoring"):
                end = start + len(batch_inputs)
                generated = scorer.generate_batch(
                    batch_inputs,
                    max(max_news[start:end]),
                    args.seed + 50_000 + start,
                    sample=False,
                )
                for offset, rows in enumerate(generated):
                    scores["bt"].append(
                        rouge1_recall(
                            tokenizer.decode(
                                trim(rows[0], max_news[start + offset]),
                                skip_special_tokens=True,
                            ),
                            baseline_texts[start + offset],
                        )
                    )

    return scores


def _run(args: argparse.Namespace, progress: RunProgress) -> None:
    methods = _normalise_methods(args.methods)
    if not 0.0 < args.prefix_ratio < 1.0:
        raise ValueError("--prefix-ratio must be in (0, 1)")
    if args.sead_samples <= 0 or args.samia_samples <= 0:
        raise ValueError("sampling counts must be positive")
    if args.generation_batch_size <= 0:
        raise ValueError("--generation-batch-size must be positive")

    pretraining = getattr(args, "pretraining_manifest", None)
    output_dir = _resolve(args.output_dir).resolve()
    progress.event("loading tokenizer and audit records")
    if pretraining is not None:
        if args.pool_path is not None:
            raise ValueError("--pool-path cannot override a frozen pretraining manifest")
        from ..pretraining.data import load_evaluation
        evaluation = load_evaluation(_resolve(pretraining))
        run_dir = evaluation.manifest_path.parent
        cfg = evaluation.config
        tokenizer = evaluation.tokenizer
        members = [AuditRecord(record, 1) for record in evaluation.members]
        nonmembers = [AuditRecord(record, 0) for record in evaluation.nonmembers]
        auxiliary = evaluation.auxiliary
        split_metadata = evaluation.manifest
    else:
        run_dir = _resolve(args.run_dir).resolve()
        cfg = load_run_config(run_dir)
        tokenizer = _target_tokenizer(run_dir, cfg)
        members, nonmembers, auxiliary, split_metadata = load_audit_records(
            run_dir, cfg, tokenizer, args.pool_path
        )
    complete_records = members + nonmembers
    if not 0 <= args.record_start <= len(complete_records):
        raise ValueError("--record-start must be within the audit-record range")
    record_end = len(complete_records) if args.record_end is None else args.record_end
    if not args.record_start <= record_end <= len(complete_records):
        raise ValueError("--record-end must satisfy start <= end <= audit-record count")
    all_records = complete_records[args.record_start:record_end]
    labels = np.asarray([row.label for row in all_records], dtype=np.int64)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    set_seed(args.seed)
    progress.event("loading target model")
    if pretraining is not None:
        from ..pretraining.data import load_model
        model = load_model(evaluation.manifest["models"]["target"], device, args.attn_implementation, len(tokenizer))
    else:
        model = load_finetuned_model(
            run_dir, cfg.target_model, device, attn_implementation=args.attn_implementation
        )
    protocol = {
        "seed": args.seed,
        "attn_implementation": args.attn_implementation,
        "run_dir": str(run_dir),
        "benchmark": cfg.benchmark,
        "target_model": cfg.target_model,
        "target_checkpoint": str(
            next(
                (run_dir / directory / "target"
                 for directory in ("checkpoints", "adapters")
                 if (run_dir / directory / "target").exists()),
            run_dir / "checkpoints" / "target",
        )
        ),
        "methods": list(methods),
        "n_member": int(np.sum(labels == 1)),
        "n_nonmember": int(np.sum(labels == 0)),
        "n_auxiliary": len(auxiliary),
        "record_start": args.record_start,
        "record_end": record_end,
        "complete_n_member": len(members),
        "complete_n_nonmember": len(nonmembers),
        "k_percent": args.k_percent,
        "recall_shots": args.recall_shots,
        "icp_top_k": args.icp_top_k,
        "sead_samples": args.sead_samples,
        "sead_temperature": args.sead_temperature,
        "samia_samples": args.samia_samples,
        "generation_batch_size": args.generation_batch_size,
        "prefix_ratio": args.prefix_ratio,
        "split_metadata": split_metadata,
        "restrictions": {
            "reference_model": False,
            "unfinetuned_target_scored": False,
            "draft_model_loaded": False,
            "petal_calibration": "fine-tuned target on auxiliary records",
            "sead_estimator": "target-only Monte Carlo frequency density",
        },
    }
    if pretraining is not None:
        protocol.update(training_regime="pretraining", target_checkpoint=evaluation.manifest["models"]["target"],
                        token_contract=evaluation.manifest["token_contract"])
        protocol["restrictions"].update(unfinetuned_target_scored=True,
            petal_calibration="pretrained target on disjoint MIMIR auxiliary nonmembers")
    else:
        protocol["training_regime"] = "controlled_sft"
    progress.configure(protocol, labels, [row.record.record_id for row in all_records])
    progress.event("audit ready", records=len(all_records), methods=list(methods))

    warmup_records = getattr(args, "cost_warmup_records", 1)
    if warmup_records < 0 or not all_records:
        raise ValueError("warmup must be nonnegative and audit must be nonempty")
    protocol["cost_measurement"] = cost_protocol(model, device, min(warmup_records, len(all_records)), args.generation_batch_size)
    scores, costs = {}, {}
    reference_cache: dict[str, list[str]] = {}
    for method in methods:
        progress.active_method = method
        scorer = TargetScorer(model, tokenizer, device, args.sead_samples, args.sead_temperature, args.seed)
        try:
            progress.event("method warmup", method=method)
            for row in all_records[:warmup_records]:
                scorer.stats(row.record)
            set_seed(args.seed)
            meter = CostMeter(device, len(all_records))
            scorer.cost_meter = meter
            progress.event("method started", method=method)
            reused_reference = method in ("ws", "rs", "bt") and "texts" in reference_cache
            with meter.measure():
                values = _score_methods(args, progress, scorer, all_records, auxiliary, tokenizer,
                                        (method,), reference_cache=reference_cache)
            scores[method] = values[method]
            measured_cost = meter.result()
            if method in ("ws", "rs", "bt"):
                if reused_reference:
                    costs[method] = {"cost_basis": "physical_incremental", "reference_reused": True,
                                     **{"physical_incremental_" + key: value
                                        for key, value in measured_cost.items()}}
                else:
                    costs[method] = {**measured_cost, "cost_basis": "standalone_measured",
                                     "reference_reused": False}
            else:
                costs[method] = measured_cost
            progress.save_method(method, scores[method], cost=costs[method])
        finally:
            scorer._probe_vector.cache_clear()
            del scorer

    write_cost_report(output_dir, protocol, costs)
    # A parallel generation shard may intentionally contain only one class.
    # Preserve its raw scores for the parent process instead of attempting an
    # AUC/TPR report that has no meaningful negative (or positive) examples.
    if np.unique(labels).size >= 2:
        _render_report(output_dir, protocol, scores, labels, costs=costs)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "baseline_metrics.json").write_text(
            json.dumps(
                {"protocol": protocol, "metrics": {}, "scores": scores, "costs": costs},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    np.savez_compressed(
        output_dir / "baseline_scores.npz",
        labels=labels,
        record_ids=np.asarray([row.record.record_id for row in all_records]),
        **{name: np.asarray(values, dtype=np.float32) for name, values in scores.items()},
    )
    print(json.dumps({"output_dir": str(output_dir), "methods": list(scores)}, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    with RunProgress(_resolve(args.output_dir), args.progress_interval) as progress:
        _run(args, progress)


if __name__ == "__main__":
    main()
