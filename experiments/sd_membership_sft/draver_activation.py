from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .audit import auc_rank, bottom_k_indices, cap_selected_positions, metric_row
from .data import SFTRecord, collate_sft, make_sft_example


ACTIVATION_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
ACTIVATION_FEATURE_NAMES = (
    "mean",
    "std",
    "min",
    "max",
    "q10",
    "q25",
    "q50",
    "q75",
    "q90",
    "rms",
    "delta_rms",
    "delta_cosine",
)


def _loader(
    records: list[SFTRecord], tokenizer: Any, batch_size: int
) -> DataLoader:
    examples = [make_sft_example(record, tokenizer) for record in records]
    return DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda rows: collate_sft(rows, int(tokenizer.pad_token_id)),
    )


def _autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", dtype=torch.bfloat16)


def _pad_token_batches(batches: list[np.ndarray], fill: float = np.nan) -> np.ndarray:
    width = max(batch.shape[1] for batch in batches)
    trailing = batches[0].shape[2:]
    result = np.full(
        (sum(len(batch) for batch in batches), width, *trailing),
        fill,
        dtype=np.float32,
    )
    start = 0
    for batch in batches:
        result[start : start + len(batch), : batch.shape[1]] = batch
        start += len(batch)
    return result


@torch.no_grad()
def extract_target_token_outputs(
    model: torch.nn.Module,
    records: list[SFTRecord],
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Extract only verifier outputs needed by the local protocol simulator."""
    model.eval()
    logp_batches: list[np.ndarray] = []
    top1_batches: list[np.ndarray] = []
    top1_token_batches: list[np.ndarray] = []
    for batch in _loader(records, tokenizer, batch_size):
        batch = {key: value.to(device) for key, value in batch.items()}
        with _autocast(device):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
        logits = output.logits[:, :-1].float()
        labels = batch["labels"][:, 1:]
        valid = labels.ne(-100)
        safe_labels = labels.clamp_min(0)
        logp = logits.log_softmax(dim=-1).gather(
            -1, safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        top1 = logits.argmax(dim=-1).eq(safe_labels)
        max_response = int(valid.sum(dim=1).max().item())
        batch_logp = np.full((len(labels), max_response), np.nan, dtype=np.float32)
        batch_top1 = np.full_like(batch_logp, np.nan)
        batch_top1_token = np.full_like(batch_logp, np.nan)
        for row in range(len(labels)):
            positions = valid[row].nonzero(as_tuple=False).flatten()
            count = len(positions)
            batch_logp[row, :count] = logp[row, positions].cpu().numpy()
            batch_top1[row, :count] = top1[row, positions].float().cpu().numpy()
            batch_top1_token[row, :count] = (
                logits[row, positions].argmax(dim=-1).float().cpu().numpy()
            )
        logp_batches.append(batch_logp)
        top1_batches.append(batch_top1)
        top1_token_batches.append(batch_top1_token)
        del output, logits, logp
    return {
        "token_logp": _pad_token_batches(logp_batches),
        "top1_match": _pad_token_batches(top1_batches),
        "top1_token_id": _pad_token_batches(top1_token_batches),
    }


@torch.no_grad()
def extract_draft_activation_outputs(
    model: torch.nn.Module,
    records: list[SFTRecord],
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Extract token-aligned, all-layer draft statistics for SD verification.

    NART summarizes the last token across all model layers. Here every response
    prediction position is retained because speculative verification is
    token-aligned. Two layer-transition statistics are appended so that the
    verifier transcript can condition a representation trajectory rather than
    a single terminal activation.
    """
    model.eval()
    logp_batches: list[np.ndarray] = []
    entropy_batches: list[np.ndarray] = []
    top1_token_batches: list[np.ndarray] = []
    activation_batches: list[np.ndarray] = []
    quantiles = torch.tensor(ACTIVATION_QUANTILES, device=device)

    for batch in _loader(records, tokenizer, batch_size):
        batch = {key: value.to(device) for key, value in batch.items()}
        with _autocast(device):
            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
        logits = output.logits[:, :-1].float()
        labels = batch["labels"][:, 1:]
        valid = labels.ne(-100)
        safe_labels = labels.clamp_min(0)
        log_probs = logits.log_softmax(dim=-1)
        probs = log_probs.exp()
        token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        # hidden_states contains the embedding output followed by one tensor per
        # transformer layer. The activation at position i predicts token i+1.
        states = torch.stack(
            [hidden[:, :-1].float() for hidden in output.hidden_states[1:]], dim=1
        )
        previous = torch.stack(
            [hidden[:, :-1].float() for hidden in output.hidden_states[:-1]], dim=1
        )
        percentiles = torch.quantile(states, quantiles, dim=-1).permute(1, 2, 3, 0)
        mean = states.mean(dim=-1)
        std = states.std(dim=-1, unbiased=False)
        rms = states.square().mean(dim=-1).sqrt()
        delta_rms = (states - previous).square().mean(dim=-1).sqrt()
        delta_cosine = F.cosine_similarity(states, previous, dim=-1)
        summaries = torch.cat(
            [
                mean.unsqueeze(-1),
                std.unsqueeze(-1),
                states.amin(dim=-1).unsqueeze(-1),
                states.amax(dim=-1).unsqueeze(-1),
                percentiles,
                rms.unsqueeze(-1),
                delta_rms.unsqueeze(-1),
                delta_cosine.unsqueeze(-1),
            ],
            dim=-1,
        ).permute(0, 2, 1, 3)

        max_response = int(valid.sum(dim=1).max().item())
        batch_logp = np.full((len(labels), max_response), np.nan, dtype=np.float32)
        batch_entropy = np.full_like(batch_logp, np.nan)
        batch_top1_token = np.full_like(batch_logp, np.nan)
        batch_activations = np.full(
            (
                len(labels),
                max_response,
                summaries.shape[2],
                summaries.shape[3],
            ),
            np.nan,
            dtype=np.float32,
        )
        for row in range(len(labels)):
            positions = valid[row].nonzero(as_tuple=False).flatten()
            count = len(positions)
            batch_logp[row, :count] = token_logp[row, positions].cpu().numpy()
            batch_entropy[row, :count] = entropy[row, positions].cpu().numpy()
            batch_top1_token[row, :count] = (
                logits[row, positions].argmax(dim=-1).float().cpu().numpy()
            )
            batch_activations[row, :count] = summaries[row, positions].cpu().numpy()
        logp_batches.append(batch_logp)
        entropy_batches.append(batch_entropy)
        top1_token_batches.append(batch_top1_token)
        activation_batches.append(batch_activations)
        del output, logits, log_probs, probs, states, previous, summaries

    return {
        "token_logp": _pad_token_batches(logp_batches),
        "entropy": _pad_token_batches(entropy_batches),
        "top1_token_id": _pad_token_batches(top1_token_batches),
        "activation_stats": _pad_token_batches(activation_batches),
    }


@torch.no_grad()
def extract_pair_alignment_outputs(
    draft: torch.nn.Module,
    target: torch.nn.Module,
    records: list[SFTRecord],
    tokenizer: Any,
    device: torch.device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Measure exact distributional alignment for a deployable SD pair.

    For ordinary speculative sampling, the expected one-token acceptance under
    the draft distribution is ``sum_v min(p(v), q(v)) = 1 - TV(p, q)``.  This
    routine computes that quantity on every teacher-forced response position,
    rather than using the fixed-candidate acceptance statistic used by the
    membership audit itself.
    """
    draft.eval()
    target.eval()
    rows: dict[str, list[float]] = {
        "exact_acceptance": [],
        "top1_agreement": [],
        "candidate_logp_mean_abs_gap": [],
        "candidate_logp_rmse": [],
        "candidate_target_minus_draft_logp": [],
    }
    for batch in _loader(records, tokenizer, batch_size):
        batch = {key: value.to(device) for key, value in batch.items()}
        with _autocast(device):
            draft_output = draft(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
            target_output = target(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                use_cache=False,
            )
        draft_logits = draft_output.logits[:, :-1]
        target_logits = target_output.logits[:, :-1]
        if draft_logits.shape[-1] != target_logits.shape[-1]:
            raise ValueError(
                "Exact SD acceptance requires draft and target to share a vocabulary"
            )
        labels = batch["labels"][:, 1:]
        valid = labels.ne(-100)
        safe_labels = labels.clamp_min(0)
        draft_logp = F.log_softmax(draft_logits, dim=-1, dtype=torch.float32)
        target_logp = F.log_softmax(target_logits, dim=-1, dtype=torch.float32)
        shared_mass = torch.minimum(draft_logp, target_logp).exp().sum(dim=-1)
        top1_equal = draft_logits.argmax(dim=-1).eq(target_logits.argmax(dim=-1))
        draft_candidate = draft_logp.gather(
            -1, safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        target_candidate = target_logp.gather(
            -1, safe_labels.unsqueeze(-1)
        ).squeeze(-1)
        gap = target_candidate - draft_candidate
        for row in range(len(labels)):
            positions = valid[row].nonzero(as_tuple=False).flatten()
            row_gap = gap[row, positions]
            rows["exact_acceptance"].append(
                float(shared_mass[row, positions].mean().cpu())
            )
            rows["top1_agreement"].append(
                float(top1_equal[row, positions].float().mean().cpu())
            )
            rows["candidate_logp_mean_abs_gap"].append(
                float(row_gap.abs().mean().cpu())
            )
            rows["candidate_logp_rmse"].append(
                float(row_gap.square().mean().sqrt().cpu())
            )
            rows["candidate_target_minus_draft_logp"].append(
                float(row_gap.mean().cpu())
            )
        del (
            draft_output,
            target_output,
            draft_logits,
            target_logits,
            draft_logp,
            target_logp,
            shared_mass,
        )
    return {
        name: np.asarray(values, dtype=np.float32) for name, values in rows.items()
    }


def make_audit_split(
    n_members: int, n_nonmembers: int, per_class: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    positive = rng.permutation(n_members)
    negative = rng.permutation(n_nonmembers) + n_members
    calibration = np.concatenate([positive[:per_class], negative[:per_class]])
    test = np.concatenate([positive[per_class:], negative[per_class:]])
    rng.shuffle(calibration)
    rng.shuffle(test)
    return calibration, test


def nested_calibration_subset(
    calibration: np.ndarray, labels: np.ndarray, per_class: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positive = rng.permutation(np.sort(calibration[labels[calibration] == 1]))
    negative = rng.permutation(np.sort(calibration[labels[calibration] == 0]))
    if per_class > min(len(positive), len(negative)):
        raise ValueError("few-shot subset exceeds the available calibration set")
    result = np.concatenate([positive[:per_class], negative[:per_class]])
    rng.shuffle(result)
    return result


def gather_tokens(values: np.ndarray, positions: np.ndarray) -> np.ndarray:
    return values[np.arange(len(values))[:, None], positions]


def sample_acceptance_rates(
    target_logp: np.ndarray,
    draft_logp: np.ndarray,
    selected: np.ndarray,
    repeats: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    target = gather_tokens(target_logp, selected).astype(np.float64)
    draft = gather_tokens(draft_logp, selected).astype(np.float64)
    alpha = np.minimum(1.0, np.exp(np.clip(target - draft, -50.0, 50.0)))
    counts = np.random.default_rng(seed).binomial(repeats, alpha)
    observed = (counts + 0.5) / (repeats + 1.0)
    return observed.astype(np.float32), alpha.astype(np.float32)


def _transcript_summary(acceptance: np.ndarray) -> np.ndarray:
    """Summarize verifier feedback without adding white-box draft scores."""
    clipped = np.clip(acceptance, 1e-5, 1.0 - 1e-5)
    quantiles = np.quantile(clipped, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1).T
    logit = np.log(clipped) - np.log1p(-clipped)
    return np.column_stack(
        [
            clipped.mean(axis=1),
            clipped.std(axis=1),
            clipped.min(axis=1),
            quantiles,
            clipped.max(axis=1),
            np.mean(clipped > 0.98, axis=1),
            np.log(clipped).mean(axis=1),
            logit.mean(axis=1),
        ]
    ).astype(np.float32)


def _nuisance_design(q_logp: np.ndarray, entropy: np.ndarray) -> np.ndarray:
    position = np.broadcast_to(
        np.linspace(0.0, 1.0, q_logp.shape[1], dtype=np.float64), q_logp.shape
    )
    q = q_logp.astype(np.float64)
    h = entropy.astype(np.float64)
    return np.stack([q, h, q * q, h * h, q * h, position], axis=-1)


def _ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_predict: np.ndarray,
    l2: float = 1.0,
) -> np.ndarray:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    train = (x_train - mean) / std
    predict = (x_predict - mean) / std
    train = np.column_stack([train, np.ones(len(train))])
    predict = np.column_stack([predict, np.ones(len(predict))])
    penalty = np.eye(train.shape[1], dtype=np.float64) * l2
    penalty[-1, -1] = 0.0
    weights = np.linalg.solve(train.T @ train + penalty, train.T @ y_train)
    return predict @ weights


def cross_fitted_verifier_residual(
    acceptance: np.ndarray,
    q_logp: np.ndarray,
    entropy: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    seed: int,
    folds: int = 4,
) -> np.ndarray:
    """Remove draft difficulty using calibration-only, label-free nuisance fits."""
    design = _nuisance_design(q_logp, entropy)
    clipped = np.clip(acceptance.astype(np.float64), 1e-5, 1.0 - 1e-5)
    response = np.log(clipped) - np.log1p(-clipped)
    residual = np.zeros_like(response)

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(np.sort(calibration))
    fold_records = np.array_split(shuffled, min(folds, len(shuffled)))
    for held_out in fold_records:
        fitting = np.setdiff1d(calibration, held_out, assume_unique=False)
        prediction = _ridge_predict(
            design[fitting].reshape(-1, design.shape[-1]),
            response[fitting].reshape(-1),
            design[held_out].reshape(-1, design.shape[-1]),
        )
        residual[held_out] = response[held_out] - prediction.reshape(
            len(held_out), response.shape[1]
        )

    test_prediction = _ridge_predict(
        design[calibration].reshape(-1, design.shape[-1]),
        response[calibration].reshape(-1),
        design[test].reshape(-1, design.shape[-1]),
    )
    residual[test] = response[test] - test_prediction.reshape(
        len(test), response.shape[1]
    )
    return residual.astype(np.float32)


def _weighted_pool(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    numerator = np.sum(values * weights[:, :, None, None], axis=1)
    denominator = np.maximum(weights.sum(axis=1), 1e-6)[:, None, None]
    return numerator / denominator


def _conditioned_activation_features(
    selected_activations: np.ndarray,
    acceptance: np.ndarray,
    transcript: np.ndarray,
    residual: np.ndarray | None,
) -> np.ndarray:
    mean = selected_activations.mean(axis=1)
    accepted = _weighted_pool(selected_activations, acceptance)
    rejected = _weighted_pool(selected_activations, 1.0 - acceptance)
    centered = acceptance - acceptance.mean(axis=1, keepdims=True)
    covariance = np.mean(
        selected_activations * centered[:, :, None, None], axis=1
    )
    blocks = [mean, accepted, rejected, accepted - rejected, covariance]
    if residual is not None:
        blocks.extend(
            [
                np.mean(selected_activations * residual[:, :, None, None], axis=1),
                np.mean(
                    selected_activations * np.abs(residual)[:, :, None, None], axis=1
                ),
            ]
        )
    flattened = [block.reshape(len(block), -1) for block in blocks]
    return np.column_stack([*flattened, transcript]).astype(np.float32)


def q_stratified_shuffle(
    acceptance: np.ndarray,
    q_logp: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    seed: int,
    bins: int = 8,
) -> np.ndarray:
    """Destroy activation/transcript alignment while preserving q strata."""
    edges = np.unique(
        np.quantile(q_logp[calibration].reshape(-1), np.linspace(0.0, 1.0, bins + 1))
    )
    shuffled = acceptance.copy()
    rng = np.random.default_rng(seed)
    for raw_subset in (calibration, test):
        subset = np.sort(raw_subset)
        subset_q = q_logp[subset].reshape(-1)
        subset_a = acceptance[subset].reshape(-1).copy()
        groups = np.digitize(subset_q, edges[1:-1], right=True)
        for group in np.unique(groups):
            positions = np.flatnonzero(groups == group)
            subset_a[positions] = rng.permutation(subset_a[positions])
        shuffled[subset] = subset_a.reshape(len(subset), acceptance.shape[1])
    return shuffled


def build_activation_feature_families(
    draft: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    selected: np.ndarray,
    acceptance: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    seed: int,
) -> dict[str, np.ndarray]:
    activations = draft["activation_stats"]
    selected_activations = gather_tokens(activations, selected)
    q_logp = gather_tokens(draft["token_logp"], selected)
    entropy = gather_tokens(draft["entropy"], selected)
    valid_count = np.isfinite(draft["token_logp"]).sum(axis=1)
    terminal = activations[np.arange(len(activations)), valid_count - 1]
    selected_mean = selected_activations.mean(axis=1)
    transcript = _transcript_summary(acceptance)

    residual = cross_fitted_verifier_residual(
        acceptance, q_logp, entropy, calibration, test, seed + 1
    )
    shuffled_acceptance = q_stratified_shuffle(
        acceptance, q_logp, calibration, test, seed + 2
    )
    shuffled_transcript = _transcript_summary(shuffled_acceptance)
    shuffled_residual = cross_fitted_verifier_residual(
        shuffled_acceptance, q_logp, entropy, calibration, test, seed + 3
    )

    return {
        "draft_nart_stat_last_triplet": terminal.reshape(len(terminal), -1),
        "draft_selected_activation_triplet": selected_mean.reshape(
            len(selected_mean), -1
        ),
        "transcript_only_triplet": transcript,
        "naive_activation_transcript_concat_triplet": np.column_stack(
            [selected_mean.reshape(len(selected_mean), -1), transcript]
        ),
        "draver_act_direct_triplet": _conditioned_activation_features(
            selected_activations, acceptance, transcript, residual=None
        ),
        "draver_act_residual_triplet": _conditioned_activation_features(
            selected_activations, acceptance, transcript, residual=residual
        ),
        "control_qbin_shuffled_draver_act_triplet": _conditioned_activation_features(
            selected_activations,
            shuffled_acceptance,
            shuffled_transcript,
            residual=shuffled_residual,
        ),
    }


class _MetricEncoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        hidden = min(96, max(32, int(math.sqrt(width * 32))))
        self.network = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Linear(hidden, 16),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.network(values), dim=-1)


def _triplet_score_once(
    features: np.ndarray,
    labels: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    seed: int,
    epochs: int = 160,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    positive = rng.permutation(np.sort(calibration[labels[calibration] == 1]))
    negative = rng.permutation(np.sort(calibration[labels[calibration] == 0]))
    support_per_class = max(2, min(12, len(positive) // 4))
    support = np.concatenate(
        [positive[:support_per_class], negative[:support_per_class]]
    )
    representation = np.concatenate(
        [positive[support_per_class:], negative[support_per_class:]]
    )
    rng.shuffle(representation)

    mean = features[representation].mean(axis=0, keepdims=True)
    std = features[representation].std(axis=0, keepdims=True)
    keep = std[0] > 1e-6
    if not np.any(keep):
        return np.zeros(len(test), dtype=np.float64)
    standardized = np.clip((features[:, keep] - mean[:, keep]) / std[:, keep], -10, 10)
    train_x = torch.from_numpy(standardized[representation].astype(np.float32))
    train_y = torch.from_numpy(labels[representation].astype(np.int64))
    support_x = torch.from_numpy(standardized[support].astype(np.float32))
    test_x = torch.from_numpy(standardized[test].astype(np.float32))

    torch.manual_seed(seed)
    encoder = _MetricEncoder(train_x.shape[1])
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=2e-3, weight_decay=1e-3)
    class_positions = {
        label: np.flatnonzero(train_y.numpy() == label) for label in (0, 1)
    }
    for _ in range(epochs):
        anchors = np.arange(len(train_x))
        positives = np.empty_like(anchors)
        negatives = np.empty_like(anchors)
        y_numpy = train_y.numpy()
        for index, label in enumerate(y_numpy):
            same = class_positions[int(label)]
            same = same[same != index]
            positives[index] = int(rng.choice(same))
            negatives[index] = int(rng.choice(class_positions[1 - int(label)]))
        optimizer.zero_grad(set_to_none=True)
        loss = F.triplet_margin_loss(
            encoder(train_x[anchors]),
            encoder(train_x[positives]),
            encoder(train_x[negatives]),
            margin=1.0,
        )
        loss.backward()
        optimizer.step()

    encoder.eval()
    with torch.no_grad():
        support_embedding = encoder(support_x)
        test_embedding = encoder(test_x)
        distances = torch.cdist(test_embedding, support_embedding).numpy()
    support_labels = labels[support]
    k = min(3, support_per_class)
    positive_distance = np.sort(distances[:, support_labels == 1], axis=1)[:, :k].mean(
        axis=1
    )
    negative_distance = np.sort(distances[:, support_labels == 0], axis=1)[:, :k].mean(
        axis=1
    )
    return negative_distance - positive_distance


def triplet_ensemble_scores(
    features: np.ndarray,
    labels: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    seeds: Iterable[int],
) -> np.ndarray:
    return np.mean(
        triplet_scores_by_seed(features, labels, calibration, test, seeds), axis=0
    )


def triplet_scores_by_seed(
    features: np.ndarray,
    labels: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    seeds: Iterable[int],
) -> np.ndarray:
    return np.asarray(
        [
        _triplet_score_once(features, labels, calibration, test, int(seed))
        for seed in seeds
        ]
    )


def paired_bootstrap_delta(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    repeats: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    deltas: list[float] = []
    for _ in range(repeats):
        sample = np.concatenate(
            [
                rng.choice(positive, len(positive), replace=True),
                rng.choice(negative, len(negative), replace=True),
            ]
        )
        deltas.append(
            auc_rank(labels[sample], left[sample])
            - auc_rank(labels[sample], right[sample])
        )
    return {
        "delta_auc": float(
            auc_rank(labels, left) - auc_rank(labels, right)
        ),
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
    }


def json_safe_scores(result: dict[str, Any]) -> dict[str, Any]:
    """Copy an audit result with per-test score arrays made JSON-serializable.

    ``scores`` is the only field holding numpy arrays; metrics, deltas, and
    protocol metadata already contain plain Python numbers. In-process
    consumers such as runner.py keep receiving arrays because only serialized
    artifacts need the conversion.
    """
    scores = result.get("scores")
    if not isinstance(scores, dict):
        return result
    return {
        **result,
        "scores": {
            name: [float(value) for value in values] for name, values in scores.items()
        },
    }


def evaluate_activation_audit(
    draft: dict[str, np.ndarray],
    target: dict[str, np.ndarray],
    labels: np.ndarray,
    calibration: np.ndarray,
    test: np.ndarray,
    min_k_fraction: float,
    transcript_repeats: int,
    bootstrap_repeats: int,
    detector_seeds: int,
    seed: int,
    few_shot_per_class: tuple[int, ...] = (16, 32, 48),
    acceptance_override: np.ndarray | None = None,
    exact_alpha_override: np.ndarray | None = None,
    metric_family_names: tuple[str, ...] | None = None,
    include_direct_scores: bool = True,
    comparison_baselines: tuple[str, ...] | None = None,
    selected_token_cap: int | None = None,
) -> dict[str, Any]:
    selected = cap_selected_positions(
        bottom_k_indices(draft["token_logp"], min_k_fraction),
        draft["token_logp"],
        selected_token_cap,
    )
    if acceptance_override is None:
        acceptance, exact_alpha = sample_acceptance_rates(
            target["token_logp"],
            draft["token_logp"],
            selected,
            transcript_repeats,
            seed,
        )
    else:
        if exact_alpha_override is None:
            raise ValueError("exact_alpha_override is required with acceptance_override")
        acceptance = np.asarray(acceptance_override, dtype=np.float32)
        exact_alpha = np.asarray(exact_alpha_override, dtype=np.float32)
        if acceptance.shape != selected.shape or exact_alpha.shape != selected.shape:
            raise ValueError("acceptance overrides must match the selected-token shape")
    families = build_activation_feature_families(
        draft, target, selected, acceptance, calibration, test, seed + 10
    )
    if metric_family_names is not None:
        unknown = sorted(set(metric_family_names) - set(families))
        if unknown:
            raise ValueError(f"unknown metric feature families: {unknown}")
        families = {name: families[name] for name in metric_family_names}
    seeds = tuple(seed + 1000 + offset for offset in range(detector_seeds))
    test_labels = labels[test]
    all_scores: dict[str, np.ndarray] = {}
    metrics: dict[str, dict[str, float]] = {}
    detector_stability: dict[str, dict[str, Any]] = {}
    for offset, (name, features) in enumerate(families.items()):
        seed_scores = triplet_scores_by_seed(
            features, labels, calibration, test, seeds
        )
        score = np.mean(seed_scores, axis=0)
        seed_aucs = [float(auc_rank(test_labels, values)) for values in seed_scores]
        all_scores[name] = score
        metrics[name] = metric_row(
            test_labels, score, bootstrap_repeats, seed + 2000 + offset
        )
        detector_stability[name] = {
            "auc_by_seed": seed_aucs,
            "auc_mean": float(np.mean(seed_aucs)),
            "auc_std": float(np.std(seed_aucs, ddof=1))
            if len(seed_aucs) > 1
            else 0.0,
            "auc_min": float(np.min(seed_aucs)),
            "auc_max": float(np.max(seed_aucs)),
        }

    q_selected = gather_tokens(draft["token_logp"], selected)
    direct_scores = (
        {
            "draft_min_k_logp": q_selected.mean(axis=1)[test],
            "verifier_mean_acceptance": acceptance.mean(axis=1)[test],
            "oracle_exact_mean_acceptance": exact_alpha.mean(axis=1)[test],
        }
        if include_direct_scores
        else {}
    )
    for offset, (name, score) in enumerate(direct_scores.items(), start=len(metrics)):
        all_scores[name] = score
        metrics[name] = metric_row(
            test_labels, score, bootstrap_repeats, seed + 2000 + offset
        )

    proposed = all_scores["draver_act_residual_triplet"]
    baselines = comparison_baselines or (
        "draft_nart_stat_last_triplet",
        "transcript_only_triplet",
        "naive_activation_transcript_concat_triplet",
        "control_qbin_shuffled_draver_act_triplet",
    )
    missing_baselines = sorted(set(baselines) - set(all_scores))
    if missing_baselines:
        raise ValueError(f"comparison baselines were not evaluated: {missing_baselines}")
    comparisons = {
        f"draver_act_residual_minus_{baseline}": paired_bootstrap_delta(
            test_labels,
            proposed,
            all_scores[baseline],
            bootstrap_repeats,
            seed + 3000 + offset,
        )
        for offset, baseline in enumerate(baselines)
    }

    few_shot: dict[str, dict[str, dict[str, float]]] = {}
    maximum = min(
        int(np.sum(labels[calibration] == 1)), int(np.sum(labels[calibration] == 0))
    )
    for per_class in few_shot_per_class:
        if per_class > maximum:
            continue
        subset = nested_calibration_subset(calibration, labels, per_class, seed + 4000)
        subset_families = (
            families
            if per_class == maximum
            else build_activation_feature_families(
                draft,
                target,
                selected,
                acceptance,
                subset,
                test,
                seed + 4100 + per_class,
            )
        )
        rows: dict[str, dict[str, float]] = {}
        for offset, name in enumerate(
            [
                "draft_nart_stat_last_triplet",
                "transcript_only_triplet",
                "naive_activation_transcript_concat_triplet",
                "draver_act_residual_triplet",
                "control_qbin_shuffled_draver_act_triplet",
            ]
        ):
            score = triplet_ensemble_scores(
                subset_families[name], labels, subset, test, seeds
            )
            rows[name] = metric_row(
                test_labels,
                score,
                bootstrap_repeats,
                seed + 5000 + per_class * 20 + offset,
            )
        few_shot[str(per_class)] = rows

    return {
        "metrics": metrics,
        "scores": all_scores,
        "detector_stability": detector_stability,
        "paired_auc_deltas": comparisons,
        "few_shot_per_class": few_shot,
        "protocol": {
            "selected_tokens_per_record": int(selected.shape[1]),
            "repeats_per_selected_token": int(transcript_repeats),
            "bits_per_record": int(selected.shape[1] * transcript_repeats),
            "mean_observed_acceptance": float(acceptance.mean()),
            "mean_exact_acceptance_probability": float(exact_alpha.mean()),
        },
        "feature_schema": {
            "activation_statistics": list(ACTIVATION_FEATURE_NAMES),
            "layers": int(draft["activation_stats"].shape[2]),
            "normalization": "representation-training subset only inside each detector",
            "nuisance_fit": "four-fold cross-fit on calibration; labels unused",
            "triplet_margin": 1.0,
            "detector_seed_count": int(detector_seeds),
        },
    }
