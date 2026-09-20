"""Single-proposal SD and fixed-candidate probes over frozen model adapters.

Target distributions and generated tokens stay inside this runtime. Returned
traces contain only draft features, reached feedback, and accounting metadata.
Context reconstruction deliberately avoids unsupported hybrid-cache rollback.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
import torch

FEATURE_NAMES = (
    "logq", "q_entropy_norm", "q_rank_norm", "q_top1_margin",
    "position_fraction", "start_fraction",
)


def resolve_starts(length: int, starts: list[str]) -> list[int]:
    """Fractions refer to response tokens, excluding prompt and appended EOS."""
    if not starts:
        raise ValueError("at least one start is required")
    positions = []
    for start in starts:
        if start == "suffix64":
            position = length - 64
        else:
            fraction = float(start)
            if not math.isfinite(fraction) or not 0 < fraction < 1:
                raise ValueError(f"invalid start fraction: {start}")
            position = math.floor(fraction * length)
        if not 0 < position < length:
            raise ValueError(f"start {start} leaves an empty prefix or suffix at length {length}")
        if position in positions:
            raise ValueError(f"duplicate resolved start at response token {position}")
        positions.append(position)
    return positions


def trajectory_seed(seed: int, record_id: str, start: str) -> int:
    payload = f"{seed}\0{record_id}\0{start}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def normalized(logits: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError("invalid proposal/target logits")
    if not torch.isfinite(logits).any(dim=-1).all():
        raise ValueError("empty distribution support")
    return torch.log_softmax(logits, dim=-1)


def map_eagle_logits(logits: torch.Tensor, offsets: torch.Tensor, vocab: int) -> torch.Tensor:
    indices = torch.arange(logits.shape[-1], device=logits.device) + offsets.long()
    if (indices.numel() != logits.shape[-1] or indices.unique().numel() != indices.numel()
            or indices.min() < 0 or indices.max() >= vocab):
        raise ValueError("invalid EAGLE draft-to-target vocabulary mapping")
    full = logits.new_full((*logits.shape[:-1], vocab), -torch.inf)
    full[..., indices] = logits
    return full


def draft_features(logq: torch.Tensor, token: int, position: float, start: float) -> list[float]:
    if not torch.isfinite(logq[token]):
        raise ValueError("candidate outside draft support")
    finite = torch.isfinite(logq)
    values = logq[finite]
    norm = math.log(logq.numel())
    if norm <= 0 or len(values) < 2:
        raise ValueError("difficulty features require at least two supported tokens")
    entropy = -(values.exp() * values).sum() / norm
    rank = torch.log1p((values > logq[token]).sum().float()) / norm
    top = values.topk(2).values
    return [float(logq[token]), float(entropy), float(rank), float(top[0] - top[1]), position, start]


@dataclass
class RuntimeCost:
    target_forward_calls: int = 0
    target_input_tokens: int = 0
    draft_forward_calls: int = 0
    draft_input_tokens: int = 0
    hidden_state_bytes: int = 0


class FrozenAdapter:
    """Full-context reference implementation; no target values enter archives.

    EAGLE uses the checkpoint's same-position fused-hidden forward contract.
    Native MTP step zero pairs h[t] with embedding(x[t+1]) to predict x[t+2].
    """

    def __init__(self, target: Any, draft: Any, kind: str, device: str | torch.device):
        if kind not in ("plain", "eagle3", "mtp"):
            raise ValueError(f"unsupported adapter {kind}")
        self.target, self.draft, self.kind = target, draft, kind
        self.device = torch.device(device)
        self.cost = RuntimeCost()
        for model in (target, draft):
            model.eval()
            model.requires_grad_(False)
        if kind == "mtp" and draft.config.num_speculative_steps != 1:
            raise ValueError("first implementation requires a depth-1 native MTP export")

    def _target(self, ids, *, hidden=False):
        self.cost.target_forward_calls += 1
        self.cost.target_input_tokens += ids.shape[1]
        return self.target(input_ids=ids, use_cache=False, output_hidden_states=hidden)

    def _draft_logits(self, ids, output):
        self.cost.draft_forward_calls += 1
        self.cost.draft_input_tokens += ids.shape[1]
        if self.kind == "plain":
            return self.draft(input_ids=ids, use_cache=False).logits
        if self.kind == "eagle3":
            from .drafts.heads import eagle3_connector

            hidden = eagle3_connector(output, self.target)
            self.cost.hidden_state_bytes += hidden.numel() * hidden.element_size()
            # Capture the draft-vocabulary logits, avoiding finite sentinels in
            # remote-code vocabulary expansion. d2t contains offsets, not IDs.
            base = self.draft.get_base_model() if hasattr(self.draft, "get_base_model") else self.draft
            captured = []
            handle = base.norm.register_forward_hook(lambda _m, _a, value: captured.append(value))
            try:
                self.draft(input_ids=ids, hidden_states=hidden, use_cache=False, return_dict=True)
                logits = base.lm_head(captured[-1])
            finally:
                handle.remove()
            return map_eagle_logits(logits, base.d2t, output.logits.shape[-1])
        # The training-oriented public MTP API excludes its final two targets.
        # Append an ignored target placeholder to expose the next-token row.
        # With depth=1, only ids[:, 1:-1] supply embeddings: the placeholder is
        # NEVER an input embedding or target hidden state used by that row.
        if ids.shape[1] < 2:
            raise ValueError("MTP requires at least two context tokens")
        hidden = output.hidden_states[-1]
        self.cost.hidden_state_bytes += hidden.numel() * hidden.element_size()
        padded = torch.cat((ids, ids.new_zeros((1, 1))), dim=1)
        logits, _, _ = self.draft(
            input_ids=padded, hidden_states=hidden, attention_mask=None,
            loss_mask=torch.zeros_like(padded), return_dict=True,
        )
        result = logits[0]
        if result.shape[1] != ids.shape[1] - 1:
            raise ValueError("native MTP step-zero alignment changed")
        # Align to ordinary next-token rows: row 0 is unobservable for MTP.
        return torch.cat((result.new_full((1, 1, result.shape[-1]), -torch.inf), result), dim=1)

    @torch.inference_mode()
    def rows(self, tokens: list[int]):
        ids = torch.tensor([tokens], device=self.device)
        output = self._target(ids, hidden=self.kind != "plain")
        q = self._draft_logits(ids, output)[0]
        p = output.logits[0]
        if q.shape != p.shape:
            raise ValueError("target and draft output supports/positions differ")
        # MTP's first row has no prediction; preserve that explicit mask.
        supported = torch.isfinite(q).any(-1)
        logq = q.float().clone()
        logq[supported] = normalized(q[supported])
        return normalized(p), logq

    def next(self, tokens: list[int]):
        p, q = self.rows(tokens)
        return p[-1], q[-1]

    @torch.inference_mode()
    def target_next(self, tokens: list[int]):
        ids = torch.tensor([tokens], device=self.device)
        return normalized(self._target(ids).logits[0, -1])


@torch.inference_mode()
def natural_trace(adapter, prefix: list[int], *, rounds: int, seed: int,
                  start_fraction: float, eos_ids: tuple[int, ...] = ()) -> dict:
    if not prefix or rounds < 1:
        raise ValueError("nonempty prefix and positive round budget required")
    generator = torch.Generator(device=adapter.device).manual_seed(seed)
    context = list(prefix)
    features, bits = [], []
    reason = "round_cap"
    for index in range(rounds):
        logp, logq = adapter.next(context)
        token = int(torch.multinomial(logq.exp(), 1, generator=generator))
        alpha = torch.exp(torch.minimum(logp[token] - logq[token], logp.new_zeros(())))
        accepted = bool(torch.rand((), device=adapter.device, generator=generator) < alpha)
        features.append(draft_features(logq, token, index / rounds, start_fraction))
        bits.append(int(accepted))
        if accepted:
            context.append(token)
            if token in eos_ids:
                reason = "accepted_eos"
                break
            distribution = adapter.target_next(context).exp()
        else:
            distribution = (logp.exp() - logq.exp()).clamp_min(0)
            mass = distribution.sum()
            if not mass > 0:
                raise ValueError("rejection without residual probability mass")
            distribution = distribution / mass
        correction_or_bonus = int(torch.multinomial(distribution, 1, generator=generator))
        context.append(correction_or_bonus)
        if correction_or_bonus in eos_ids:
            reason = "bonus_eos" if accepted else "correction_eos"
            break
    return {
        "features": np.asarray(features, dtype=np.float32),
        "counts": np.asarray(bits, dtype=np.uint8),
        "generated_tokens": len(context) - len(prefix),
        "prefix_tokens": len(prefix), "rounds": len(bits),
        "termination": reason, "supported_candidates": len(bits),
        "candidate_positions": len(bits),
    }


@torch.inference_mode()
def fixed_trace(adapter, prompt: list[int], response: list[int], *, seed: int) -> dict:
    """B=2 diagnostic probes. q=0 positions are excluded and counted, never clipped."""
    if not prompt or not response:
        raise ValueError("nonempty prompt and response required")
    generator = torch.Generator(device=adapter.device).manual_seed(seed)
    logp, logq = adapter.rows(prompt + response)
    features, counts = [], []
    for index, token in enumerate(response):
        row = len(prompt) + index - 1
        if not torch.isfinite(logq[row, token]):
            continue
        alpha = torch.exp(torch.minimum(logp[row, token] - logq[row, token], logp.new_zeros(())))
        features.append(draft_features(logq[row], token, index / max(1, len(response) - 1), 0.))
        counts.append(int((torch.rand(2, device=adapter.device, generator=generator) < alpha).sum()))
    if not counts:
        raise ValueError("document has no supported fixed candidates")
    return {
        "features": np.asarray(features, dtype=np.float32),
        "counts": np.asarray(counts, dtype=np.uint8),
        "generated_tokens": 0, "prefix_tokens": len(prompt), "rounds": 0,
        "termination": "fixed_candidates_complete", "supported_candidates": len(counts),
        "candidate_positions": len(response),
    }
