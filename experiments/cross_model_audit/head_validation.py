"""Small actual-checkpoint protocol gate before collecting thousands of records."""
from __future__ import annotations

import torch
from experiments.sd_membership_sft.protocols.sd_protocol import fixed_trace


@torch.inference_mode()
def validate_adapter(adapter, prompt, response):
    # Fit auxiliary only; no test labels or score-based model selection.
    tokens = (list(prompt) + list(response))[:64]
    if len(tokens) < 5:
        raise ValueError("head validation needs at least five context tokens")
    p, q = adapter.rows(tokens)
    checks = []
    for index in sorted({len(tokens) // 2, len(tokens) - 2}):
        prefix_p, prefix_q = adapter.next(tokens[:index + 1])
        for name, full, prefix in (("target", p[index], prefix_p), ("draft", q[index], prefix_q)):
            support = torch.isfinite(full)
            if not support.any() or not torch.equal(support, torch.isfinite(prefix)):
                raise ValueError(f"{name} support changes with future context")
            if not torch.allclose(full[support], prefix[support], atol=.15, rtol=.01):
                raise ValueError(f"{name} future-token leakage or position misalignment")
            if abs(float(torch.logsumexp(prefix, -1))) > .01:
                raise ValueError(f"{name} distribution is not normalized")
        checks.append(index)
    trace = fixed_trace(adapter, tokens[:2], tokens[2:], seed=20260914)
    if len(trace["counts"]) == 0 or trace["counts"].max() > 2:
        raise ValueError("invalid fixed-candidate feedback")
    return {"status": "passed", "scope": "checkpoint_runtime_and_prefix_consistency",
            "checked_positions": checks, "tokens": len(tokens),
            "candidate_positions": trace["candidate_positions"],
            "supported_candidates": trace["supported_candidates"],
            "effectiveness_or_speedup_claim": False}
