"""Separate causal integrity from sequence-shape drift at inference precision."""
from __future__ import annotations

import torch
from experiments.shared.models.precision import fp32_reference
from experiments.shared.protocols.sd_protocol import fixed_trace

VALIDATION_SCHEMA = 'adapter_causality_precision_v2'


def _distribution(row, name, index):
    support = torch.isfinite(row)
    if (row.ndim != 1 or not support.any() or torch.isnan(row).any()
            or torch.isposinf(row).any()
            or abs(float(torch.logsumexp(row, -1))) > 1e-4):
        raise ValueError(f'{name} invalid normalized distribution at position {index}')
    return support


def _difference(full, other, name, index):
    support = _distribution(full, name, index)
    _distribution(other, name, index)
    if full.shape != other.shape or not torch.equal(support, torch.isfinite(other)):
        raise ValueError(f'{name} support changes with context at position {index}')
    return dict(max_abs_log_error=float((full[support] - other[support]).abs().max()),
                total_variation=float((full.exp() - other.exp()).abs().sum() / 2))


def _checks(adapter, tokens, positions, *, reference=False):
    full = adapter.rows(tokens)
    vocab = full[0].shape[-1]
    results, needs_reference = [], False
    for index in positions:
        prefix = adapter.next(tokens[:index + 1])
        # Two nontrivial interventions preserve shape and every observed token.
        # They separate causal dependence from length-dependent kernel rounding.
        futures = [adapter.rows(tokens[:index + 1] + [(t + offset) % vocab for t in tokens[index + 1:]])
                   for offset in (1, max(1, vocab // 2))]
        for channel, name in enumerate(('target', 'draft')):
            row, short = full[channel][index], prefix[channel]
            drift = _difference(row, short, name, index)
            causal_errors = []
            for future in futures:
                changed = future[channel][index]
                difference = _difference(row, changed, name, index)
                support = torch.isfinite(row)
                if (not torch.allclose(row[support], changed[support], atol=1e-5, rtol=1e-5)
                        or difference['total_variation'] > 1e-5):
                    raise ValueError(f'{name} future-token dependence at position {index} '
                                     f'(same-length intervention): {difference}')
                causal_errors.append(difference['max_abs_log_error'])
            support = torch.isfinite(row)
            consistent = (torch.allclose(row[support], short[support],
                                          atol=1e-3 if reference else .15,
                                          rtol=1e-4 if reference else .01)
                          and drift['total_variation'] <= (1e-4 if reference else .01))
            if reference and not consistent:
                raise ValueError(f'{name} FP32 prefix inconsistency at position {index}: {drift}')
            needs_reference |= not consistent
            results.append(dict(model=name, position=index, prefix=drift,
                                prefix_within_tolerance=bool(consistent),
                                causality_max_abs_log_error=max(causal_errors)))
    return results, needs_reference


@torch.inference_mode()
def validate_adapter(adapter, prompt, response, *, seed=20260914):
    tokens = (list(prompt) + list(response))[:64]
    if len(tokens) < 5:
        raise ValueError('adapter validation needs at least five context tokens')
    positions = sorted({len(tokens) // 2, len(tokens) - 2})
    checks, needs_reference = _checks(adapter, tokens, positions)
    reference = dict(status='not_needed')
    if needs_reference:
        try:
            with fp32_reference(adapter):
                reference_checks, _ = _checks(adapter, tokens, positions, reference=True)
            reference = dict(status='passed', checks=reference_checks)
        except (RuntimeError, TypeError, NotImplementedError) as error:
            # No silent tolerance inflation or continuation after a failed kernel.
            raise ValueError(f'FP32 recheck failed; condition remains blocked: {error}') from error
    # Runs only after original parameter dtypes/buffers have been restored.
    trace = fixed_trace(adapter, tokens[:2], tokens[2:], seed=seed)
    if len(trace['counts']) == 0 or trace['counts'].max() > 2:
        raise ValueError('invalid protocol feedback')
    return dict(status='passed', schema=VALIDATION_SCHEMA,
                scope='sampled_causality_and_prefix_numerics', checked_positions=positions,
                tokens=len(tokens), checks=checks, fp32_recheck=reference,
                prefix_tolerances=dict(inference_atol=.15, inference_rtol=.01, inference_tv=.01,
                                       reference_atol=1e-3, reference_rtol=1e-4, reference_tv=1e-4),
                candidate_positions=trace['candidate_positions'],
                supported_candidates=trace['supported_candidates'],
                effectiveness_or_speedup_claim=False)
