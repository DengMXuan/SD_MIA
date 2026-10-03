"""Bounded positive evidence additions to a smoothed acceptance-tail score.

All transformations fit auxiliary nonmembers only. Bounds protect the anchor
from arbitrarily large penalties/additions, not from all ranking or FPR loss.
The Jeffreys-smoothed rejection transform is a score, not a posterior claim:
token positions need not share a common acceptance probability.
"""
from __future__ import annotations

import numpy as np

from experiments.shared.methods.preserved_accept_only import OLD, PRIMARY as PREVIOUS

PRIMARY = 'tail_guarded_positive'
CONTROL_METHODS = (OLD, PREVIOUS, 'accept_rate', 'q_only', 'accept_q_fusion', 'preserved_no_partial')
NEW_METHODS = ('smoothed_reject_tail', 'tail_q_bounded', PRIMARY, 'tail_guarded_no_q',
               'tail_guarded_no_transform', 'tail_signed_fusion', 'tail_guarded_unbounded')
METHODS = (*CONTROL_METHODS, *NEW_METHODS)
CHANNELS = ('accept_rate', 'smoothed_reject_tail', 'q_only', 'dense_per_token', 'sparse_per_token')
SPEC = dict(primary=PRIMARY, pseudocount=.5, q_weight=.5, q_softplus_cap=2.,
            conditional_weight=.25, conditional_positive_cap=1.,
            conditional_dense_weight=.5, conditional_sparse_weight=.5,
            normalization='reference-only document mean and sample SD; inactive constant channels')


def components(base, counts, lengths, *, budget=2):
    """Build document features without labels or target-side information."""
    counts, lengths = np.asarray(counts), np.asarray(lengths)
    if (type(budget) is not int or budget < 1 or lengths.ndim != 1 or not len(lengths)
            or not np.issubdtype(lengths.dtype, np.integer) or (lengths <= 0).any()
            or counts.shape != (lengths.sum(),) or not np.issubdtype(counts.dtype, np.integer)
            or (counts < 0).any() or (counts > budget).any()):
        raise ValueError('invalid document lengths, query budget or acceptance counts')
    for name in ('accept_rate', 'q_only', 'dense_positive', OLD):
        values = np.asarray(base[name], float)
        if values.shape != lengths.shape or not np.isfinite(values).all():
            raise ValueError('finite aligned base scores required')
    offsets = np.r_[0, lengths.cumsum()]
    accepted = np.add.reduceat(counts.astype(float), offsets[:-1])
    trials = budget * lengths
    if not np.allclose(accepted / trials, base['accept_rate'], atol=1e-12, rtol=1e-12):
        raise ValueError('base acceptance score disagrees with observations')
    return dict(accept_rate=np.asarray(base['accept_rate'], float),
                smoothed_reject_tail=-np.log((trials - accepted + SPEC['pseudocount'])
                                            / (trials + 2 * SPEC['pseudocount'])),
                q_only=np.asarray(base['q_only'], float),
                dense_per_token=np.asarray(base['dense_positive'], float) / lengths,
                sparse_per_token=np.asarray(base[OLD], float) / lengths)


def fit_reference(reference):
    """Receive only reference documents, never calibration/test records."""
    result = {}
    size = None
    for name in CHANNELS:
        values = np.asarray(reference[name], float)
        if (values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all()
                or (size is not None and size != len(values))):
            raise ValueError('aligned finite auxiliary scores required')
        size = len(values)
        scale = float(values.std(ddof=1))
        result[name] = dict(mean=float(values.mean()), scale=scale, active=scale > 1e-12, n=size)
    return result


def score(features, reference):
    """Apply a frozen score map; other test documents cannot affect a score."""
    z, size = {}, None
    for name in CHANNELS:
        values, spec = np.asarray(features[name], float), reference[name]
        if (values.ndim != 1 or not np.isfinite(values).all()
                or (size is not None and size != len(values))
                or not np.isfinite(spec['mean']) or not np.isfinite(spec['scale'])
                or spec['scale'] < 0 or bool(spec['active']) != (spec['scale'] > 1e-12)):
            raise ValueError('invalid features or frozen reference normalization')
        size = len(values)
        z[name] = (values - spec['mean']) / spec['scale'] if spec['active'] else np.zeros_like(values)
    anchor = z['smoothed_reject_tail']
    q_positive = np.logaddexp(0, z['q_only']) if reference['q_only']['active'] else np.zeros(size)
    conditional = (SPEC['conditional_dense_weight'] * z['dense_per_token']
                   + SPEC['conditional_sparse_weight'] * z['sparse_per_token'])
    q_bonus = SPEC['q_weight'] * np.minimum(q_positive, SPEC['q_softplus_cap'])
    conditional_bonus = SPEC['conditional_weight'] * np.clip(
        conditional, 0, SPEC['conditional_positive_cap'])
    return {
        'smoothed_reject_tail': np.asarray(features['smoothed_reject_tail'], float).copy(),
        'tail_q_bounded': anchor + q_bonus,
        PRIMARY: anchor + q_bonus + conditional_bonus,
        'tail_guarded_no_q': anchor + conditional_bonus,
        'tail_guarded_no_transform': z['accept_rate'] + q_bonus + conditional_bonus,
        'tail_signed_fusion': anchor + SPEC['q_weight'] * z['q_only'] + SPEC['conditional_weight'] * conditional,
        'tail_guarded_unbounded': anchor + SPEC['q_weight'] * q_positive
                                 + SPEC['conditional_weight'] * np.maximum(conditional, 0),
    }
