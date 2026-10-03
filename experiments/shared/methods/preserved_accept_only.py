"""Fixed evidence fusion that preserves draft and uncorrected acceptance information.

Only auxiliary nonmembers fit the document-score normalization. The independent
calibration set is reserved for thresholds. No labels or target probabilities
enter these functions. These are ranking scores, not likelihoods or e-values.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

from experiments.shared.methods.protocol_accept_only import document_scores

OLD = 'main_fixed_sparse_positive'
PRIMARY = 'preserved_multiscale'
CORRECTION = 0.5
FUSIONS = {
    'conditional_multiscale': {'dense_positive': .5, OLD: .5},
    'accept_q_fusion': {'accept_rate': .5, 'q_only': .5},
    'accept_multiscale': {'accept_rate': .5, 'dense_positive': .25, OLD: .25},
    PRIMARY: {'partial_residual_050': .5, 'q_only': .25, 'dense_positive': .125, OLD: .125},
    'preserved_no_q': {'partial_residual_050': .5, 'dense_positive': .25, OLD: .25},
    'preserved_no_partial': {'accept_rate': .5, 'q_only': .25, 'dense_positive': .125, OLD: .125},
}
BASE_METHODS = (OLD, 'accept_rate', 'q_only', 'partial_residual_050', 'full_residual', 'dense_positive')
METHODS = (*BASE_METHODS, *FUSIONS)


def base_scores(logpmf, counts, lengths, logq):
    """Return document scores for one fixed trajectory per document, at budget B."""
    logpmf, logq = np.asarray(logpmf, float), np.asarray(logq, float)
    counts, lengths = np.asarray(counts), np.asarray(lengths)
    if (lengths.ndim != 1 or not len(lengths) or (lengths <= 0).any()
            or not np.issubdtype(lengths.dtype, np.integer)):
        raise ValueError('positive integer document lengths required')
    if (logpmf.ndim != 2 or logpmf.shape[1] < 2 or len(logpmf) != lengths.sum()
            or counts.shape != (len(logpmf),) or logq.shape != counts.shape
            or not np.issubdtype(counts.dtype, np.integer)
            or (counts < 0).any() or (counts >= logpmf.shape[1]).any()
            or not np.isfinite(logpmf).all() or not np.isfinite(logq).all()
            or (logq > 1e-5).any()
            or not np.allclose(logsumexp(logpmf, axis=1), 0, atol=2e-6)):
        raise ValueError('invalid observable counts, draft log probabilities or count PMFs')
    data = dict(record_ids=np.arange(len(lengths)), document_indices=np.arange(len(lengths)),
                lengths=lengths, counts=counts, features=logq[:, None])
    shared = document_scores(data, logpmf)
    offsets = np.r_[0, lengths.cumsum()]
    expected = np.add.reduceat(np.exp(logpmf) @ np.arange(logpmf.shape[1]), offsets[:-1])
    expected /= (logpmf.shape[1] - 1) * lengths
    return {OLD: shared['sparse_positive'], 'accept_rate': shared['accept_rate'],
            'q_only': shared['q_only'],
            'partial_residual_050': shared['accept_rate'] - CORRECTION * expected,
            'full_residual': shared['accept_rate'] - expected,
            'dense_positive': shared['global_positive']}


def fit_normalization(reference_scores):
    """Fit location/scale from explicitly supplied auxiliary scores only.

    Constant reference channels contribute zero, including on future records;
    this avoids amplifying an unidentifiable scale through an epsilon divisor.
    """
    result = {}
    for name in BASE_METHODS:
        values = np.asarray(reference_scores[name], float)
        if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
            raise ValueError('at least two finite auxiliary document scores required')
        center, scale = float(values.mean()), float(values.std(ddof=1))
        result[name] = dict(mean=center, scale=scale, active=scale > 1e-12, n=len(values))
    return result


def fuse_scores(scores, normalization):
    """Apply the predeclared weights; never refit using the scoring batch."""
    standardized = {}
    shape = None
    for name in BASE_METHODS:
        values = np.asarray(scores[name], float)
        if values.ndim != 1 or not np.isfinite(values).all() or (shape is not None and values.shape != shape):
            raise ValueError('finite aligned document scores required')
        shape = values.shape
        spec = normalization[name]
        if (not np.isfinite(spec['mean']) or not np.isfinite(spec['scale'])
                or spec['scale'] < 0 or bool(spec['active']) != (spec['scale'] > 1e-12)):
            raise ValueError('invalid frozen normalization')
        standardized[name] = ((values - spec['mean']) / spec['scale']
                              if spec['active'] else np.zeros_like(values))
    return {**{k: np.asarray(scores[k], float).copy() for k in BASE_METHODS},
            **{name: sum(weight * standardized[channel] for channel, weight in weights.items())
               for name, weights in FUSIONS.items()}}
