"""Draft-side position features; target probabilities and membership labels are absent.

All document maps are fixed at scoring time. Reference fitting receives only
explicitly supplied auxiliary nonmembers. Position scores are hypotheses about
discriminative evidence, not probabilities that a token was memorized.
"""
from __future__ import annotations

import numpy as np


def validate(features, counts, lengths):
    x, c, lengths = np.asarray(features, float), np.asarray(counts), np.asarray(lengths)
    if (lengths.ndim != 1 or not len(lengths) or (lengths <= 0).any()
            or not np.issubdtype(lengths.dtype, np.integer) or x.shape != (lengths.sum(), 6)
            or c.shape != (len(x),) or not np.issubdtype(c.dtype, np.integer)
            or (c < 0).any() or (c > 2).any() or not np.isfinite(x).all()
            or (x[:, 0] > 1e-5).any()):
        raise ValueError('aligned finite q features and B=2 counts required')
    return x, c.astype(float), lengths


def fit_document_reference(values):
    result = {}
    for name, vector in values.items():
        vector = np.asarray(vector, float)
        if vector.ndim != 1 or len(vector) < 2 or not np.isfinite(vector).all():
            raise ValueError('finite reference documents required')
        result[name] = {'mean': float(vector.mean()), 'scale': float(vector.std(ddof=1)), 'n': len(vector)}
    return result


def standardize(values, reference):
    result = {}
    for name, spec in reference.items():
        vector = np.asarray(values[name], float)
        if (not np.isfinite(vector).all() or not np.isfinite(spec['mean'])
                or not np.isfinite(spec['scale']) or spec['scale'] < 0):
            raise ValueError('invalid fixed reference map')
        result[name] = (vector - spec['mean']) / spec['scale'] if spec['scale'] > 1e-12 else np.zeros_like(vector)
    return result


def position_features(features, counts, lengths, *, vocab_size):
    """Round 1: non-learned draft difficulty, rank, entropy and position controls."""
    x, c, lengths = validate(features, counts, lengths)
    if vocab_size < 2:
        raise ValueError('invalid tokenizer vocabulary')
    offsets = np.r_[0, lengths.cumsum()]
    collected = {}
    for left, right in zip(offsets[:-1], offsets[1:]):
        v, a = x[left:right], c[left:right] / 2
        q = v[:, 0]
        entropy = v[:, 1] * np.log(vocab_size)
        centered = q + entropy
        normalized = centered / np.sqrt(np.maximum(entropy, .25))
        result = {'q_mean': q.mean(), 'accept_mean': a.mean(),
                  'q_entropy_centered': centered.mean(), 'q_entropy_scaled': normalized.mean(),
                  'q_rank': -v[:, 2].mean()}
        # q-only selection is fixed before viewing the acceptance feedback.
        orders = {'lowq': np.argsort(q, kind='stable'),
                  'qcenter': np.argsort(normalized, kind='stable')}
        for fraction in (.2, .4, .6):
            suffix = f'{int(100*fraction):02d}'
            n = max(1, int(np.ceil(len(q) * fraction)))
            for key, order in orders.items():
                chosen = order[:n]
                result[f'accept_{key}_{suffix}'] = a[chosen].mean()
                result[f'q_{key}_{suffix}'] = (q if key == 'lowq' else normalized)[chosen].mean()
        for name, value in (('highentropy', entropy), ('highrank', v[:, 2]),
                            ('lowmargin', -v[:, 3])):
            chosen = np.argsort(-value, kind='stable')[:max(1, int(np.ceil(len(q)/2)))]
            result[f'accept_{name}_50'] = a[chosen].mean()
        for start in (.25, .5):
            chosen = v[:, 4] >= start
            if not chosen.any():
                chosen = np.ones(len(v), bool)
            result[f'accept_late_{int(start*100)}'] = a[chosen].mean()
            result[f'q_late_{int(start*100)}'] = q[chosen].mean()
        for power in (.5, 1.):
            weights = np.maximum(-q, .1) ** power
            result[f'accept_qhard_{int(power*10)}'] = np.average(a, weights=weights)
        # A censored q-plus-feedback proxy, never interpreted as recovered p.
        proxy = q + np.log((2*a + .5) / 3)
        result['censored_q_feedback'] = proxy.mean()
        proxy_center = (proxy + entropy) / np.sqrt(np.maximum(entropy, .25))
        result['censored_q_feedback_centered'] = proxy_center.mean()
        chosen = np.argsort(proxy_center, kind='stable')[:max(1, int(np.ceil(len(q)*.2)))]
        result['censored_q_feedback_min20'] = proxy_center[chosen].mean()
        for name, value in result.items():
            collected.setdefault(name, []).append(float(value))
    return {name: np.asarray(value) for name, value in collected.items()}


def round1_fusions(z):
    """A small fixed grid, evaluated only on the authorized development split."""
    result = {}
    for q in ('q_mean', 'q_entropy_scaled', 'q_lowq_20', 'q_qcenter_20'):
        for weight in (.25, .5, .75):
            result[f'fusion_a_{q}_w{int(100*weight)}'] = (1-weight)*z['accept_mean'] + weight*z[q]
    for accept in ('accept_lowq_20', 'accept_lowq_40', 'accept_lowq_60', 'accept_qcenter_20',
                   'accept_highentropy_50', 'accept_qhard_5', 'accept_late_25'):
        result[f'fusion_all_{accept}'] = .5*z['accept_mean'] + .5*z[accept]
        result[f'fusion_q_{accept}'] = .5*z['q_mean'] + .25*z['accept_mean'] + .25*z[accept]
    return result


def q_context(features, lengths):
    """Context summaries use draft features only, never acceptance outcomes."""
    x = np.asarray(features, float)
    lengths = np.asarray(lengths)
    if x.shape != (lengths.sum(), 6) or not np.isfinite(x).all():
        raise ValueError('invalid q-only context input')
    output = []
    for left, right in zip(np.r_[0, lengths.cumsum()][:-1], lengths.cumsum()):
        v = x[left:right, :5]
        extras = []
        for width in (7, 31):
            index = np.arange(len(v))
            lo, hi = np.maximum(0, index-width//2), np.minimum(len(v), index+width//2+1)
            for column in (0, 1):
                summed = np.r_[0., np.cumsum(v[:, column])]
                mean = (summed[hi] - summed[lo])/(hi-lo)
                extras.append(mean)
                if column == 0:
                    square = np.r_[0., np.cumsum(v[:, column]**2)]
                    extras.append(np.sqrt(np.maximum((square[hi]-square[lo])/(hi-lo)-mean**2, 0)))
        extras += [v[:, 0]-v[:, 0].mean(), np.r_[0., np.diff(v[:, 0])]]
        output.append(np.column_stack([v, *extras]))
    return np.concatenate(output)


def fit_nonmember_counts(train_x, train_counts, train_lengths,
                         validation_x, validation_counts, validation_lengths, *, seed=1919):
    """Choose local/context prediction by held-out NONMEMBER document log loss.

    API intentionally accepts only already separated training/validation data;
    neither calibration data nor membership labels are accepted.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    train_counts, validation_counts = np.asarray(train_counts), np.asarray(validation_counts)
    for x, counts, lengths in ((train_x, train_counts, train_lengths),
                               (validation_x, validation_counts, validation_lengths)):
        if (len(x) != np.sum(lengths) or counts.shape != (len(x),)
                or not np.isin(counts, [0, 1, 2]).all() or not np.isfinite(x).all()):
            raise ValueError('invalid separated nonmember observations')
    if set(np.unique(train_counts)) != {0, 1, 2}:
        raise ValueError('all count outcomes needed for this conditional model')
    weights = np.repeat(1/np.asarray(train_lengths), train_lengths)
    weights *= len(weights)/weights.sum()
    fits, losses = {}, {}
    for kind, columns in (('local', 5), ('context', train_x.shape[1])):
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, learning_rate=.08,
                    min_samples_leaf=128, l2_regularization=20., max_bins=63,
                    early_stopping=False, random_state=seed)
        model.fit(train_x[:, :columns], train_counts, sample_weight=weights)
        probability = model.predict_proba(validation_x[:, :columns])
        token_loss = -np.log(np.maximum(probability[np.arange(len(validation_counts)), validation_counts], 1e-12))
        offsets = np.r_[0, np.cumsum(validation_lengths)]
        loss = float(np.mean([token_loss[a:b].mean() for a,b in zip(offsets[:-1], offsets[1:])]))
        losses[kind], fits[kind] = loss, {'model': model, 'columns': columns}
    chosen = min(losses, key=losses.get)
    return {**fits[chosen], 'kind': chosen}, {'nonmember_validation_nll': losses, 'selected': chosen,
        'train_documents': len(train_lengths), 'validation_documents': len(validation_lengths),
        'train_member_count': 0, 'validation_member_count': 0}


def count_position_features(features, counts, lengths, probability):
    """q-predicted weights select difficulty before observing the candidate count."""
    x, c, lengths = validate(features, counts, lengths)
    pi = np.asarray(probability, float)
    if (pi.shape != (len(x), 3) or not np.isfinite(pi).all() or (pi <= 0).any()
            or not np.allclose(pi.sum(1), 1)):
        raise ValueError('strictly positive count probability map required')
    mu = pi @ np.array([0., .5, 1.])
    variance = np.maximum(pi @ np.array([0., .25, 1.]) - mu**2, .005)
    result = {}
    offsets = np.r_[0, lengths.cumsum()]
    for left, right in zip(offsets[:-1], offsets[1:]):
        a, expected, var, prob = c[left:right]/2, mu[left:right], variance[left:right], pi[left:right]
        weights = {'difficulty': np.maximum(1-expected, .05),
                   'surprise': np.minimum(-np.log(prob[:, 2]), 4),
                   'sensitivity': expected*(1-expected)/(var+.02)}
        values = {'nm_expected_accept': expected.mean(),
                  'nm_residual': (a-expected).mean(),
                  'nm_student_residual': np.clip((a-expected)/np.sqrt(var), -5, 5).mean()}
        for name,w in weights.items():
            for correction in (0., .25, .5, 1.):
                values[f'nm_{name}_r{int(100*correction)}'] = np.average(a-correction*expected, weights=w)
        for fraction in (.2, .4, .6):
            chosen = np.argsort(expected, kind='stable')[:max(1,int(np.ceil(len(a)*fraction)))]
            values[f'nm_low_accept_{int(fraction*100)}'] = a[chosen].mean()
        for name, value in values.items():
            result.setdefault(name, []).append(float(value))
    return {name:np.asarray(values) for name,values in result.items()}


def round2_fusions(z, learned_z):
    result = {}
    for name in ('nm_difficulty_r0', 'nm_surprise_r0', 'nm_sensitivity_r0', 'nm_low_accept_40'):
        result[f'fusion_a_{name}'] = .5*z['accept_mean']+.5*learned_z[name]
        for q in ('q_mean', 'q_qcenter_20'):
            result[f'fusion_q_{name}_{q}'] = .5*learned_z[name]+.25*z['accept_mean']+.25*z[q]
    for name in ('nm_residual', 'nm_student_residual', 'nm_difficulty_r100'):
        result[f'fusion_residual_{name}'] = .5*z['accept_mean']+.25*z['q_qcenter_20']+.25*learned_z[name]
    return result


def round3_fusions(z, learned_z):
    """Fixed 57-rule grid: position aggregation plus reference-relative q gates.

    Negative-tail q is a candidate hypothesis prompted by DEVELOPMENT results:
    strong q information in MIMIR7's low-q negatives, weak q differences in Wiki.
    Confirmation records cannot choose the gate, weights or anchor.
    """
    anchors = {
        'difficulty': .5*z['accept_mean']+.5*learned_z['nm_difficulty_r0'],
        'multiscale': (z['accept_mean']+learned_z['nm_low_accept_40']+z['accept_qcenter_60'])/3,
        'qhard': .5*z['accept_mean']+.5*z['accept_qhard_5'],
    }
    gates = {'linear': z['q_mean'], 'negative': np.minimum(z['q_mean'], 0),
             'clipped': np.clip(z['q_mean'], -2, 2)}
    scores, recipes = {}, {}
    for anchor, values in anchors.items():
        name = f'qref_{anchor}_only'
        scores[name] = values
        recipes[name] = {'anchor': anchor, 'gate': None, 'q_weight': 0., 'center_weight': 0.}
        for gate, q in gates.items():
            for weight in (.25, .5, 1.):
                for center in (0., .25):
                    name = f'qref_{anchor}_{gate}_q{int(100*weight)}_c{int(100*center)}'
                    scores[name] = values + weight*q + center*z['q_qcenter_20']
                    recipes[name] = {'anchor': anchor, 'gate': gate, 'q_weight': weight, 'center_weight': center}
    return scores, recipes
