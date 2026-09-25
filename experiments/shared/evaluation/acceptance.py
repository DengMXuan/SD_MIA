"""Exact one-token expected acceptance on teacher-forced response prefixes."""
import numpy as np
import torch

from experiments.shared.data.data import prompt_prefix_ids


def mean_ci(values, repeats, seed):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or repeats < 1:
        raise ValueError('finite nonempty scores and positive bootstrap repeats required')
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, len(values), replace=True).mean() for _ in range(repeats)]
    return dict(mean=float(values.mean()), ci95_low=float(np.quantile(means, .025)),
                ci95_high=float(np.quantile(means, .975)), count=len(values))


@torch.inference_mode()
def record_acceptance(adapter, record, tokenizer):
    """Include response EOS as in the historical asset-quality evaluator.

    Unmapped EAGLE vocabulary entries have zero proposal mass, not a masked
    denominator. Ground-truth vocabulary coverage is a separate diagnostic.
    """
    prompt = prompt_prefix_ids(record, tokenizer)
    response = list(record.response_ids)
    if record.append_eos and tokenizer.eos_token_id is not None:
        response.append(int(tokenizer.eos_token_id))
    if not prompt or not response:
        raise ValueError('nonempty prompt and response required')
    logp, logq = adapter.rows(prompt + response)
    start = len(prompt) - 1
    p, q = logp[start:start + len(response)], logq[start:start + len(response)]
    if p.shape != q.shape or len(p) != len(response):
        raise ValueError('target/draft response positions do not align')
    supported = torch.isfinite(q).any(-1)
    # Current SFT prompts have many tokens: even MTP must cover every response.
    if not supported.all():
        raise ValueError('draft has no prediction for a response position')
    if (torch.isnan(p).any() or torch.isnan(q).any()
            or not torch.allclose(torch.logsumexp(p, -1), torch.zeros(len(p), device=p.device), atol=1e-4)
            or not torch.allclose(torch.logsumexp(q, -1), torch.zeros(len(q), device=q.device), atol=1e-4)):
        raise ValueError('acceptance requires normalized distributions')
    # Reduce in chunks to avoid another full sequence-by-vocabulary FP32 copy.
    overlap, agreement = [], []
    for offset in range(0, len(p), 32):
        pp, qq = p[offset:offset + 32], q[offset:offset + 32]
        overlap.append(torch.minimum(pp, qq).exp().sum(-1))
        agreement.append(pp.argmax(-1).eq(qq.argmax(-1)).float())
    truth = torch.tensor(response, device=q.device).unsqueeze(-1)
    coverage = torch.isfinite(q.gather(-1, truth)).float().mean()
    return dict(exact_acceptance=float(torch.cat(overlap).mean()),
                top1_agreement=float(torch.cat(agreement).mean()),
                truth_vocab_coverage=float(coverage), response_positions=len(response))


def summarize_acceptance(rows, repeats, seed):
    result = {}
    for role in ('overall', 'member', 'nonmember', 'auxiliary'):
        selected = rows if role == 'overall' else [r for r in rows if r['role'] == role]
        result[role] = {metric: mean_ci([r[metric] for r in selected], repeats, seed)
                        for metric in ('exact_acceptance', 'top1_agreement', 'truth_vocab_coverage')}
    return result
