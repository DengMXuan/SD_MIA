"""Audit the largest fresh-FP32/archived-BF16 draft-logp differences on CPU.

Select the worst document in each class/domain, using only numerical mismatch.
This checks draft precision, not full delta equivalence or sampling variance.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging

import verify as v


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=v.ROOT/'artifacts/audits/pythia_delta_pilot_v1')
    p.add_argument('--threads', type=int, default=16)
    args = p.parse_args()
    plan = v.read_json(args.output/'PLAN.json')
    a = SimpleNamespace(datasets=list(dict.fromkeys(c['source'] for c in plan['conditions'])),
        seeds=list(dict.fromkeys(c['seed'] for c in plan['conditions'])),
        per_class=plan['per_class'], sample_seed=plan['sample_seed'], dtype=plan['dtype'],
        device=plan['device_type'], data_root=v.DEFAULT_DATA, audit_root=v.DEFAULT_AUDIT)
    reproduced, items = v.selection(a)
    v.require(reproduced == plan, 'source plan changed')
    selected, summary = [], []
    for c in plan['conditions']:
        old = v.feedback(a, c['source'], c['seed'])
        differences, worst = [], {}
        for r in items:
            if r['domain'] != c['source'] or r['seed'] != c['seed']:
                continue
            q = v.load_cached(args.output, 'draft', r)
            cached = old[r['record_id']]['logq']
            difference = q-cached
            differences.append(difference)
            label = r['label']
            if label not in worst or abs(difference).max() > worst[label][0]:
                worst[label] = (abs(difference).max(), r, q, cached)
        d = np.concatenate(differences)
        summary.append(dict(source=c['source'], seed=c['seed'], mean_absolute=float(abs(d).mean()),
                            mean_signed=float(d.mean()), absolute_quantiles=np.quantile(abs(d),[.5,.9,.99,1]).tolist()))
        selected.extend(worst.values())
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    logging.disable_progress_bar()
    spec = plan['models']['draft']
    path = Path('/home/mxd/.cache/huggingface/hub')/('models--'+spec['repo_id'].replace('/','--'))/'snapshots'/spec['revision']
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                dtype=torch.bfloat16, attn_implementation='sdpa').eval().requires_grad_(False)
    checks = []
    for mode in ('cpu_bf16', 'cpu_bf16_weights_fp32_math'):
        if mode.endswith('fp32_math'):
            model.float()
        for _, r, fresh, archived in selected:
            ids = torch.tensor([r['token_ids']])
            with torch.inference_mode():
                logits = model(input_ids=ids, use_cache=False).logits[0,:-1,:len(tokenizer)].float()
                q = torch.log_softmax(logits,-1).gather(-1,ids[0,1:,None]).squeeze(-1).numpy().astype(np.float64)
            error = q-archived
            check = dict(mode=mode, source=r['domain'], seed=r['seed'], label=r['label'], record_id=r['record_id'],
                fp32_vs_archive_max_abs=float(abs(fresh-archived).max()),
                checked_vs_archive_max_abs=float(abs(error).max()),
                checked_vs_archive_mean_abs=float(abs(error).mean()),
                checked_vs_archive_rmse=float(np.sqrt(np.mean(error**2))))
            checks.append(check)
            print(check, flush=True)
    v.write_json(args.output/'PRECISION.json',dict(plan_sha256=v.sha256(args.output/'PLAN.json'),
        script_sha256=v.sha256(__file__), draft_only=True, summary=summary, checks=checks,
        note='Worst-mismatch documents selected per class/domain. Both primary delta endpoints use fresh FP32. '
             'Archived GPU BF16 feedback is not numerically identical; comparing it with fresh alpha does not isolate Bernoulli noise.'))


if __name__ == '__main__':
    main()
