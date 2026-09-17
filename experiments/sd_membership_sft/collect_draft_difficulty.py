"""Collect observable difficulty features from existing frozen draft weights."""
from __future__ import annotations
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .data import (make_sft_example, collate_sft)
from .generalization import (load_draft_model)
from .m1_features import (Q_FEATURE_NAMES, q_features_from_logits)
from .scoring_common import (prepare_scoring_records, role_provenance)
from .audit_runtime import (_write_json)


def finalize_archive(output, contract, shape):
    """Also recover a crash after final array rename but before SOURCE.json."""
    values = np.load(output / 'q.npy', mmap_mode='r')
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError('invalid final feature array')
    _write_json(output / 'SOURCE.json', {'models_frozen': True, 'training_member_count': 0,
                'q_sha256': hashlib.sha256((output / 'q.npy').read_bytes()).hexdigest(),
                'provenance': contract['provenance'], 'feature_names': Q_FEATURE_NAMES})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg, data = prepare_scoring_records(args.run_dir)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(r.response_ids) for r in data.records], dtype=np.int64)
    offsets = np.r_[0, lengths.cumsum()]
    contract = {'provenance': role_provenance(cfg, args.run_dir, 'draft_auxiliary_distilled'),
                'feature_names': Q_FEATURE_NAMES, 'eos_included': False, 'models_frozen': True,
                'record_ids': data.record_ids.tolist(), 'batch_size': args.batch_size}
    contract = json.loads(json.dumps(contract))
    manifest = output / 'COLLECTION.json'
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError('collection contract differs')
    _write_json(manifest, contract)
    if (output / 'SOURCE.json').exists():
        return
    if (output / 'q.npy').exists():
        finalize_archive(output, contract, (int(offsets[-1]), len(Q_FEATURE_NAMES)))
        return
    np.save(output / 'record_ids.npy', data.record_ids)
    np.save(output / 'lengths.npy', lengths)
    partial = output / 'q.partial.npy'
    features = np.lib.format.open_memmap(partial, mode='r+' if partial.exists() else 'w+',
                 dtype=np.float32, shape=(int(offsets[-1]), len(Q_FEATURE_NAMES)))
    done_path = output / 'done.npy'
    done = np.load(done_path) if done_path.exists() else np.zeros(len(lengths), dtype=bool)
    device = torch.device(args.device)
    model = load_draft_model(args.run_dir, cfg.draft_model, 'draft_auxiliary_distilled', device, attn_implementation='sdpa')
    model.requires_grad_(False)
    model.eval()
    examples = [make_sft_example(replace(r, append_eos=False), data.tokenizer) for r in data.records]
    order = sorted(np.flatnonzero(~done), key=lambda i: len(examples[i]['input_ids']))
    with torch.inference_mode():
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start + args.batch_size]
            batch = {k: v.to(device) for k,v in collate_sft([examples[i] for i in indices], int(data.tokenizer.pad_token_id)).items()}
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits[:, :-1]
            valid = batch['labels'][:, 1:] != -100
            position = torch.cat([torch.linspace(0, 1, int(lengths[i]), device=device) for i in indices])
            loglength = torch.cat([torch.full((int(lengths[i]),), float(np.log(lengths[i])), device=device) for i in indices])
            values = q_features_from_logits(logits[valid], batch['labels'][:, 1:][valid], position, loglength).cpu().numpy()
            cursor = 0
            for i in indices:
                features[offsets[i]:offsets[i+1]] = values[cursor:cursor+lengths[i]]
                cursor += lengths[i]
                done[i] = True
            del logits, values, batch, valid
            if start % (40 * args.batch_size) == 0:
                features.flush()
                np.save(output / 'done.tmp.npy', done)
                (output / 'done.tmp.npy').replace(done_path)
                print(json.dumps({'completed': int(done.sum()), 'total': len(done)}), flush=True)
    features.flush()
    del features
    partial.replace(output / 'q.npy')
    np.save(done_path, done)
    finalize_archive(output, contract, (int(offsets[-1]), len(Q_FEATURE_NAMES)))
    print('COMPLETE', flush=True)

if __name__ == '__main__':
    main()
