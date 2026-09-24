"""Produce Pythia p/q and M1 Q/H caches for the existing M1 fitter."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from experiments.pretraining.cache import ROLE, freeze_partitions, validate_probability_cache
from experiments.pretraining.data import load_evaluation, load_model, sha256, TOKEN_CONTRACT
from experiments.baseline.runtime import RunProgress
from experiments.sd_membership_sft.analysis.m1_extract import _build_position_metadata, _validate_checkpoint, extract_qh_features, selected_blocks_for_model
from experiments.shared.methods.features import Q_FEATURE_NAMES, ACTIVATION_STAT_NAMES
from experiments.shared.models.token_scores import record_logprobabilities
from experiments.shared.training.training import set_seed


def extract(manifest_path, output_dir, *, device, stage='all', batch_size=1,
            attn_implementation='sdpa', progress_interval=30, selected_blocks=None):
    if batch_size < 1:
        raise ValueError('batch size must be positive')
    output_dir = Path(output_dir).resolve()
    evaluation = load_evaluation(Path(manifest_path), verify_draft=True)
    records = evaluation.members + evaluation.nonmembers
    labels = np.asarray([1] * len(evaluation.members) + [0] * len(evaluation.nonmembers), dtype=np.int64)
    ids = np.asarray([r.record_id for r in records])
    models = evaluation.manifest['models']
    probability_dir = output_dir / 'probabilities'
    base = dict(training_regime='pretraining', benchmark=evaluation.config.benchmark,
        models=models, dataset_manifest={'path': str(evaluation.manifest_path), 'sha256': sha256(evaluation.manifest_path)},
        token_contract=TOKEN_CONTRACT, records=len(records),
        record_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest(), eos_included=False,
        dtype='bfloat16' if device.type == 'cuda' else 'float32', attention_backend=attn_implementation)
    with RunProgress(output_dir, progress_interval) as progress:
        if stage in ('all', 'probabilities'):
            if (probability_dir / 'pq_gap_provenance.json').exists():
                validate_probability_cache(probability_dir, evaluation.manifest_path, models)
                progress.event('reusing validated probability cache')
            else:
                probability_dir.mkdir(parents=True, exist_ok=True)
                values = {}
                for role, key in (('target', 'target'), (ROLE, 'draft')):
                    progress.event('loading model', role=role, model=models[key])
                    model = load_model(models[key], device, attn_implementation, len(evaluation.tokenizer))
                    try:
                        rows = record_logprobabilities(model, records, evaluation.tokenizer, device,
                                                       batch_size, progress=progress)
                    finally:
                        del model
                        gc.collect()
                        if device.type == 'cuda':
                            torch.cuda.empty_cache()
                    lengths = np.asarray([len(row) for row in rows], dtype=np.int64)
                    values[role] = np.concatenate(rows).astype(np.float32)
                    if not np.array_equal(lengths, [len(r.response_ids) for r in records]):
                        raise RuntimeError('scored lengths differ from raw-text contract')
                    temporary = probability_dir / f'{role}.npz.tmp'
                    with temporary.open('wb') as stream:
                        np.savez_compressed(stream, lengths=lengths, logp=values[role], labels=labels, record_ids=ids,
                                            provenance_json=np.asarray(json.dumps(base)))
                    temporary.replace(probability_dir / f'{role}.npz')
                    progress.event('role saved', role=role)
                np.savez_compressed(probability_dir / 'pq_gap_token_logps.npz', lengths=lengths, **values)
                np.savez_compressed(probability_dir / 'pq_gap_scores.npz', labels=labels, record_ids=ids)
                metadata = {**base, 'kind': 'pretraining_probability_v1', 'tokens': int(lengths.sum()),
                    'files': {name: sha256(probability_dir / name) for name in ('pq_gap_token_logps.npz', 'pq_gap_scores.npz')}}
                (probability_dir / 'pq_gap_provenance.json').write_text(json.dumps(metadata, indent=2))
        if stage == 'probabilities':
            return
        probability = validate_probability_cache(probability_dir, evaluation.manifest_path, models)
        destination = output_dir / 'features'
        if destination.exists():
            raise FileExistsError(destination)
        examples, lengths, offsets, token_ids, prediction_positions, input_positions, eos_mask = _build_position_metadata(
            records, evaluation.tokenizer, include_eos=False)
        with np.load(probability_dir / 'pq_gap_scores.npz') as reference:
            if not np.array_equal(ids, reference['record_ids']) or not np.array_equal(labels, reference['labels']):
                raise RuntimeError('probability cache record IDs/labels mismatch')
        with np.load(probability_dir / 'pq_gap_token_logps.npz') as reference:
            cached_q = reference[ROLE].copy()
            if not np.array_equal(lengths, reference['lengths']):
                raise RuntimeError('probability cache length mismatch')
        progress.event('loading draft activations', model=models['draft'])
        model = load_model(models['draft'], device, attn_implementation, len(evaluation.tokenizer))
        temporary_dir = Path(tempfile.mkdtemp(prefix='features-partial-', dir=output_dir))
        try:
            if selected_blocks is None:
                _validate_checkpoint(model)
                blocks = selected_blocks_for_model(model)
            else:
                blocks = selected_blocks  # Explicit small-model integration-test seam.
            total = int(offsets[-1])
            q = np.lib.format.open_memmap(temporary_dir / 'q.npy', mode='w+', dtype=np.float32, shape=(total, 6))
            h = np.lib.format.open_memmap(temporary_dir / 'h.npy', mode='w+', dtype=np.float32, shape=(total, 40))
            extract_qh_features(model, examples, lengths, offsets, evaluation.tokenizer, device,
                                batch_size, q, h, selected_blocks=blocks, progress=progress)
            error = float(np.max(np.abs(q[:, 0] - cached_q)))
            if not np.allclose(q[:, 0], cached_q, atol=1e-4, rtol=1e-5):
                raise RuntimeError(f'fresh draft q disagrees with probability cache: {error}')
            q.flush(); h.flush()
            del q, h
            arrays = dict(labels=labels, record_ids=ids, lengths=lengths, offsets=offsets, token_ids=token_ids,
                          prediction_positions=prediction_positions, input_positions=input_positions, eos_mask=eos_mask)
            for name, value in arrays.items():
                np.save(temporary_dir / f'{name}.npy', value)
            names_h = [f'block{block+1}_{stat}' for block in blocks for stat in ACTIVATION_STAT_NAMES]
            feature = {**base, 'kind': 'pretraining_m1_features_v1', 'role': ROLE,
                'run_dir': str(evaluation.manifest_path.parent), 'total_tokens': total,
                'selected_blocks_zero_based': list(blocks),
                'feature_names': {'q': list(Q_FEATURE_NAMES), 'h': names_h, 'combined': list(Q_FEATURE_NAMES) + names_h},
                'activation_definition': 'decoder block residual output before final GPTNeoX LayerNorm',
                'model_config': model.config.to_dict(),
                'probability_cache_alignment': {'passed': True, 'max_abs_error': error},
                'files': {p.name: sha256(p) for p in temporary_dir.glob('*.npy')}}
            (temporary_dir / 'feature_manifest.json').write_text(json.dumps(feature, indent=2))
            freeze_partitions(labels, ids, temporary_dir / 'partition_manifest.json')
            temporary_dir.replace(destination)
            progress.event('features saved', directory=str(destination), tokens=total)
        finally:
            del model
            gc.collect()
            if temporary_dir.exists():
                shutil.rmtree(temporary_dir)
            if device.type == 'cuda':
                torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--stage', choices=('all', 'probabilities', 'features'), default='all')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--attn-implementation', choices=('eager', 'sdpa'), default='sdpa')
    parser.add_argument('--progress-interval', type=float, default=30)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    set_seed(20260824)
    extract(args.manifest, args.output_dir, device=device, stage=args.stage, batch_size=args.batch_size,
            attn_implementation=args.attn_implementation, progress_interval=args.progress_interval)


if __name__ == '__main__':
    main()
