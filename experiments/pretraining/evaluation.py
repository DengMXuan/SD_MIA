"""Current fixed B=2 main method on frozen pretrained models and raw text.

This module runs one explicitly requested condition. It neither fine-tunes a
language model nor schedules experiments. Official and temporal-proxy labels
remain distinguishable in the report.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import json
from pathlib import Path
import re

from huggingface_hub import snapshot_download
import numpy as np
import torch

from experiments.paths import ROOT
from experiments.pretraining.data import load_evaluation, load_model, sha256
from experiments.shared.audit.artifacts import (
    checkpoint_inventory, check_sources_light, digest, read_result, runtime_files,
)
from experiments.shared.audit.fixed import run_prepared_main
from experiments.shared.core.audit_partitions import deployment_partitions
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.core.deployment_archive import checkpoint_fingerprint
from experiments.shared.core.scoring_common import DeploymentScoringRecords
from experiments.shared.protocols.sd_protocol import FrozenAdapter


@dataclass(frozen=True)
class _AuditCounts:
    """Only the audit roles; a pretrained pair has no draft-training split."""
    members: int
    nonmembers: int
    audit_auxiliary: int
    detector_train: int
    detector_validation: int
    calibration: int

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in vars(self).values()):
            raise ValueError('audit counts must be positive integers')
        if self.detector_train + self.detector_validation + self.calibration != self.audit_auxiliary:
            raise ValueError('train, validation and calibration must exhaust the frozen auxiliaries')


def _snapshot(spec, local_files_only):
    path = Path(spec['repo_id']).expanduser()
    if path.is_dir():
        return path.resolve()
    if not re.fullmatch(r'[0-9a-f]{40}', spec['revision']):
        raise ValueError('Hub models require an exact 40-character commit revision')
    return Path(snapshot_download(
        spec['repo_id'], revision=spec['revision'], local_files_only=local_files_only,
        allow_patterns=['*.json', '*.safetensors', '*.bin', '*.txt', '*.model', '*.tiktoken'],
    )).resolve()


def _sources(manifest_path, manifest, models):
    files = set(runtime_files()) | set(Path(__file__).parent.glob('*.py'))
    files.update([manifest_path, manifest_path.parent / manifest['records_file']])
    checkpoints = []
    for path in models.values():
        before = checkpoint_inventory(path)
        fingerprint = checkpoint_fingerprint(path)
        if before != checkpoint_inventory(path):
            raise ValueError(f'pretrained model changed during fingerprinting: {path}')
        checkpoints.append(dict(path=str(path), sha256=fingerprint, inventory=before))
    return dict(files=[dict(path=str(p.resolve()), sha256=sha256(p)) for p in sorted(files)],
                checkpoints=checkpoints)


def _validate_output(output, manifest_path, models):
    protected = [manifest_path.parent, ROOT / 'experiments', ROOT / 'tests', ROOT / 'docs',
                 ROOT / '.git', *models.values()]
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError('evaluation output must be separate from data, models and source code')


def evaluate_main(manifest_path, output_dir, *, seed, device='cuda:0',
                  detector_epochs=30, detector_train=320, detector_validation=80,
                  calibration=200, local_files_only=True):
    """Evaluate one frozen MIMIR or temporal manifest using the current method.

    The required seed must equal data selection_seed. All test records retain
    their labels; only auxiliaries fit/validate/calibrate the detector. Explicit
    smaller partition sizes are supported for small official MIMIR caches.
    Models must already be cached unless local_files_only=False is explicit.
    Returns a verified REPORT.json dict; identical calls resume, changed inputs
    require a new output directory. See README.md for the data/label contract.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('seed must be an integer in [0, 2**32)')
    if type(detector_epochs) is not int or detector_epochs < 1:
        raise ValueError('detector_epochs must be positive')
    manifest_path, output = Path(manifest_path).resolve(), Path(output_dir).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest['selection_seed'] != seed:
        raise ValueError('audit seed must match frozen data selection_seed')
    counts = _AuditCounts(manifest['counts']['member'], manifest['counts']['nonmember'],
                          manifest['counts']['auxiliary'], detector_train, detector_validation, calibration)
    models = {role: _snapshot(manifest['models'][role], local_files_only) for role in ('target', 'draft')}
    _validate_output(output, manifest_path, models)
    sources = _sources(manifest_path, manifest, models)
    evaluation = load_evaluation(manifest_path, verify_draft=True, model_paths=models)
    records = evaluation.auxiliary + evaluation.members + evaluation.nonmembers
    prepared = DeploymentScoringRecords(
        tokenizer=evaluation.tokenizer, audit_auxiliary=evaluation.auxiliary,
        members=evaluation.members, nonmembers=evaluation.nonmembers, records=records,
        labels=np.asarray([0]*counts.audit_auxiliary + [1]*counts.members + [0]*counts.nonmembers),
        record_ids=np.asarray([r.record_id for r in records]),
        record_roles=np.asarray(['audit_auxiliary']*counts.audit_auxiliary
                                + ['member']*counts.members + ['nonmember']*counts.nonmembers),
    )
    parts = deployment_partitions(prepared.labels, prepared.record_ids, prepared.record_roles, counts, seed=seed)
    partitions = dict(schema='pretraining_main_partitions_v1', seed=seed,
                      manifest_sha256=sha256(manifest_path),
                      record_ids={name: prepared.record_ids[idx].tolist() for name, idx in parts.items()})
    context = dict(training_regime='pretraining', language_models_frozen=True,
                   language_model_finetuning=False, models=manifest['models'],
                   label_provenance=manifest['label_provenance'],
                   membership_verified=manifest['kind'] == 'mimir_pretraining_v1',
                   draft_exposure=manifest['draft_exposure'], token_contract=manifest['token_contract'],
                   partition_counts=vars(counts), partitions_file='PARTITIONS.json',
                   data_manifest=str(manifest_path), model_release=manifest.get('model_release'),
                   caveats=manifest.get('caveats', []))
    task = dict(id=str(output), output=str(output), kind='main', protocol='fixed',
                model_pair='pretrained', draft_role='pretrained',
                methods=['main_fixed_sparse_positive'],
                condition=dict(benchmark=manifest['benchmark'], condition_seed=seed, models=manifest['models']),
                settings=dict(audit_seed=seed, detector_epochs=detector_epochs),
                partitions_sha256=digest(partitions), sources_sha256=digest(sources), device=str(device),
                evaluation_context=context)

    def adapter_factory():
        # Load precisely the fingerprinted directories, never a moving Hub ref.
        pair = [load_model(dict(repo_id=str(models[role]), revision=manifest['models'][role]['revision']),
                           torch.device(device), vocab_size=len(evaluation.tokenizer))
                for role in ('target', 'draft')]
        return FrozenAdapter(*pair, 'plain', device)

    output.mkdir(parents=True, exist_ok=True)
    with (output / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker, partition_file = output / 'MAIN_REQUEST.json', output / 'PARTITIONS.json'
        if marker.exists():
            if json.loads(marker.read_text()) != task:
                raise ValueError('main-method request or frozen sources changed; use a separate output directory')
        else:
            if any(p.name != '.worker.lock' for p in output.iterdir()):
                raise ValueError('refusing to adopt unrelated experiment results')
            _write_json(marker, task)
        if partition_file.exists() and json.loads(partition_file.read_text()) != partitions:
            raise ValueError('frozen audit partitions changed')
        _write_json(partition_file, partitions)
        check_sources_light(sources)
        run_prepared_main(task, device, prepared, sources, parts=parts, adapter_factory=adapter_factory,
                          data_contract='pretraining_disjoint_nonmember_reference_v1', report_context=context)
        method = task['methods'][0]
        return read_result(output / method, digest(dict(task=task, method=method)), digest(sources))
