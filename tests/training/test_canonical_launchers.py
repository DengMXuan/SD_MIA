"""Real shell/queue integration using CPU stand-ins for GPUs and trainers."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments.paths import ROOT
from experiments.launchers import devices

SCRIPTS = ROOT / 'experiments/scripts/training'


@pytest.mark.parametrize('name,count', [
    ('train_effectiveness_qwen3_gemma4_epoch1_3.sh', 36),
    ('train_effectiveness_eagle3_mtp_epoch1_3.sh', 54),
    ('train_effectiveness_fivepairs_epoch1_3.sh', 90),
])
@pytest.mark.parametrize('gpus', [['2'], ['0', '2'], ['0', '1', '2']])
def test_training_previews_accept_any_gpu_count(tmp_path, name, count, gpus):
    result = subprocess.run(['bash', str(SCRIPTS / name), 'dry-run', '--gpus', *gpus],
        cwd=tmp_path, env={**os.environ, 'PYTHON': sys.executable, 'RESULTS_ROOT': str(tmp_path / 'results')},
        text=True, capture_output=True, check=True)
    assert f'conditions={count}' in result.stdout
    assert f'workers={len(gpus)}' in result.stdout
    assert not (tmp_path / 'results').exists()


def fake_runtime(tmp_path):
    smi = tmp_path / 'nvidia-smi'
    # Synthetic physical IDs keep CPU tests off real experiment GPU lock files.
    smi.write_text('#!/bin/sh\nprintf "900000, GPU-first, 0\\n900001, GPU-second, 0\\n"\n')
    smi.chmod(0o755)
    python = tmp_path / 'python'
    python.write_text(f'#!{sys.executable}\n' + r'''
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
module = args[args.index('-m') + 1]
if module.startswith('experiments.launchers.'):
    os.execv(sys.executable, [sys.executable, *args])
def value(flag): return args[args.index(flag)+1]
output = Path(value('--output-dir'))
event = dict(module=module, output=str(output), gpu=os.environ['CUDA_VISIBLE_DEVICES'])
with open(os.environ['TRAIN_EVENTS'], 'a') as stream:
    stream.write(json.dumps(event) + '\n')
if os.environ.get('TRAIN_FAIL_SMOKE') == '1':
    sys.exit(7)
if module.endswith('.plain'):
    checkpoints = [output/'checkpoints'/role for role in ('target','draft_auxiliary_distilled','draft_member_sft')]
else:
    stage = next(a for a in args if a in ('mtp-source','mtp-target','mtp-head','eagle-target','eagle-head'))
    if stage == 'mtp-source': checkpoints = [Path(value('--source-head'))]
    elif stage.endswith('-target'): checkpoints = [output/'checkpoints/target']
    else: checkpoints = [output/'heads'/('auxiliary_head' if value('--variant')=='aux' else 'member_head')]
for folder in checkpoints:
    folder.mkdir(parents=True, exist_ok=True)
    for name in ('config.json', 'model.safetensors', '_COMPLETE.json'):
        (folder/name).write_text('{}')
if module.endswith('.plain'): (output/'results.json').write_text('{}')
time.sleep(.01)
''')
    python.chmod(0o755)
    return {**os.environ, 'PATH': str(tmp_path) + os.pathsep + os.environ['PATH'],
            'PYTHON': str(python), 'RESULTS_ROOT': str(tmp_path / 'results'),
            'MATRIX_SKIP_PREFLIGHT': '1', 'TRAIN_EVENTS': str(tmp_path / 'events.jsonl'),
            'CUDA_VISIBLE_DEVICES': 'GPU-first,GPU-second'}


@pytest.mark.parametrize('name,count,events', [
    ('train_effectiveness_qwen3_gemma4_epoch1_3.sh', 36, 36),
    ('train_effectiveness_eagle3_mtp_epoch1_3.sh', 54, 163),
])
def test_actual_training_shell_dispatches_every_condition_and_resumes(tmp_path, name, count, events):
    env = fake_runtime(tmp_path)
    command = ['bash', str(SCRIPTS / name), 'run', '--gpus', '0', '1']
    result = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f'completed_conditions={count}' in result.stdout
    rows = [json.loads(line) for line in (tmp_path / 'events.jsonl').read_text().splitlines()]
    assert len(rows) == events
    assert {row['gpu'] for row in rows} == {'GPU-first', 'GPU-second'}
    statuses = list((tmp_path / 'results/executions').glob('*/STATUS.json'))
    assert statuses and all(row['state'] == 'complete' for path in statuses for row in json.loads(path.read_text())['rows'])
    result = subprocess.run(command, env=env, cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert len((tmp_path / 'events.jsonl').read_text().splitlines()) == events


def test_failed_smoke_prevents_remaining_training(tmp_path):
    env = {**fake_runtime(tmp_path), 'TRAIN_FAIL_SMOKE': '1'}
    result = subprocess.run(['bash', str(SCRIPTS / 'train_effectiveness_qwen3_gemma4_epoch1_3.sh'),
                             'run', '--gpus', '0', '1'], env=env, cwd=ROOT,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert len((tmp_path / 'events.jsonl').read_text().splitlines()) == 1


def test_device_admission_uses_mask_and_rejects_busy_cards(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-second,GPU-first')
    monkeypatch.setattr(devices, 'inventory', lambda: {
        'GPU-first': ('0', 'GPU-first', 4000), 'GPU-second': ('1', 'GPU-second', 8)})
    assert devices.check_idle([0]) == [('1', 'GPU-second')]
    with pytest.raises(RuntimeError, match='already uses'):
        devices.check_idle([1])


def test_legacy_shells_resolve_to_canonical_files():
    for path in ROOT.joinpath('experiments').rglob('*.sh'):
        if 'scripts' in path.parts and path.is_relative_to(ROOT / 'experiments/scripts'):
            continue
        assert path.is_symlink(), path
        assert path.resolve().is_relative_to(ROOT / 'experiments/scripts')
