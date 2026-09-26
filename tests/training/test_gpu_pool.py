"""Exercise concurrent CPU children, device masks, failures and cancellation."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from experiments.paths import ROOT
from experiments.shared.core import gpu_pool


def config(*args):
    parser = argparse.ArgumentParser()
    gpu_pool.add_arguments(parser)
    return gpu_pool.configuration(parser.parse_args(args))


def test_configuration_and_parent_visible_device_mapping(monkeypatch):
    assert config()['active_gpus'] == [0]
    assert config('--gpu', '3')['active_gpus'] == [3]
    assert config('--gpus', '2', '0', '1', '--workers', '2')['active_gpus'] == [2, 0]
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7,GPU-uuid,MIG-uuid')
    assert gpu_pool.visible_devices([2, 0]) == ['MIG-uuid', '7']
    with pytest.raises(ValueError, match='logical indices'):
        gpu_pool.visible_devices([3])
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    assert gpu_pool.visible_devices([2, 0]) == ['2', '0']


CHILD = '''
import json, os, sys, time
from pathlib import Path
index, gate = int(sys.argv[1]), Path(sys.argv[2])
print(json.dumps(dict(event='start', t=time.monotonic(),
    gpu=os.environ['CUDA_VISIBLE_DEVICES'], seed=os.environ['PYTHONHASHSEED'])), flush=True)
if index == 0:
    deadline = time.monotonic() + 10
    while not gate.exists():
        if time.monotonic() > deadline:
            raise RuntimeError('idle worker failed to drain the queue')
        time.sleep(.02)
elif index == 3:
    gate.write_text('done')
print(json.dumps(dict(event='end', t=time.monotonic())), flush=True)
sys.exit(7 if index == 1 else 0)
'''


def test_dynamic_queue_is_bounded_and_continues_after_failure(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-first,GPU-second')
    seeds = [1919, 1949, 1978, 1919]
    jobs = [gpu_pool.Job(f'condition-{i}', [sys.executable, '-c', CHILD, str(i), str(tmp_path / 'gate')], seed)
            for i, seed in enumerate(seeds)]
    before = signal.getsignal(signal.SIGTERM)
    rows = gpu_pool.run_jobs(jobs, scheduling=config('--gpus', '1', '0'), log_root=tmp_path / 'logs', cwd=ROOT)
    assert signal.getsignal(signal.SIGTERM) == before
    assert [r['exit_code'] for r in rows] == [0, 7, 0, 0]
    assert [r['gpu'] for r in rows] == [1, 0, 0, 0]
    intervals = []
    for row, seed in zip(rows, seeds):
        start, end = [json.loads(line) for line in Path(row['log']).read_text().splitlines()]
        assert start['seed'] == str(seed)
        assert start['gpu'] == ('GPU-second' if row['gpu'] == 1 else 'GPU-first')
        intervals += [(start['t'], 1, row['gpu']), (end['t'], -1, row['gpu'])]
    running, peak = set(), 0
    for _, delta, gpu in sorted(intervals):
        if delta == 1:
            assert gpu not in running
            running.add(gpu)
        else:
            running.remove(gpu)
        peak = max(peak, len(running))
    assert peak == 2 and not running
    status, = (tmp_path / 'logs').glob('*/STATUS.json')
    assert json.loads(status.read_text())['rows'] == rows
    assert os.environ['CUDA_VISIBLE_DEVICES'] == 'GPU-first,GPU-second'


def test_one_worker_and_failed_spawn(tmp_path, monkeypatch):
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES', raising=False)
    jobs = [gpu_pool.Job('missing', [str(tmp_path / 'missing-executable')], 1919)]
    jobs += [gpu_pool.Job(f'ok-{i}', [sys.executable, '-c', 'print("ok")'], 1978) for i in range(2)]
    rows = gpu_pool.run_jobs(jobs, scheduling=config('--gpus', '3', '1', '--workers', '1'),
                             log_root=tmp_path / 'logs', cwd=ROOT)
    assert [r['exit_code'] for r in rows] == [127, 0, 0]
    assert {r['gpu'] for r in rows} == {3}
    assert 'error' in rows[0]


def test_cpu_preparation_works_with_no_visible_cuda(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    job = gpu_pool.Job('prepare', [sys.executable, '-c', 'import os; print(repr(os.environ["CUDA_VISIBLE_DEVICES"]))'], 1919)
    with pytest.raises(ValueError, match='unavailable'):
        gpu_pool.run_jobs([job], scheduling=config(), log_root=tmp_path / 'gpu-logs', cwd=ROOT)
    assert not (tmp_path / 'gpu-logs').exists()
    rows = gpu_pool.run_jobs([job], scheduling=config(), log_root=tmp_path / 'cpu-logs', cwd=ROOT, use_cuda=False)
    assert rows[0]['exit_code'] == 0 and rows[0]['gpu'] is None
    assert Path(rows[0]['log']).read_text().strip() == "''"


def _wait_for(predicate, process, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        if process.poll() is not None:
            pytest.fail(f'coordinator exited early: {process.returncode}')
        time.sleep(.02)
    pytest.fail('timed out waiting for test worker')


def test_cleanup_kills_worker_which_ignores_term(tmp_path):
    ready = tmp_path / 'ready'
    code = ('import signal, sys, time; from pathlib import Path; '
            'signal.signal(signal.SIGTERM, signal.SIG_IGN); '
            'Path(sys.argv[1]).touch(); time.sleep(30)')
    process = subprocess.Popen([sys.executable, '-c', code, str(ready)], start_new_session=True)
    try:
        _wait_for(ready.exists, process)
        gpu_pool._stop({0: (process, None, {})}, timeout=.05)
        assert process.returncode == -signal.SIGKILL
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


@pytest.mark.parametrize('signum', [signal.SIGINT, signal.SIGTERM])
def test_cancellation_stops_children_and_does_not_launch_pending(tmp_path, signum):
    # Real coordinator + child + descendant, all CPU. The worker reaps its child
    # on TERM so the test can also assert the descendant PID has disappeared.
    child = tmp_path / 'child.py'
    child.write_text('''
import os, signal, subprocess, sys, time
from pathlib import Path
descendant = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
def stop(*args):
    descendant.wait(timeout=3)
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
Path(sys.argv[1]).write_text(f'{os.getpid()} {descendant.pid}')
time.sleep(30)
''')
    coordinator = '''
import sys
from pathlib import Path
from experiments.shared.core.gpu_pool import Job, run_jobs
root = Path(sys.argv[1])
jobs = [Job(str(i), [sys.executable, str(root/'child.py'), str(root/f'pid{i}')], 1919) for i in range(3)]
try:
    run_jobs(jobs, scheduling={'active_gpus':[0,1]}, log_root=root/'logs', cwd=root, use_cuda=False)
except KeyboardInterrupt:
    sys.exit(130)
'''
    with (tmp_path / 'coordinator.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-c', coordinator, str(tmp_path)], cwd=ROOT,
                                   stdout=log, stderr=subprocess.STDOUT)
        try:
            _wait_for(lambda: all((tmp_path / f'pid{i}').exists() for i in range(2)), process)
            process.send_signal(signum)
            assert process.wait(timeout=10) == 130
            assert not (tmp_path / 'pid2').exists()
            for i in range(2):
                for pid in (tmp_path / f'pid{i}').read_text().split():
                    with pytest.raises(ProcessLookupError):
                        os.kill(int(pid), 0)
            status, = (tmp_path / 'logs').glob('*/STATUS.json')
            assert [r['state'] for r in json.loads(status.read_text())['rows']] == [
                'interrupted', 'interrupted', 'not_started']
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
