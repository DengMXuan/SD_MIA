"""Bounded subprocess queue for independent conditions, one job per GPU.

No model/runtime imports: previews work without CUDA. Each child sees only its
assigned device as cuda:0, keeping experiment requests independent of dispatch.
"""
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid


@dataclass(frozen=True)
class Job:
    id: str
    command: list[str]
    seed: int


def add_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpu', type=int, help='one logical CUDA index (default: 0)')
    group.add_argument('--gpus', type=int, nargs='+', help='logical CUDA indices in CUDA_VISIBLE_DEVICES')
    parser.add_argument('--workers', type=int, help='concurrent conditions; default: one per selected GPU')
    parser.add_argument('--log-root', type=Path, help='separate per-attempt worker logs and status')


def configuration(args):
    gpus = args.gpus if args.gpus is not None else [0 if args.gpu is None else args.gpu]
    workers = len(gpus) if args.workers is None else args.workers
    if not gpus or any(gpu < 0 for gpu in gpus) or len(set(gpus)) != len(gpus):
        raise ValueError('choose unique nonnegative GPU indices')
    if not 1 <= workers <= len(gpus):
        raise ValueError('--workers must be between 1 and the number of selected GPUs')
    return dict(gpus=gpus, workers=workers, active_gpus=gpus[:workers],
                policy='next condition on next free worker; one worker per GPU',
                worker_device='cuda:0')


def visible_devices(gpus):
    """Map parent logical ordinals through its mask, including GPU/MIG UUIDs."""
    mask = os.environ.get('CUDA_VISIBLE_DEVICES')
    if mask is None:
        return [str(gpu) for gpu in gpus]
    devices = [item.strip() for item in mask.split(',')]
    if (not all(devices) or '-1' in devices or len(set(devices)) != len(devices)
            or max(gpus) >= len(devices)):
        raise ValueError('selected GPU is unavailable in CUDA_VISIBLE_DEVICES; use logical indices')
    return [devices[gpu] for gpu in gpus]


def _save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def _signal_group(process, signum):
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _stop(active, timeout=5):
    # Signal all process groups first, then wait once for the whole pool. Model
    # data-loader descendants must not survive a cancelled coordinator.
    for process, _, _ in active.values():
        _signal_group(process, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    for process, _, _ in active.values():
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for process, _, _ in active.values():
        _signal_group(process, signal.SIGKILL)
        process.wait()


def run_jobs(jobs, *, scheduling, log_root, cwd, use_cuda=True):
    """Execute unique jobs; ordinary failures do not stop unrelated conditions.

    SIGINT/SIGTERM stop dispatch and clean up active process groups. Completed
    experiment stages remain owned by their existing resume protocols. Callers
    choose a log_root separate from individual experiment output folders.
    """
    jobs = list(jobs)
    if len({job.id for job in jobs}) != len(jobs):
        raise ValueError('duplicate condition in worker queue')
    if not jobs:
        return []
    gpus = scheduling['active_gpus'][:len(jobs)]
    if not gpus:
        raise ValueError('worker queue requires at least one slot')
    devices = visible_devices(gpus) if use_cuda else [''] * len(gpus)
    attempt = Path(log_root).resolve() / (
        datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:12])
    attempt.mkdir(parents=True)
    rows = [dict(id=job.id, seed=job.seed, state='pending') for job in jobs]
    _save(attempt / 'JOBS.json', [asdict(job) for job in jobs])
    status = dict(scheduling=scheduling, use_cuda=use_cuda, rows=rows)
    _save(attempt / 'STATUS.json', status)
    print(json.dumps(dict(execution=str(attempt), workers=len(gpus))), flush=True)
    pending, active = deque(enumerate(jobs)), {}
    stop_requested = False

    def interrupted(signum, frame):
        # Defer raising until a just-spawned child has entered active. Otherwise
        # a signal between Popen and registration could leave an orphan worker.
        nonlocal stop_requested
        stop_requested = True

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, interrupted)
    try:
        while pending or active:
            if stop_requested:
                raise KeyboardInterrupt
            for slot, (process, log, row) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                if code in (130, -signal.SIGINT, -signal.SIGTERM):
                    raise KeyboardInterrupt
                log.close()
                del active[slot]
                row.update(state='complete' if code == 0 else 'failed', exit_code=code)
                print(json.dumps(row), flush=True)
                _save(attempt / 'STATUS.json', status)
            for slot, (gpu, device) in enumerate(zip(gpus, devices)):
                if stop_requested:
                    raise KeyboardInterrupt
                if slot in active or not pending:
                    continue
                index, job = pending.popleft()
                row = rows[index]
                log_path = attempt / f'{index:04d}.log'
                row.update(state='running', gpu=gpu if use_cuda else None,
                           visible_device=device, worker_device='cuda:0' if use_cuda else None,
                           log=str(log_path))
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': device,
                       'PYTHONHASHSEED': str(job.seed), 'PYTHONDONTWRITEBYTECODE': '1',
                       'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'}
                log = log_path.open('w')
                try:
                    process = subprocess.Popen(job.command, cwd=cwd, env=env,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                except OSError as error:
                    log.close()
                    row.update(state='failed', exit_code=127, error=str(error))
                else:
                    active[slot] = (process, log, row)
                    row['pid'] = process.pid
                print(json.dumps(row), flush=True)
                _save(attempt / 'STATUS.json', status)
            if active:
                time.sleep(.1)
    except BaseException:
        # Ignore repeated terminal signals until all owned children are gone.
        for sig in handlers:
            signal.signal(sig, signal.SIG_IGN)
        _stop(active)
        for process, log, row in active.values():
            log.close()
            row.update(state='interrupted', exit_code=process.returncode)
        for row in rows:
            if row['state'] == 'running':
                row.update(state='interrupted', exit_code=None)
        for index, _ in pending:
            rows[index]['state'] = 'not_started'
        _save(attempt / 'STATUS.json', status)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return rows
