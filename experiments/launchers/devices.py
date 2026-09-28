"""GPU admission for new launchers, reusing the existing bounded process queue."""
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import subprocess

from experiments.shared.core import gpu_pool


def inventory():
    try:
        raw = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,uuid,memory.used', '--format=csv,noheader,nounits'],
            text=True, stderr=subprocess.STDOUT, timeout=15)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError('cannot verify GPU usage; no model will be loaded') from error
    devices = {}
    for line in raw.splitlines():
        if line.strip():
            index, uuid, memory = (item.strip() for item in line.split(',', 2))
            devices[index] = devices[uuid] = (index, uuid, int(memory))
    return devices


def check_idle(gpus):
    threshold = int(os.environ.get('SD_AUDIT_GPU_MAX_USED_MIB', '1024'))
    if threshold < 0:
        raise ValueError('GPU memory threshold must be nonnegative')
    devices = inventory()
    selected = []
    for visible in gpu_pool.visible_devices(gpus):
        if visible not in devices:
            raise RuntimeError(f'GPU {visible} is unavailable in nvidia-smi')
        index, uuid, memory = devices[visible]
        if memory > threshold:
            raise RuntimeError(f'GPU {visible} already uses {memory} MiB (limit {threshold} MiB)')
        selected.append((index, uuid))
    return selected


def run_jobs(jobs, *, scheduling, cwd, log_root, use_cuda=True):
    jobs = list(jobs)
    if not use_cuda or not jobs:
        return gpu_pool.run_jobs(jobs, scheduling=scheduling, cwd=cwd, log_root=log_root, use_cuda=use_cuda)
    selected = check_idle(scheduling['active_gpus'][:len(jobs)])
    with ExitStack() as stack:
        # Match the existing audit scheduler's physical-device lock names.
        for index, _ in sorted(selected):
            lock = stack.enter_context(Path(f'/tmp/sd_mia_audit_gpu_{os.getuid()}_{index}.lock').open('a'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f'GPU {index} is reserved by another experiment') from error
        check_idle(scheduling['active_gpus'][:len(jobs)])
        return gpu_pool.run_jobs(jobs, scheduling=scheduling, cwd=cwd, log_root=log_root)


if __name__ == '__main__':
    check_idle([0])
