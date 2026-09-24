"""Read-only GPU admission policy for audit launchers."""
import os
import subprocess

def check_gpus(gpus):
    if not gpus or len(set(gpus)) != len(gpus) or any(g < 0 for g in gpus):
        raise ValueError("choose distinct nonnegative physical GPUs")
    limit = int(os.environ.get("SD_AUDIT_GPU_MAX_USED_MIB", "1024"))
    if limit < 0:
        raise ValueError("SD_AUDIT_GPU_MAX_USED_MIB must be nonnegative")
    inventory = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    used = {}
    for line in inventory.splitlines():
        if line.strip():
            index, memory = line.split(",")
            used[int(index)] = int(memory.strip())
    for gpu in gpus:
        if gpu not in used:
            raise RuntimeError(f"GPU {gpu} is absent")
        if used[gpu] > limit:
            raise RuntimeError(
                f"GPU {gpu} uses {used[gpu]} MiB, above the background allowance of {limit} MiB "
                "(SD_AUDIT_GPU_MAX_USED_MIB)"
            )
