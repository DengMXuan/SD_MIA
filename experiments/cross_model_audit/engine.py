"""Isolated fixed-probe scheduler; existing Qwen coordinator is never modified."""
from __future__ import annotations

from experiments.shared.audit.config import condition_settings

import argparse
import fcntl
import json
from pathlib import Path


from experiments.shared.audit import scheduler, reporting
from experiments.shared.audit.scheduler import wait_worker
from experiments.baseline import METHODS
from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.audit.provenance import read_result, sources_for
from experiments.shared.audit.baselines import BASELINE_DEFAULTS
from experiments.shared.audit.fixed import MAIN_METHODS

from experiments.shared.models.registry import pair_for, MODEL_PAIRS

ALL_METHODS = (*MAIN_METHODS["fixed"], *METHODS)


def make_tasks(model_root, output_root, benchmarks, epochs, seeds, settings, model_pair="gemma4"):
    spec = MODEL_PAIRS[model_pair]
    tasks = []
    for benchmark in benchmarks:
        for epoch in epochs:
            for seed in seeds:
                condition = dict(benchmark=benchmark, epoch=epoch, condition_seed=seed)
                key = f"{benchmark}/epoch{epoch}/seed{seed}"
                base = dict(run_dir=str((model_root / key).resolve()), condition=condition, settings=condition_settings(settings, seed))
                condition["model_pair"] = model_pair
                base["model_pair"] = model_pair
                key = model_pair + "/" + key
                tasks.append({**base, "id": key + "/baseline", "kind": "baseline", "methods": list(METHODS),
                              "output": str((output_root / key / "baseline").resolve())})
                for role in spec.roles:
                    for protocol in ("fixed",):
                        suffix = f"{role}/{protocol}"
                        tasks.append({**base, "id": key + "/" + suffix, "kind": "main", "draft_role": role,
                                      "protocol": protocol, "methods": list(MAIN_METHODS[protocol]),
                                      "output": str((output_root / key / suffix).resolve())})
    return tasks


from experiments.shared.models.readiness import ready


def inspect_task(task, *, check_sources=True):
    return scheduler.inspect_task(task, ready=ready, read_result=read_result,
                                  check_sources=check_sources)



from experiments.shared.audit.devices import check_gpus


def run_tasks(tasks, output_root, gpus):
    return scheduler.run_tasks(tasks, output_root, gpus, inspect_task=inspect_task,
                               check_gpus=check_gpus, worker_module='experiments.cross_model_audit.engine', wait=wait_worker)



def summarize(tasks, output_root, *, execution_root=None):
    def describe(task):
        spec = pair_for(task)
        return dict(model_pair=spec.name, adapter=spec.adapter,
                    target_model=spec.target, draft_model=spec.draft), spec.roles
    return reporting.summarize(tasks, output_root, methods=ALL_METHODS,
                               baseline_methods=METHODS, describe=describe,
                               title="Cross-model audit matrix", read_result=read_result,
                               execution_root=execution_root)



def execute_worker(task):
    import torch
    from experiments.shared.models.loading import prepare_records
    from experiments.shared.audit.baselines import run_baselines
    from experiments.shared.audit.fixed import run_main

    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("matrix worker requires a visible GPU")
    output = Path(task["output"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        valid, reason = ready(task)
        if not valid:
            raise RuntimeError(reason)
        spec = pair_for(task)
        if spec.adapter == "plain":
            cfg, prepared = prepare_records(Path(task["run_dir"]), "plain")
        else:
            cfg, prepared = prepare_records(Path(task["run_dir"]), spec.adapter, task.get("draft_role"))
        roles = ["target"] if task["kind"] == "baseline" else ["target", task["draft_role"]]
        sources = sources_for(Path(task["run_dir"]), roles, adapter=spec.adapter)
        (run_baselines if task["kind"] == "baseline" else run_main)(task, "cuda:0", cfg, prepared, sources)


def main():
    parser = argparse.ArgumentParser(description='Internal isolated cross-model worker')
    parser.add_argument('command', choices=('worker',))
    parser.add_argument('--task-file', type=Path, required=True)
    args = parser.parse_args()
    execute_worker(json.loads(args.task_file.read_text()))


if __name__ == '__main__':
    main()
