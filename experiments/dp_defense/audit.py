"""Refit the existing B=2 difficulty TCN separately for each DP draft condition."""
from __future__ import annotations

from experiments.shared.audit.config import audit_settings
import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path

from experiments.shared.core.audit_runtime import _write_json
from experiments.shared.core.deployment_archive import sha256_file
from experiments.shared.audit.provenance import digest, sources_for, read_result
from experiments.shared.models.registry import identify_pair
from experiments.shared.models.loading import prepare_records
from experiments.shared.audit.fixed import run_main
from experiments.shared.audit.baselines import BASELINE_DEFAULTS, METHODS
from experiments.dp_defense.artifacts import dp_runtime_files, verify_run


def make_tasks(run_dir, output, artifact, *, seed=20260914, epochs=30, baselines=False):
    if epochs <= 0:
        raise ValueError("detector epochs must be positive")
    cfg = artifact["config"]
    spec = identify_pair(artifact)
    condition = dict(benchmark=cfg["benchmark"], epoch=cfg["target_epochs"], condition_seed=cfg["seed"], model_pair=spec.name)
    settings = audit_settings(audit_seed=seed, detector_epochs=epochs)
    common = dict(run_dir=str(run_dir.resolve()), condition=condition, settings=settings, model_pair=spec.name,
                  dp_request_key=artifact["privacy"]["request_key"])
    tasks = []
    for role in spec.roles:
        tasks.append({**common, "id": str(output.resolve() / role / "fixed"),
                      "output": str(output.resolve() / role / "fixed"), "kind": "main",
                      "draft_role": role, "protocol": "fixed", "methods": ["main_fixed_sparse_positive"]})
    if baselines:
        tasks.append({**common, "id": str(output.resolve() / "baseline"),
                      "output": str(output.resolve() / "baseline"), "kind": "baseline", "methods": list(METHODS)})
    return tasks


def audit_sources(run_dir, roles, *, adapter="plain"):
    sources = sources_for(run_dir, roles, adapter=adapter)
    sources["files"] += [{"path": str(p.resolve()), "sha256": sha256_file(p)}
                         for p in [*dp_runtime_files(), run_dir / "DP_REQUEST.json"]]
    return sources


def summarize(tasks, output, artifact):
    rows = []
    for task in tasks:
        for method in task["methods"]:
            folder = Path(task["output"]) / method
            row = dict(**task["condition"], method=method, draft_role=task.get("draft_role", "target_only"),
                       status="missing", report=str(folder / "REPORT.json"))
            if (folder / "REPORT.json").exists():
                report = read_result(folder, digest({"task": task, "method": method}))
                if "privacy" not in report:
                    # A worker can stop after the legacy scorer saves its report
                    # but before this wrapper attaches pair accounting. Run again
                    # to finish without repeating collection or detector fitting.
                    rows.append(row)
                    continue
                row.update(status="complete", metrics=report["metrics"], cost=report["cost"],
                           privacy=report["privacy"])
            rows.append(row)
    result = dict(complete=all(r["status"] == "complete" for r in rows), rows=rows,
                  pairs=artifact["privacy"]["pairs"], dp_request_key=artifact["privacy"]["request_key"])
    _write_json(output / "SUMMARY.json", result)
    return result


def run_audit(run_dir, output, *, device="cuda:0", seed=20260914, epochs=30, baselines=False, execute=True):
    run_dir, output = run_dir.resolve(), output.resolve()
    if output == run_dir or output in run_dir.parents or run_dir in output.parents:
        raise ValueError("audit output must be separate from training artifacts")
    artifact = verify_run(run_dir)
    tasks = make_tasks(run_dir, output, artifact, seed=seed, epochs=epochs, baselines=baselines)
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        training_lock = stack.enter_context((run_dir / ".dp.lock").open("r"))
        fcntl.flock(training_lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        audit_lock = stack.enter_context((output / ".dp_audit.lock").open("a"))
        fcntl.flock(audit_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        request = output / "DP_AUDIT.json"
        if request.exists():
            if json.loads(request.read_text()) != tasks:
                raise ValueError("DP audit request changed; use a separate output directory")
        else:
            if any(p.name != ".dp_audit.lock" for p in output.iterdir()):
                raise ValueError("refusing to adopt existing non-DP audit results")
            _write_json(request, tasks)
        if execute:
            import torch
            from experiments.shared.audit.baselines import run_baselines
            torch.set_num_threads(2)
            if torch.device(device).type != "cuda" or not torch.cuda.is_available():
                raise RuntimeError("real-model audit requires CUDA")
            spec = identify_pair(artifact)
            for task in tasks:
                cfg, prepared = prepare_records(run_dir, spec.adapter, task.get("draft_role"))
                roles = ["target"] if task["kind"] == "baseline" else ["target", task["draft_role"]]
                sources = audit_sources(run_dir, roles, adapter=spec.adapter)
                (run_baselines if task["kind"] == "baseline" else run_main)(task, device, cfg, prepared, sources)
                for method in task["methods"]:
                    path = Path(task["output"]) / method / "REPORT.json"
                    report = json.loads(path.read_text())
                    pair_role = task.get("draft_role", "draft_auxiliary_distilled")
                    pair_role = {"auxiliary_head": "draft_auxiliary_distilled", "member_head": "draft_member_sft"}.get(pair_role, pair_role)
                    # Extend the ordinary schema without touching old evaluators.
                    privacy = {**artifact["privacy"]["pairs"][pair_role],
                               "target_epsilon_cap": artifact["privacy"]["stages"]["target"]["privacy"]["epsilon"],
                               "scope": "target_only" if task["kind"] == "baseline" else "deployment_pair",
                               "dp_request_key": artifact["privacy"]["request_key"]}
                    if report.get("privacy") != privacy:
                        _write_json(path, {**report, "privacy": privacy})
        return summarize(tasks, output, artifact)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "summarize"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-seed", type=int, default=20260914)
    parser.add_argument("--detector-epochs", type=int, default=30)
    parser.add_argument("--include-baselines", action="store_true")
    args = parser.parse_args()
    result = run_audit(args.run_dir, args.output_dir, device=args.device, seed=args.audit_seed,
                       epochs=args.detector_epochs, baselines=args.include_baselines, execute=args.command == "run")
    print(json.dumps({"complete": result["complete"], "rows": len(result["rows"])}))
    if not result["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
