"""Single-condition main-method evaluation for ordinary and private checkpoints.

These functions do not schedule workers or train language models. A caller owns
device selection and any experiment loop; evaluation fits its own nonmember TCN.
"""
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path

from experiments.paths import ROOT, DATA, MODELS, TRAINING, QWEN_AUDIT, QWEN_CACHE
from experiments.sd_membership_sft.core.audit_runtime import _write_json
from experiments.sd_membership_sft.core.deployment_archive import sha256_file
from experiments.sd_membership_sft.audit.matrix_baselines import BASELINE_DEFAULTS
from .artifacts import digest, read_result, sources_for
from .model_registry import identify_pair


def _read_run(run_dir):
    artifact = json.loads((run_dir / "results.json").read_text())
    # A request without a final passport must never be treated as non-private.
    if "privacy" in artifact or (run_dir / "DP_REQUEST.json").exists():
        from experiments.dp_defense.artifacts import verify_run
        artifact = verify_run(run_dir)
    return identify_pair(artifact), artifact


def _task(run_dir, output, spec, artifact, role, seed, epochs):
    if role not in spec.roles:
        raise ValueError(f"choose a draft role from {spec.roles}")
    if type(epochs) is not int or epochs < 1 or type(seed) is not int or seed < 0:
        raise ValueError("positive detector epochs and nonnegative integer audit seed required")
    cfg = artifact["config"]
    task = dict(run_dir=str(run_dir), output=str(output), id=str(output),
                model_pair=spec.name, kind="main", protocol="fixed", draft_role=role,
                methods=["main_fixed_sparse_positive"],
                condition=dict(model_pair=spec.name, benchmark=cfg["benchmark"],
                               epoch=cfg["target_epochs"], condition_seed=cfg["seed"]),
                settings=dict(starts=["suffix64"], rounds_per_start=32, audit_seed=seed,
                              detector_epochs=epochs, baseline=dict(BASELINE_DEFAULTS)))
    if "privacy" in artifact:
        task["dp_request_key"] = artifact["privacy"]["request_key"]
    return task


def inspect_run(run_dir):
    """Read-only passport/weight checks; no inference or output creation.

Private runs additionally require verified stage checksums and privacy accounting.
"""
    from .engine import ready
    run_dir = Path(run_dir).resolve()
    spec, artifact = _read_run(run_dir)
    for role in spec.roles:
        task = _task(run_dir, run_dir, spec, artifact, role, 20260914, 30)
        valid, reason = ready(task)
        if not valid:
            raise ValueError(reason)
    return dict(model_pair=spec.name, adapter=spec.adapter, draft_roles=list(spec.roles),
                private="privacy" in artifact,
                condition=task["condition"], privacy_pairs=artifact.get("privacy", {}).get("pairs"))


def _validate_output(run_dir, output):
    # Check resolved physical weight paths as well as their training aliases.
    protected = [run_dir, QWEN_AUDIT, QWEN_CACHE, DATA, MODELS, TRAINING, ROOT / "experiments"]
    protected += [p.resolve() for name in ("checkpoints", "heads", "adapters")
                  if (p := run_dir / name).exists()]
    if any(output == path or output.is_relative_to(path) or path.is_relative_to(output)
           for path in protected):
        raise ValueError("main-method output must be separate from model/data/source and active Qwen directories")


def evaluate_main(run_dir, output_dir, *, draft_role, device="cuda:0",
                  audit_seed=20260914, detector_epochs=30):
    """Collect/reuse fixed B=2 observations and return a checked method report.

Use a distinct output directory for each model condition, draft role and privacy
budget. The same call resumes safely. DP accounting is attached automatically;
ordinary checkpoints are never relabeled as private. No baselines are scheduled.
"""
    from .engine import ready
    from .models import prepare_records
    from .main_method import run_main

    run_dir, output = Path(run_dir).resolve(), Path(output_dir).resolve()
    _validate_output(run_dir, output)
    with ExitStack() as stack:
        if (run_dir / "DP_REQUEST.json").exists():
            lock = stack.enter_context((run_dir / ".dp.lock").open("r"))
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        spec, artifact = _read_run(run_dir)
        task = _task(run_dir, output, spec, artifact, draft_role, audit_seed, detector_epochs)
        valid, reason = ready(task)
        if not valid:
            raise ValueError(reason)
        output.mkdir(parents=True, exist_ok=True)
        lock = stack.enter_context((output / ".worker.lock").open("a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = output / "MAIN_REQUEST.json"
        if marker.exists():
            if json.loads(marker.read_text()) != task:
                raise ValueError("main-method request changed; use a separate output directory")
        else:
            if any(p.name != ".worker.lock" for p in output.iterdir()):
                raise ValueError("refusing to adopt unrelated experiment results")
            _write_json(marker, task)
        cfg, prepared = prepare_records(run_dir, spec.adapter, draft_role)
        sources = sources_for(run_dir, ["target", draft_role], adapter=spec.adapter)
        if "privacy" in artifact:
            from experiments.dp_defense.artifacts import dp_runtime_files
            sources["files"] += [dict(path=str(p.resolve()), sha256=sha256_file(p))
                                 for p in [*dp_runtime_files(), run_dir / "DP_REQUEST.json"]]
        run_main(task, device, cfg, prepared, sources)
        method = task["methods"][0]
        report_dir = output / method
        report = read_result(report_dir, digest({"task": task, "method": method}), digest(sources))
        if "privacy" in artifact:
            budget_role = {"auxiliary_head": "draft_auxiliary_distilled",
                           "member_head": "draft_member_sft"}.get(draft_role, draft_role)
            privacy = {**artifact["privacy"]["pairs"][budget_role],
                       "target_epsilon_cap": artifact["privacy"]["stages"]["target"]["privacy"]["epsilon"],
                       "scope": "deployment_pair", "dp_request_key": artifact["privacy"]["request_key"]}
            if report.get("privacy") not in (None, privacy):
                raise ValueError("report privacy differs from the verified training passport")
            if report.get("privacy") is None:
                report = {**report, "privacy": privacy}
                _write_json(report_dir / "REPORT.json", report)
        elif "privacy" in report:
            raise ValueError("ordinary checkpoint cannot have a private report")
        return report
