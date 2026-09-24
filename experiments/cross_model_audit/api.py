"""Historical single-condition entry points; common implementation is shared.audit.evaluation."""
from experiments.shared.audit import evaluation


def _verification():
    from experiments.dp_defense.artifacts import evaluation_verification
    return evaluation_verification()


def inspect_run(run_dir):
    return evaluation.inspect_run(run_dir, verification=_verification())


def evaluate_main(run_dir, output_dir, *, draft_role, device="cuda:0",
                  audit_seed=20260914, detector_epochs=30):
    return evaluation.evaluate_main(run_dir, output_dir, draft_role=draft_role, device=device,
                                    audit_seed=audit_seed, detector_epochs=detector_epochs,
                                    verification=_verification())
