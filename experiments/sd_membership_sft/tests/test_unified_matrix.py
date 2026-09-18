import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from experiments.sd_membership_sft import unified_matrix_preflight


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/sd_membership_sft/retrain_unified_matrix.sh"


def _environment(results_root: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHON": sys.executable,
        "RESULTS_ROOT": str(results_root),
        "MATRIX_GPUS": "3 4 5 6",
    }


def test_dry_run_combines_both_matrices_without_writing(tmp_path):
    results_root = tmp_path / "unified"
    result = subprocess.run(
        [str(SCRIPT), "--dry-run"],
        cwd=ROOT,
        env=_environment(results_root),
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.splitlines()
    plain_conditions = [line for line in lines if "] condition " in line]
    head_conditions = [line for line in lines if line.startswith("[condition]")]
    command_lines = [
        line
        for line in lines
        if "] condition " in line or line.startswith("[stage]")
    ]

    assert len(plain_conditions) == 36
    assert len(head_conditions) == 54
    assert sum("pair=qwen3 " in line for line in plain_conditions) == 18
    assert sum("pair=gemma4 " in line for line in plain_conditions) == 18
    assert sum("pair=qwen3_8b_eagle3 " in line for line in head_conditions) == 18
    assert sum("pair=llama31_8b_eagle3 " in line for line in head_conditions) == 18
    assert sum("pair=qwen35_9b_mtp " in line for line in head_conditions) == 18
    expected_split_root = str(results_root / "shared_splits")
    assert all(
        f"--split-manifest {expected_split_root}/" in line
        for line in command_lines
    )
    assert result.stdout.rstrip().endswith(
        "[unified-plan-ok] conditions=90 artifacts=270 "
        "workers=4 sequential_matrices=2"
    )
    assert not results_root.exists()


def test_status_combines_both_matrices_without_writing(tmp_path):
    results_root = tmp_path / "unified"
    result = subprocess.run(
        [str(SCRIPT), "--status"],
        cwd=ROOT,
        env=_environment(results_root),
        check=True,
        capture_output=True,
        text=True,
    )

    assert (
        "[model-pairs] [status] completed_conditions=0 expected_conditions=36"
        in result.stdout
    )
    assert (
        "[speculators] [status] completed_conditions=0 expected_conditions=54"
        in result.stdout
    )
    assert result.stdout.rstrip().endswith(
        "[unified-status] completed_conditions=0 expected_conditions=90 "
        "completed_artifacts=0 expected_artifacts=270"
    )
    assert not results_root.exists()


def test_preflight_combines_all_five_training_tokenizers(monkeypatch, tmp_path):
    prepared = {}
    args = SimpleNamespace(
        split_root=tmp_path / "shared_splits",
        model_revisions_env=tmp_path / "revisions.env",
        gpus=[2, 3, 4, 5],
        skip_gpu_check=False,
        skip_gpu_busy_check=True,
    )
    monkeypatch.setattr(unified_matrix_preflight, "parse_args", lambda: args)
    monkeypatch.setattr(
        unified_matrix_preflight,
        "validate_gpus",
        lambda indices, **kwargs: prepared.update(
            {"gpus": indices, "gpu_options": kwargs}
        ),
    )
    monkeypatch.setattr(
        unified_matrix_preflight,
        "validate_cached_models",
        lambda: {"eagle-qwen": object(), "eagle-llama": object(), "mtp": object()},
    )
    monkeypatch.setattr(
        unified_matrix_preflight,
        "validate_plain_models",
        lambda path: {"plain-qwen": object(), "plain-gemma": object()},
    )
    monkeypatch.setattr(
        unified_matrix_preflight,
        "prepare_shared_splits",
        lambda tokenizers, root: prepared.update(
            {"tokenizers": set(tokenizers), "split_root": root}
        ),
    )

    unified_matrix_preflight.main()

    assert prepared["gpus"] == [2, 3, 4, 5]
    assert prepared["gpu_options"] == {
        "skip_check": False,
        "skip_busy_check": True,
    }
    assert prepared["tokenizers"] == {
        "eagle-qwen",
        "eagle-llama",
        "mtp",
        "plain-qwen",
        "plain-gemma",
    }
    assert prepared["split_root"] == args.split_root
