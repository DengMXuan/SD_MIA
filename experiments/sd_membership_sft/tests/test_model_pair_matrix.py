import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "experiments/sd_membership_sft/retrain_model_pairs.sh"


def test_dry_run_has_smoke_gate_and_exact_matrix():
    result = subprocess.run(
        [str(SCRIPT), "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "PYTHON": sys.executable},
        check=True,
        capture_output=True,
        text=True,
    )
    conditions = [
        line for line in result.stdout.splitlines() if "] condition " in line
    ]
    assert len(conditions) == 36
    assert conditions[0].startswith(
        "[smoke] condition pair=gemma4 benchmark=newstection "
        "epoch=1 seed=1919 gpu=3"
    )
    assert sum(line.startswith("[smoke]") for line in conditions) == 1
    assert sum("pair=qwen3 " in line for line in conditions) == 18
    assert sum("pair=gemma4 " in line for line in conditions) == 18
    assert len(
        {
            tuple(
                field
                for field in line.split(" command=", 1)[0].split()
                if field.startswith(("pair=", "benchmark=", "epoch=", "seed="))
            )
            for line in conditions
        }
    ) == 36
    assert all("--trainer full --optimizer adamw8bit" in line for line in conditions)
    assert all("--n-per-class 2000 --n-aux 2000" in line for line in conditions)
    assert result.stdout.rstrip().endswith(
        "[plan-ok] conditions=36 checkpoints=108 workers=4"
    )
