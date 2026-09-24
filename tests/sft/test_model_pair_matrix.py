import os
from pathlib import Path
import re
import subprocess
import sys


from experiments.paths import ROOT
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
    assert all(
        "--n-per-class 2000 --n-aux 2000 --n-audit-aux 600" in line
        for line in conditions
    )
    assert all("--target-lr 2e-5 --draft-lr 2e-5" in line for line in conditions)
    assert all("--distill-steps 384 --distill-temperature 2.0" in line for line in conditions)
    assert all("--seed " in line and "--data-seed " in line for line in conditions)
    for line in conditions:
        seed = re.search(r"\bseed=(\d+)\b", line)
        assert seed is not None
        assert f"PYTHONHASHSEED={seed.group(1)}" in line
        assert f"--seed {seed.group(1)} --data-seed {seed.group(1)}" in line
    assert all(
        "--target-batch-size 2 --target-grad-accum 8" in line
        and "--draft-batch-size 2 --draft-grad-accum 8" in line
        for line in conditions
    )
    assert all("--split-manifest" in line for line in conditions)
    assert all("artifacts/training/controlled_sft_v2/splits" in line for line in conditions)
    assert result.stdout.rstrip().endswith(
        "[plan-ok] conditions=36 checkpoints=108 workers=4"
    )
