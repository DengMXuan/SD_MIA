"""Create an isolated, single-commit source repository from the current worktree.

The existing Git repository and all of its history remain untouched. The output
belongs under ignored artifacts/ and has no remote configured.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
NEW_FILES = (
    "docs/reproduction.md", "docs/publication.md",
    "tools/prepare_public_release.py", "LICENSE",
)
EXCLUDED_PREFIXES = (
    "artifacts/", "experiments/results/", "docs/patent/",
    "docs/news_wiki_audit_", "docs/model_quality_experiment_report_",
    "docs/qwen3_kd_epoch1_main_method_report_", "docs/reports/experiment_audit_",
    "docs/experiment_presentation_plan_zh.md", "docs/introduction_figures_zh.md",
    "docs/history/method_development.md", "experiments/pretraining/DATA_INVENTORY.md",
)
MARKDOWN_LINK = re.compile(r"(?<!!)\[[^]]*\]\(([^)]+)\)")


def tracked_paths() -> set[str]:
    result = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return {p.decode() for p in result.split(b"\0") if p}


def verify(paths: set[str], *, draft: bool) -> None:
    if "LICENSE" not in paths and not draft:
        raise ValueError("Choose a license before creating a publishable release; use --draft only for review")
    for relative in paths:
        if relative.startswith(EXCLUDED_PREFIXES):
            raise ValueError(f"Excluded research material is still selected: {relative}")
        source = ROOT / relative
        if source.is_symlink() and not source.resolve().is_relative_to(ROOT):
            raise ValueError(f"Symlink leaves source tree: {relative}")
        if source.suffix != ".md":
            continue
        for line in source.read_text(errors="replace").splitlines():
            for link in MARKDOWN_LINK.findall(line):
                target = link.split("#", 1)[0].split(" ", 1)[0]
                if not target or target.startswith(("http:", "https:", "mailto:")):
                    continue
                resolved = (source.parent / target).resolve()
                if not resolved.is_relative_to(ROOT):
                    raise ValueError(f"Markdown link leaves source tree: {relative}: {link}")
                if resolved.is_dir():
                    continue
                linked = resolved.relative_to(ROOT).as_posix()
                if linked not in paths:
                    raise ValueError(f"Markdown link is absent from release: {relative}: {link}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="new directory under artifacts/ for the isolated source repository")
    parser.add_argument("--draft", action="store_true", help="allow a review copy before a license is chosen")
    args = parser.parse_args()
    output = args.output.resolve()
    artifacts = ROOT / "artifacts"
    if not output.is_relative_to(artifacts) or output == artifacts or output.exists():
        parser.error("--output must name a new directory beneath artifacts/")
    paths = {name for name in tracked_paths() | set(NEW_FILES) if (ROOT / name).exists()}
    try:
        verify(paths, draft=args.draft)
    except ValueError as error:
        parser.error(str(error))
    output.mkdir(parents=True)
    for relative in sorted(paths):
        source, destination = ROOT / relative, output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            destination.symlink_to(source.readlink())
        else:
            shutil.copy2(source, destination)
    subprocess.run(["git", "init", "-q", "--initial-branch=main"], cwd=output, check=True)
    subprocess.run(["git", "add", "-A"], cwd=output, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "Public source release"], cwd=output, check=True)
    print(f"Created {output} with {len(paths)} files and one new root commit")


if __name__ == "__main__":
    main()
