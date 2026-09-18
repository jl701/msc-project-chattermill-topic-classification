"""Run repository correctness checks and the maintained-interface quality gate."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
MAINTAINED = (
    "src/msc_project/demo.py",
    "src/msc_project/reporting.py",
    "src/msc_project/evaluation/resampling.py",
    "src/msc_project/baselines/candidate_tfidf.py",
    "scripts/check_quality.py",
    "scripts/audit_repository.py",
    "tests/test_resampling.py",
    "tests/test_candidate_tfidf_cpu.py",
    "tests/test_repository_interfaces.py",
    "tests/test_quality_tools.py",
)
CPU_TESTS = (
    "tests/test_quality_tools.py",
    "tests/test_resampling.py",
    "tests/test_candidate_tfidf_cpu.py",
    "tests/test_repository_interfaces.py",
    "tests/test_taxonomy_training_policy_sensitivity.py",
    "tests/test_taxonomy_two_stage.py",
    "tests/test_taxonomy_two_stage_training.py",
    "tests/test_metrics.py",
    "tests/test_fabsa_loader.py",
)
GUIDES = (
    "README.md",
    "CONTRIBUTING.md",
    "docs/README.md",
    "docs/methods.md",
    "docs/results.md",
    "docs/architecture.md",
    "docs/reproducibility.md",
    "docs/history.md",
    "docs/results/training_policy_completed/README.md",
)


def broken_links(root: Path = ROOT) -> list[str]:
    """Check local file targets in maintained Markdown, without network access."""
    failures = []
    for name in GUIDES:
        path = root / name
        if not path.is_file():
            failures.append(f"Missing guide: {name}")
            continue
        for target in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            if not (path.parent / unquote(target)).exists():
                failures.append(f"{name}: missing target {target}")
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", action="store_true", help="Also run the offline CPU smoke suite")
    args = parser.parse_args(argv)
    commands = [
        [sys.executable, "-m", "ruff", "check", "src", "scripts", "tests"],
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--select",
            "E,F,I,B",
            "--ignore",
            "E501",
            *MAINTAINED,
        ],
        [sys.executable, "-m", "ruff", "format", "--check", *MAINTAINED],
    ]
    if args.test:
        commands.append([sys.executable, "-m", "pytest", "-q", *CPU_TESTS])
    for command in commands:
        print("Running:", " ".join(command[1:]), flush=True)
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode:
            return result.returncode
    failures = broken_links()
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print(f"Local links checked in {len(GUIDES)} maintained guides.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
