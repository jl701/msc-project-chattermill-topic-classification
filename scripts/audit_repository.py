"""Audit public-candidate files without printing any matched value.

Inside a Git checkout, the scan follows the same boundary as a commit: tracked files
plus unignored untracked files.  Ignored author-only material, datasets, weights and
checkpoints are therefore not read.  A non-Git fixture falls back to the maintained
source/configuration/documentation folders so the detector remains unit-testable.

This is a bounded release check, not a security certification or a Git-history scan.
It never imports project modules or executes experiments.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECRET_PATTERNS = {
    "private_key_block": re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    "openai_key": re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{40,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "slack_token": re.compile(r"\bxox[baprs]-[0-9A-Za-z-]{20,}\b"),
}
PUBLICATION_PATTERNS = {
    "author_machine_path": re.compile(r"(?i)\bC:[\\/]Users[\\/][^\\/\s]+"),
    "live_ssh_endpoint": re.compile(
        r"\b(?:root|ubuntu)@(?!(?:192\.0\.2|198\.51\.100|203\.0\.113)\.)"
        r"(?:\d{1,3}\.){3}\d{1,3}(?::\d{2,5})?\b"
    ),
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".toml",
    ".yml",
    ".yaml",
    ".sh",
    ".ps1",
    ".cfg",
    ".ini",
    ".txt",
    ".cff",
}


def public_candidate_files(root: Path) -> list[Path]:
    """Return files Git would expose in the next commit, without reading ignored files."""
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            check=True,
            capture_output=True,
        )
        return [root / item.decode() for item in result.stdout.split(b"\0") if item]

    paths: list[Path] = []
    for folder in ("src", "scripts", "tests", "configs", "docs", ".github"):
        if (root / folder).exists():
            paths.extend(path for path in (root / folder).rglob("*") if path.is_file())
    return sorted(paths)


def scan(root: Path) -> dict:
    modules, credential_candidates, publication_candidates = [], [], []
    scanned_files = 0
    for path in public_candidate_files(root):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        scanned_files += 1
        text = path.read_text(encoding="utf-8-sig")
        relative = path.relative_to(root).as_posix()
        for group, destination in (
            (SECRET_PATTERNS, credential_candidates),
            (PUBLICATION_PATTERNS, publication_candidates),
        ):
            for kind, pattern in group.items():
                for match in pattern.finditer(text):
                    destination.append(
                        {
                            "file": relative,
                            "line": text.count("\n", 0, match.start()) + 1,
                            "pattern": kind,
                        }
                    )
        if path.suffix.lower() != ".py":
            continue
        tree = ast.parse(text, filename=relative)
        functions = [
            n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        modules.append(
            {
                "file": relative,
                "lines": len(text.splitlines()),
                "functions": len(functions),
                "largest_function_lines": max(
                    (n.end_lineno - n.lineno + 1 for n in functions), default=0
                ),
                "bare_except": [
                    n.lineno
                    for n in ast.walk(tree)
                    if isinstance(n, ast.ExceptHandler) and n.type is None
                ],
                "dynamic_import_calls": sum(
                    isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Attribute)
                    and n.func.attr in {"spec_from_file_location", "exec_module"}
                    for n in ast.walk(tree)
                ),
            }
        )
    return {
        "scope": "Tracked and unignored public-candidate text files; no ignored files or Git history",
        "scanned_text_files": scanned_files,
        "python_files": len(modules),
        "modules": modules,
        "credential_pattern_candidates": credential_candidates,
        "publication_pattern_candidates": publication_candidates,
        "interpretation": (
            "Complexity is a review signal, not a defect. Pattern matches require inspection; "
            "matched values are deliberately omitted."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON inventory file")
    args = parser.parse_args()
    report = scan(ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    print(f"Inventoried {report['python_files']} Python files.")
    print(
        f"Credential-pattern candidates: {len(report['credential_pattern_candidates'])}. Values are not emitted."
    )
    print(f"Publication-boundary candidates: {len(report['publication_pattern_candidates'])}.")
    return int(
        bool(report["credential_pattern_candidates"] or report["publication_pattern_candidates"])
    )


if __name__ == "__main__":
    raise SystemExit(main())
