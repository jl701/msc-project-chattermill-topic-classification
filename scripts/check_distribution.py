"""Reject built archives that contain author-only files or credential-like text."""

from __future__ import annotations

import argparse
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

from audit_repository import PUBLICATION_PATTERNS, SECRET_PATTERNS, TEXT_SUFFIXES

PRIVATE_NAMES = {
    "PROJECT_OVERVIEW.md",
    "START_NEW_CHAT_PROMPT.md",
    "report_notes.md",
    "experiment_log.md",
    "dissertation_internal_spec.md",
    "experiment_reproducibility_register.md",
    "literature_review_matrix.md",
    "next_stage_and_literature_review_plan.md",
    "tasks_1_to_3_thesis_prep.md",
    "thesis_completion_roadmap.md",
}
PRIVATE_PREFIXES = (
    "aji_",
    "kaan_",
    "handoff_notes_",
    "thesis_T0_review_",
    "thesis_final_rewrite_",
    "thesis_literature_gap_",
    "thesis_pro_revision_",
    "thesis_reference_audit_",
    "render_policy_research_report_",
)


def _is_private_name(name: str) -> bool:
    path = PurePosixPath(name)
    parts = path.parts[1:] if len(path.parts) > 1 else path.parts
    return (
        "thesis" in parts
        or "deliverables" in parts
        or path.name in PRIVATE_NAMES
        or path.name.startswith(PRIVATE_PREFIXES)
    )


def audit_archive(path: Path) -> list[dict[str, str]]:
    """Return finding locations and types, never the matched content."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            records = [
                (name, archive.read(name)) for name in archive.namelist() if not name.endswith("/")
            ]
    elif tarfile.is_tarfile(path):
        with tarfile.open(path) as archive:
            records = []
            for member in archive.getmembers():
                if member.isfile():
                    stream = archive.extractfile(member)
                    if stream is not None:
                        records.append((member.name, stream.read()))
    else:
        raise ValueError(f"Unsupported archive: {path.name}")

    findings: list[dict[str, str]] = []
    for name, payload in records:
        if _is_private_name(name):
            findings.append({"file": name, "pattern": "author_only_path"})
            continue
        if PurePosixPath(name).suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = payload.decode("utf-8-sig", errors="replace")
        for group in (SECRET_PATTERNS, PUBLICATION_PATTERNS):
            for kind, pattern in group.items():
                if pattern.search(text):
                    findings.append({"file": name, "pattern": kind})
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Directory containing built archives")
    args = parser.parse_args()
    archives = sorted(path for path in args.directory.iterdir() if path.is_file())
    if not archives:
        parser.error("No distribution archives found")
    findings = [
        {"archive": archive.name, **finding}
        for archive in archives
        for finding in audit_archive(archive)
    ]
    print(f"Audited {len(archives)} distribution archives; findings: {len(findings)}.")
    for finding in findings:
        print(f"{finding['archive']}: {finding['file']} ({finding['pattern']})")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
