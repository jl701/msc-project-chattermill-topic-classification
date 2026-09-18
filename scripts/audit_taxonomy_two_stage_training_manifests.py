from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.splits import load_official_fabsa_splits  # noqa: E402
from msc_project.experiments.taxonomy_protocol import (  # noqa: E402
    build_taxonomy_fold_splits,
    registered_folds,
    training_scope_id,
)
from msc_project.experiments.taxonomy_resources import (  # noqa: E402
    load_description_bundle,
)
from msc_project.experiments.taxonomy_two_stage_runtime import (  # noqa: E402
    select_qwen_two_stage_demonstrations,
)
from msc_project.experiments.taxonomy_two_stage_training import (  # noqa: E402
    build_two_stage_training_manifests,
    training_manifest_summary,
)
from msc_project.llm.qwen_two_stage_classifier import (  # noqa: E402
    demonstrations_sha256,
)


DEFAULT_OUTPUT = Path(
    "outputs/experimental/taxonomy_two_stage_precloud_v2/"
    "training_manifest_audit.json"
)


def _sha256_values(values: list[str]) -> str:
    payload = json.dumps(
        sorted(values), ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _representative_folds() -> dict[str, object]:
    representatives: dict[str, object] = {}
    for level in ("L1", "L2", "L3", "L4"):
        for fold in registered_folds(level):
            representatives.setdefault(training_scope_id(fold), fold)
    return representatives


def audit_training_scopes(
    frame: pd.DataFrame,
    resource: dict[str, object],
    *,
    total_budget: int = 4096,
    aspect_budget: int = 2048,
    seed: int = 13,
) -> dict[str, object]:
    if frame.empty or set(frame["original_split"].astype(str)) != {"train"}:
        raise ValueError("Training-scope audit accepts the official train split only.")
    representatives = _representative_folds()
    scopes: list[dict[str, object]] = []
    failures: list[str] = []
    for scope_id, fold in representatives.items():
        split = build_taxonomy_fold_splits(
            frame, fold, evaluation_splits=()
        )["train"]
        first = build_two_stage_training_manifests(
            split,
            fold.seen_aspects,
            resource,
            total_budget=total_budget,
            aspect_budget=aspect_budget,
            seed=seed,
        )
        second = build_two_stage_training_manifests(
            split.sample(frac=1.0, random_state=seed),
            fold.seen_aspects,
            resource,
            total_budget=total_budget,
            aspect_budget=aspect_budget,
            seed=seed,
        )
        first_summaries = {
            task: training_manifest_summary(manifest)
            for task, manifest in first.items()
        }
        second_summaries = {
            task: training_manifest_summary(manifest)
            for task, manifest in second.items()
        }
        reproducible = all(
            first_summaries[task]["manifest_sha256"]
            == second_summaries[task]["manifest_sha256"]
            for task in first
        )
        heldout = set(fold.heldout_aspects)
        manifest_aspects = {
            str(value)
            for manifest in first.values()
            for value in manifest["candidate_aspect"].astype(str)
        }
        source_aspects = {
            str(aspect)
            for labels in split["supervision_labels"]
            for aspect, _ in labels
        }
        demonstrations = select_qwen_two_stage_demonstrations(
            split, fold.seen_aspects, resource, seed=seed
        )
        entry = {
            "training_scope_id": scope_id,
            "representative_fold_id": fold.fold_id,
            "representative_level": fold.level,
            "heldout_aspects": list(fold.heldout_aspects),
            "seen_aspect_count": len(fold.seen_aspects),
            "train_rows_after_example_filter": int(len(split)),
            "train_row_uid_sha256": _sha256_values(
                split["row_uid"].astype(str).tolist()
            ),
            "source_heldout_overlap": sorted(source_aspects & heldout),
            "manifest_heldout_overlap": sorted(manifest_aspects & heldout),
            "manifest_reproducible_after_input_shuffle": reproducible,
            "manifests": first_summaries,
            "few_shot_demonstrations": {
                task: {
                    "count": len(values),
                    "answers": [value.answer for value in values],
                    "sha256": demonstrations_sha256(values),
                    "heldout_overlap": sorted(
                        {
                            value.candidate_aspect
                            for value in values
                        }
                        & heldout
                    ),
                }
                for task, values in demonstrations.items()
            },
        }
        if (
            entry["source_heldout_overlap"]
            or entry["manifest_heldout_overlap"]
            or not reproducible
            or any(
                value["heldout_overlap"]
                for value in entry["few_shot_demonstrations"].values()
            )
        ):
            failures.append(scope_id)
        scopes.append(entry)
    return {
        "audit_id": "taxonomy_two_stage_training_manifest_audit_v1",
        "status": "pass" if not failures else "fail",
        "official_splits_opened": ["train"],
        "test_contract_count": 0,
        "seed": seed,
        "total_budget": total_budget,
        "aspect_budget": aspect_budget,
        "sentiment_budget": total_budget - aspect_budget,
        "expected_unique_training_scopes": 26,
        "observed_unique_training_scopes": len(scopes),
        "failed_scopes": failures,
        "scopes": scopes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit train-only true-two-stage manifests without opening test."
    )
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--total-budget", type=int, default=4096)
    parser.add_argument("--aspect-budget", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = load_official_fabsa_splits(args.data_dir, ("train",))
    resource = load_description_bundle(require_approved=False)
    audit = audit_training_scopes(
        frame,
        resource,
        total_budget=args.total_budget,
        aspect_budget=args.aspect_budget,
        seed=args.seed,
    )
    audit["completed_at"] = datetime.now(timezone.utc).isoformat()
    if audit["observed_unique_training_scopes"] != audit[
        "expected_unique_training_scopes"
    ]:
        audit["status"] = "fail"
        audit["failed_scopes"].append("unexpected_scope_count")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: audit[key]
                for key in (
                    "status",
                    "observed_unique_training_scopes",
                    "failed_scopes",
                    "test_contract_count",
                )
            },
            indent=2,
        )
    )
    return 0 if audit["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
