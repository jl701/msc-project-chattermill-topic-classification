from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import all_aspects, build_heldout_aspect_split, load_all_fabsa
from msc_project.experiments.duplicate_text_sensitivity import (
    ANALYSIS_CLASSIFICATION,
    ANALYSIS_PROTOCOL_ID,
    FoldSensitivity,
    analyse_model_fold,
    assert_experimental_output_dir,
    compare_complete_models,
    slugify,
    validate_run_root,
)


def _write_json(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() or None if completed.returncode == 0 else None


def _flat_result_rows(results: list[FoldSensitivity]) -> pd.DataFrame:
    return pd.DataFrame.from_records([item.result for item in results])


def _analyse_model(
    *,
    model_name: str,
    aspects: list[str],
    frame: pd.DataFrame,
    full_root: Path,
    pilot_root: Path,
) -> list[FoldSensitivity]:
    values: list[FoldSensitivity] = []
    for aspect in aspects:
        splits = build_heldout_aspect_split(
            frame,
            [aspect],
            strategy="example_filtered",
            eval_label_scope="heldout",
            eval_row_scope="all",
        )
        values.append(
            analyse_model_fold(
                model_name=model_name,
                heldout_aspect=aspect,
                train_frame=splits["train"],
                validation_frame=splits["validation"],
                test_frame=splits["test"],
                full_root=full_root,
                pilot_root=pilot_root,
            )
        )
    return values


def _write_model_artifacts(
    output_dir: Path,
    model_name: str,
    values: list[FoldSensitivity],
) -> None:
    model_root = output_dir / model_name
    model_root.mkdir(parents=True, exist_ok=False)
    _flat_result_rows(values).to_csv(model_root / "per_fold.csv", index=False)
    for value in values:
        aspect = str(value.result["heldout_aspect"])
        fold_root = model_root / slugify(aspect)
        fold_root.mkdir(parents=True, exist_ok=False)
        value.duplicate_rows.to_csv(fold_root / "duplicate_rows.csv", index=False)
        value.original_threshold_sweep.to_csv(
            fold_root / "validation_threshold_sweep_original.csv", index=False
        )
        value.retained_threshold_sweep.to_csv(
            fold_root / "validation_threshold_sweep_retained.csv", index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc/descriptive duplicate-text sensitivity using saved pair scores only; "
            "this script never loads a model."
        )
    )
    parser.add_argument("--analysis-mode", choices=["frozen-smoke", "full-comparison"], required=True)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--frozen-full-root", type=Path, required=True)
    parser.add_argument("--frozen-pilot-root", type=Path, required=True)
    parser.add_argument("--qlora-full-root", type=Path)
    parser.add_argument("--qlora-pilot-root", type=Path)
    parser.add_argument("--heldout-aspect", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = assert_experimental_output_dir(args.output_dir, PROJECT_ROOT)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty sensitivity output: {output_dir}")

    frame = load_all_fabsa(args.data_dir)
    canonical_aspects = all_aspects(frame)
    if args.analysis_mode == "frozen-smoke":
        if args.qlora_full_root is not None or args.qlora_pilot_root is not None:
            raise ValueError("frozen-smoke forbids QLoRA inputs and cannot produce a comparison.")
        if len(args.heldout_aspect) != 1 or args.heldout_aspect[0] not in canonical_aspects:
            raise ValueError("frozen-smoke requires exactly one canonical --heldout-aspect.")
        aspects = list(args.heldout_aspect)
    else:
        if args.heldout_aspect:
            raise ValueError("full-comparison always uses all canonical folds.")
        if args.qlora_full_root is None or args.qlora_pilot_root is None:
            raise ValueError("full-comparison requires both QLoRA full and pilot roots.")
        if len(canonical_aspects) != 12:
            raise ValueError("full-comparison requires exactly 12 canonical FABSA aspects.")
        aspects = canonical_aspects

    frozen_full_manifest = validate_run_root(
        args.frozen_full_root,
        expected_mode="frozen",
        expected_stage="full_12_fold_confirmation",
        expected_folds=canonical_aspects,
        require_finished=True,
    )
    frozen_pilot_manifest = validate_run_root(
        args.frozen_pilot_root,
        expected_mode="frozen",
        expected_stage="pilot",
        expected_folds=[
            "Company brand: Competitor",
            "Company brand: General satisfaction",
            "Staff support: Email",
        ],
        require_finished=True,
    )
    qlora_full_manifest: dict[str, object] | None = None
    qlora_pilot_manifest: dict[str, object] | None = None
    if args.analysis_mode == "full-comparison":
        assert args.qlora_full_root is not None and args.qlora_pilot_root is not None
        qlora_full_manifest = validate_run_root(
            args.qlora_full_root,
            expected_mode="qlora",
            expected_stage="full_12_fold_confirmation",
            expected_folds=canonical_aspects,
            require_finished=True,
        )
        qlora_pilot_manifest = validate_run_root(
            args.qlora_pilot_root,
            expected_mode="qlora",
            expected_stage="pilot",
            expected_folds=[
                "Company brand: Competitor",
                "Company brand: General satisfaction",
                "Staff support: Email",
            ],
            require_finished=True,
        )
        if qlora_full_manifest.get("dataset_identity") != frozen_full_manifest.get(
            "dataset_identity"
        ):
            raise ValueError("Frozen and QLoRA full-run dataset identities differ.")

    # Complete every read, identity check and calculation before creating output.
    frozen = _analyse_model(
        model_name="frozen",
        aspects=aspects,
        frame=frame,
        full_root=args.frozen_full_root,
        pilot_root=args.frozen_pilot_root,
    )
    qlora: list[FoldSensitivity] | None = None
    comparison: dict[str, object] | None = None
    if args.analysis_mode == "full-comparison":
        assert args.qlora_full_root is not None and args.qlora_pilot_root is not None
        qlora = _analyse_model(
            model_name="qlora",
            aspects=aspects,
            frame=frame,
            full_root=args.qlora_full_root,
            pilot_root=args.qlora_pilot_root,
        )
        comparison = compare_complete_models(frozen, qlora, canonical_aspects)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "analysis_protocol_id": ANALYSIS_PROTOCOL_ID,
        "analysis_classification": ANALYSIS_CLASSIFICATION,
        "analysis_mode": args.analysis_mode,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "command_argv": [sys.executable, *sys.argv],
        "analysis_code_git_commit": _git_commit(),
        "no_retraining": True,
        "no_model_inference": True,
        "not_confirmatory": True,
        "not_for_main_story": True,
        "normalisation": "Unicode NFKC, casefold, whitespace collapse and strip; empty excluded",
        "fold_protocol": "example_filtered; eval_label_scope=heldout; eval_row_scope=all",
        "aspects": aspects,
        "data_dir": str(args.data_dir.resolve()),
        "frozen_full_root": str(args.frozen_full_root.resolve()),
        "frozen_pilot_root": str(args.frozen_pilot_root.resolve()),
        "qlora_full_root": str(args.qlora_full_root.resolve()) if args.qlora_full_root else None,
        "qlora_pilot_root": str(args.qlora_pilot_root.resolve()) if args.qlora_pilot_root else None,
        "source_run_finished_at": {
            "frozen_full": frozen_full_manifest.get("finished_at"),
            "frozen_pilot": frozen_pilot_manifest.get("finished_at"),
            "qlora_full": qlora_full_manifest.get("finished_at") if qlora_full_manifest else None,
            "qlora_pilot": qlora_pilot_manifest.get("finished_at") if qlora_pilot_manifest else None,
        },
    }
    _write_json(manifest, output_dir / "analysis_manifest.json")
    _write_model_artifacts(output_dir, "frozen", frozen)
    summary: dict[str, object] = {
        **manifest,
        "frozen_results": [item.result for item in frozen],
        "comparison": comparison,
    }
    if qlora is not None:
        _write_model_artifacts(output_dir, "qlora", qlora)
        summary["qlora_results"] = [item.result for item in qlora]
        assert comparison is not None
        pd.DataFrame.from_records(comparison["per_fold"]).to_csv(
            output_dir / "comparison_per_fold.csv", index=False
        )
    _write_json(summary, output_dir / "summary.json")
    print(json.dumps({"event": "duplicate_sensitivity_complete", "output_dir": str(output_dir)}))


if __name__ == "__main__":
    main()
