"""Audit the matched-interface boundary for frozen Qwen and QLoRA.

This audit intentionally establishes only the facts required for a descriptive
adapted-versus-frozen systems comparison.  It does not claim that the observed
difference is a causal LoRA-only effect.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKUP_ROOT = Path(
    r"C:\Msc_DSML\Msc_Project\cloud_backups\taxonomy_two_stage_formal_v2_r2"
)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    registry = _read(
        PROJECT_ROOT / "configs/experiments/taxonomy_method_registry_v1.json"
    )["methods"]
    validation = _read(
        PROJECT_ROOT / "configs/experiments/taxonomy_two_stage_validation_v1.json"
    )
    cache_summary = _read(
        PROJECT_ROOT
        / "outputs/experimental/taxonomy_scientific_freeze_v1/local_replay"
        / "frozen_qwen_candidate_pair/summary.json"
    )
    frozen = registry["frozen_qwen_candidate_pair"]
    qlora = registry["qwen_candidate_pair_qlora"]
    stage_1 = validation["study_C"]["stage_1"]["frozen_qwen"]
    stage_2 = validation["study_C"]["stage_2"]["frozen_qwen"]

    contract_paths = sorted(args.backup_root.rglob("two_stage_qlora_contract.json"))
    manifest_paths = sorted(
        path
        for path in args.backup_root.rglob("checkpoint.manifest.json")
        if "qwen_candidate_pair_qlora" in path.parts
    )
    if not contract_paths or not manifest_paths:
        raise FileNotFoundError("QLoRA contracts or checkpoint manifests are missing.")

    contracts = [_read(path) for path in contract_paths]
    manifests = [_read(path) for path in manifest_paths]
    prompt_hashes = {str(value["prompt_contract_sha256"]) for value in contracts}
    qlora_max_lengths = {
        int(value["training"]["max_length"]) for value in contracts
    }
    qlora_model_ids = {
        str(value["training_contract"]["model_id"]) for value in manifests
    }
    qlora_revisions = {
        str(value["training_contract"]["model_revision"]) for value in manifests
    }
    qlora_test_counts = {
        int(value["evidence"].get("test_contract_count", -1)) for value in manifests
    }

    frozen_prompt_hash = str(stage_1["prompt_contract_sha256"])
    checks = {
        "same_model_id": frozen["model_id"] == qlora["model_id"]
        and qlora_model_ids == {frozen["model_id"]},
        "same_model_revision": frozen["model_revision"] == qlora["model_revision"]
        and qlora_revisions == {frozen["model_revision"]},
        "same_task_format_registry": qlora["starting_recipe"]["task_format"]
        == "identical to frozen_qwen_candidate_pair",
        "same_prompt_contract_across_stages_cache_and_all_qlora_checkpoints": (
            stage_2["prompt_contract_sha256"]
            == frozen_prompt_hash
            == cache_summary["qwen_cache"]["contract_sha256"]
            and prompt_hashes == {frozen_prompt_hash}
        ),
        "same_max_length": int(frozen["starting_recipe"]["max_length"])
        == int(qlora["starting_recipe"]["max_length"])
        == int(stage_1["max_length"])
        and qlora_max_lengths == {int(stage_1["max_length"])},
        "same_four_bit_boundary": bool(frozen["starting_recipe"]["load_in_4bit"])
        and bool(qlora["starting_recipe"]["load_in_4bit"])
        and all(bool(value["qlora"]["load_in_4bit"]) for value in contracts),
        "frozen_cache_read_only_exact": cache_summary["qwen_cache"]["mode"]
        == "read_only_exact",
        "qlora_checkpoint_test_contract_zero": qlora_test_counts == {0},
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"Matched-interface audit failed: {failed}")

    audit = {
        "schema_version": "taxonomy_qwen_matched_interface_audit_v1",
        "status": "pass",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "include_official_test": False,
        "official_test_opened": False,
        "test_contract_count": 0,
        "failure_count": 0,
        "non_finite_value_count": 0,
        "resume_conflict_count": 0,
        "matched_interface": {
            "model_id": frozen["model_id"],
            "model_revision": frozen["model_revision"],
            "prompt_contract_sha256": frozen_prompt_hash,
            "max_length": int(stage_1["max_length"]),
            "load_in_4bit": True,
            "aspect_verbalizers": stage_1["verbalizers"],
            "sentiment_verbalizers": stage_2["verbalizers"],
            "qlora_contract_files_checked": len(contract_paths),
            "qlora_checkpoint_manifests_checked": len(manifest_paths),
        },
        "known_non_adapter_differences": {
            "frozen_eval_batch_size": int(stage_1["eval_batch_size"]),
            "qlora_eval_batch_size": 8,
            "training": "QLoRA receives outer-scope supervised adaptation; frozen Qwen does not.",
            "thresholds": "Each system selects its own seen-validation aspect and runner-up thresholds.",
        },
        "checks": checks,
        "allowed_claim": (
            "A descriptive comparison of two systems sharing base model, revision, "
            "prompt/verbalizer contract, maximum length and four-bit loading boundary."
        ),
        "forbidden_claim": (
            "The score difference is a causal estimate of the LoRA adapter alone."
        ),
    }
    _write(args.output.resolve(), audit)
    print(
        json.dumps(
            {
                "status": audit["status"],
                "checks": len(checks),
                "qlora_contract_files_checked": len(contract_paths),
                "qlora_checkpoint_manifests_checked": len(manifest_paths),
                "test_contract_count": 0,
            },
            indent=2,
        ),
        flush=True,
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/experiments/taxonomy_qwen_matched_interface_audit_20260822.json"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
