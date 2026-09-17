from __future__ import annotations

import importlib.util
from pathlib import Path

from msc_project.experiments.taxonomy_post_supervisor import post_supervisor_l2_folds


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = PROJECT_ROOT / "scripts" / "run_taxonomy_level2_local_validation.py"
    spec = importlib.util.spec_from_file_location("run_taxonomy_level2_local_validation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_level2_contract_is_exactly_three_methods_and_twelve_ndr_folds() -> None:
    module = _load_script()
    assert module.METHODS == (
        "strict_train_only_tfidf",
        "e5_base_v2",
        "frozen_qwen_candidate_pair",
    )
    folds = post_supervisor_l2_folds()
    assert len(folds) == 12
    assert all(fold.conditions == ("N", "D", "R") for fold in folds)


def test_variant_translation_matches_two_stage_runtime_names() -> None:
    module = _load_script()
    assert module.VARIANT_MAP == {
        "name_only": "name_only",
        "minimal": "name_and_description",
        "rich": "rich",
    }
