import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyse_taxonomy_policy_qlora_completion as a
from msc_project.data.fabsa import format_pair_label


def manifest(tmp_path):
    return a.template(tmp_path / "fewshot", tmp_path / "filtered", tmp_path / "masked", tmp_path / "data")


def test_exact_inventory_and_no_partial_headlines(tmp_path):
    value = manifest(tmp_path)
    assert len(value["sources"]) == 48
    status = a.analyse(value, tmp_path / "analysis")
    assert status["status"] == "PENDING"
    assert status["pending_source_folds"] == 48
    assert status["headline_metrics_generated"] is False
    assert not (tmp_path / "analysis").exists()


@pytest.mark.parametrize("change", ["duplicate", "missing", "test", "draws", "composition"])
def test_manifest_rejects_contract_changes(tmp_path, change):
    value = manifest(tmp_path)
    if change == "duplicate":
        value["sources"][-1] = value["sources"][0]
    elif change == "missing":
        value["sources"].pop()
    elif change == "test":
        value["allowed_splits"] = ["validation", "test"]
    elif change == "draws":
        value["bootstrap"]["draws"] = 100
    else:
        value["composition"] = "mixed_component_diagnostic"
    with pytest.raises(ValueError):
        a.validate_manifest(value)


def synthetic_grid(reviews=3):
    fold = a.registered_folds("L2")[0]
    rows = []
    for i in range(reviews):
        for j, aspect in enumerate(fold.evaluation_aspects):
            for s, sentiment in enumerate(("negative", "neutral", "positive")):
                rows.append((f"validation:{i}", aspect, sentiment, int(i == j % 3 and s == 2),
                             .8 if i == j % 3 else .1, (.1, .2, .7)[s]))
    grid = pd.DataFrame(rows, columns=a.KEY + ["target", "aspect_score", "sentiment_score"])
    grid["pair_label"] = [format_pair_label(x, y) for x, y in zip(grid.candidate_aspect, grid.candidate_sentiment)]
    grid["representation_variant"] = "name_only"
    grid = grid.sort_values(a.KEY).reset_index(drop=True)
    return fold, grid


def test_compose_keeps_own_gate_and_decoder():
    _, first = synthetic_grid()
    second = first.copy()
    second["aspect_score"] = .3
    second["sentiment_score"] = first.sentiment_score / 2
    combined = a.compose(first, second, a.POLICIES[0], a.POLICIES[0])
    assert np.array_equal(combined.aspect_score, first.aspect_score)
    assert np.array_equal(combined.sentiment_score, second.sentiment_score)
    with pytest.raises(ValueError):
        a.compose(first, second, a.POLICIES[0], a.POLICIES[1])
    with pytest.raises(ValueError):
        a.compose(first, second.iloc[::-1], a.POLICIES[0], a.POLICIES[0])


def test_grid_rejects_missing_target_nonfinite_and_wrong_partition():
    fold, grid = synthetic_grid()
    expected = grid[a.KEY + ["target"]]
    assert len(a.validate_grid(grid, expected, fold, "N")) == len(expected)
    for bad in (grid.iloc[1:].copy(), grid.assign(aspect_score=np.nan), grid.assign(split="test"), grid.assign(target=1)):
        with pytest.raises(ValueError):
            a.validate_grid(bad, expected, fold, "N")


def test_sealed_payload_detects_mutation():
    value = {"failure_count": 0, "test_contract_count": 0}
    value["seal"] = a.canonical_sha256(value)
    a.sealed(value, "seal")
    value["failure_count"] = 1
    with pytest.raises(ValueError):
        a.sealed(value, "seal")


def test_fixed_contrast_family_has_no_cross_policy_component():
    terms = a.comparison_terms()
    assert len(terms) == 18
    assert len({name for name, _ in terms}) == 18
    assert all(sum(coefficient for _, coefficient in values) == 0 for _, values in terms)
    assert not any("mixed" in name or "reverse" in name for name, _ in terms)


def test_bootstrap_identical_systems_have_exact_zero_difference():
    base = np.array([[[1, 0, 0], [0, 1, 1]], [[1, 1, 0], [1, 0, 0]]])
    draws, pooled = a.bootstrap(np.stack([base, base]), draws=201, seed=13)
    assert np.array_equal(draws[:, 0], draws[:, 1])
    assert np.array_equal(pooled[:, 0], pooled[:, 1])


def test_thresholds_not_reselected():
    expected = {"aspect_threshold": .4, "second_sentiment_threshold": .99}
    assert a.thresholds(expected) == expected
    assert a.thresholds({"selected_thresholds": {"aspect": .4, "runner_up_sentiment": .99}}, formal=True) == expected
    with pytest.raises(ValueError):
        a.thresholds({**expected, "aspect_threshold": float("nan")})


def test_diagnostics_reconstruct_counts_and_cap_two():
    fold, grid = synthetic_grid()
    selected = {"aspect_threshold": .5, "second_sentiment_threshold": .15}
    row, counts = a.summarise_grid(grid, fold, selected)
    assert counts.shape == (3, 3)
    assert row["two_sentiment_instances"] == row["selected_review_aspects"]
    assert row["heldout_pair_micro_f1"] == pytest.approx(a.f1(counts.sum(axis=0).to_numpy()))


def test_pilot_pending_is_not_complete_study(tmp_path):
    result = a.audit_pilot(manifest(tmp_path))
    assert result["status"] == "PENDING"
    assert result["headline_metrics_generated"] is False
    assert result["scope"] == "masked_a01_pilot_only"


def test_formal_shard_loader_and_seen_invariance(tmp_path, monkeypatch):
    from msc_project.experiments.taxonomy_execution import select_pair_shard
    from msc_project.experiments.taxonomy_two_stage_artifacts import build_two_stage_score_artifact, write_two_stage_score_shard
    fold, grid = synthetic_grid(reviews=64)
    grid["fold_id"] = fold.fold_id
    grid["row_index"] = grid.row_uid.map({uid: i for i, uid in enumerate(sorted(grid.row_uid.unique()))})
    grid["is_seen"] = grid.candidate_aspect.isin(fold.seen_aspects)
    grid["is_heldout"] = ~grid.is_seen
    expected = grid[a.KEY + ["target"]]
    source = {"method": "qlora", "policy": a.POLICIES[1], "fold_id": fold.fold_id, "root": str(tmp_path)}
    a.write_json(tmp_path / "policy_provenance/run_identity.json", {"study_id": a.STUDY, "policy": source["policy"], "canonical_seen_policy": a.CANONICAL})
    chosen = {"aspect": .5, "runner_up_sentiment": .99}
    selection = {"failure_count": 0, "test_contract_count": 0, "training_scope_id": "heldout-a01", "method_id": a.QLORA,
                 "selected_thresholds": chosen, "selected_candidate_contract_sha256": "c" * 64}
    selection["selection_payload_sha256"] = a.canonical_sha256(selection)
    a.write_json(a.selection_file(source), selection)
    # Candidate/checkpoint validation is independently exercised against real frozen
    # selected files. This fixture isolates both-condition score-shard intake.
    monkeypatch.setattr(a, "validate_selection_sources", lambda *args: None)
    monkeypatch.setattr(a, "verify_policy_identity", lambda *args: None)
    for condition in a.CONDITIONS:
        frame = grid.assign(condition=condition)
        contract = a.RunContract(protocol_id=a.STUDY, scientific_protocol_sha256="a"*64, method_id=a.QLORA,
                                method_spec_sha256="a"*64, method_registry_sha256="a"*64, description_resource_sha256="a"*64,
                                candidate_representation_sha256="a"*64, level="L2", fold_id=fold.fold_id, condition=condition,
                                split="validation", seed=13, training_manifest_sha256="a"*64, evaluation_data_sha256="a"*64,
                                evaluation_pair_identity_sha256="a"*64, scientific_parameters_sha256="a"*64, shard_count=8, formal=True)
        artifacts = []
        for i in range(8):
            shard = select_pair_shard(frame, i, 8)
            assert len(shard)
            artifact = build_two_stage_score_artifact(shard, contract, shard_index=i)
            write_two_stage_score_shard(artifact, tmp_path, contract, shard_index=i)
            artifacts.append(artifact)
        merged = pd.concat(artifacts)
        result = {"protocol_id": a.STUDY, "failure_count": 0, "test_contract_count": 0, "evaluation_partition": "validation_only",
                  "selection_partition": "seen_validation_only", "fold_id": fold.fold_id, "condition": condition,
                  "thresholds": chosen, "training_contract_sha256": "c"*64, "run_contract_sha256": contract.contract_sha256,
                  "score_sha256": a.dataframe_sha256(merged, a.KEY + ["aspect_score", "sentiment_score"], sort_columns=a.KEY)}
        result["result_payload_sha256"] = a.canonical_sha256(result)
        a.write_json(tmp_path / "results" / a.QLORA / "L2" / fold.fold_id / f"{condition}.json", result)
    hashes = {}
    loaded, selected = a.load_source(source, expected, fold, hashes)
    assert set(loaded) == {"N", "D"}
    assert selected == {"aspect_threshold": .5, "second_sentiment_threshold": .99}
    assert len(hashes) == 36
    # Both manifest and score bytes remain protected, not just existence counted.
    path = next((tmp_path / "scores").rglob("*.csv"))
    path.write_text(path.read_text() + "corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="CSV hash"):
        a.load_source(source, expected, fold, {})
