"""Run the train-only pseudo-unseen rich-description interface study."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from msc_project.baselines.candidate_similarity import (  # noqa: E402
    FrozenTransformerSentenceEncoder,
    REGISTERED_CONFIGS,
)
from msc_project.data.fabsa import default_data_dir  # noqa: E402
from msc_project.data.splits import load_official_fabsa_splits  # noqa: E402
from msc_project.experiments.taxonomy_resources import (  # noqa: E402
    load_description_bundle,
)
from msc_project.experiments.taxonomy_rich_description import (  # noqa: E402
    INTERFACES,
    TfidfFieldSimilarity,
    all_field_texts,
    audit_rich_resource,
    canonical_json_sha256,
    development_partition,
    presence_metrics,
    rich_aspect_fields,
    score_interfaces,
    select_f1_threshold,
    select_global_rich_interface,
)


METHODS = ("tfidf_field_similarity", "e5_base_v2_field_similarity")
PROTOCOL_ID = "taxonomy_rich_description_interface_study_v1"
EXPECTED_TRAIN_SHA256 = "3ffcc7407cc8077fe9efc5ca06532847589a59464ea929d7f23d1f732b9efc7c"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _targets(rows: pd.DataFrame, aspect: str) -> np.ndarray:
    return np.asarray(
        [int(any(str(label_aspect) == aspect for label_aspect, _ in labels)) for labels in rows["labels"]],
        dtype=int,
    )


def _contains_aspect(labels: Sequence[tuple[str, str]], aspect: str) -> bool:
    return any(str(label_aspect) == aspect for label_aspect, _ in labels)


class E5FieldSimilarity:
    method_id = "e5_base_v2_field_similarity"

    def __init__(self, rows: pd.DataFrame, *, local_files_only: bool) -> None:
        ordered = rows.assign(_uid=rows["row_uid"].astype(str)).sort_values(
            "_uid", kind="stable"
        )
        self.encoder = FrozenTransformerSentenceEncoder(
            REGISTERED_CONFIGS["e5_base_v2"],
            device="auto",
            local_files_only=local_files_only,
        )
        self.row_index = {
            str(uid): index for index, uid in enumerate(ordered["row_uid"])
        }
        self.review_embeddings = self.encoder.encode(
            ordered["text"].astype(str).tolist(), role="review"
        )
        self.candidate_embeddings: dict[str, np.ndarray] = {}

    def score_rows(self, rows: pd.DataFrame, candidate_text: str) -> np.ndarray:
        text = str(candidate_text)
        if text not in self.candidate_embeddings:
            self.candidate_embeddings[text] = self.encoder.encode(
                [text], role="candidate"
            )[0]
        candidate = self.candidate_embeddings[text]
        similarities = np.asarray(
            [
                float(self.review_embeddings[self.row_index[str(uid)]] @ candidate)
                for uid in rows["row_uid"]
            ],
            dtype=float,
        )
        values = (np.clip(similarities, -1.0, 1.0) + 1.0) / 2.0
        if not np.isfinite(values).all():
            raise ValueError("E5 field scores must be finite.")
        return values

    def close(self) -> None:
        self.encoder.close()


def _seen_threshold(
    evaluation: pd.DataFrame,
    seen_aspects: Sequence[str],
    resource: dict[str, object],
    score_text: Any,
) -> tuple[dict[str, float | int], str]:
    targets: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for aspect in seen_aspects:
        targets.append(_targets(evaluation, aspect))
        scores.append(score_text(evaluation, rich_aspect_fields(aspect, resource).base))
    target_values = np.concatenate(targets)
    score_values = np.concatenate(scores)
    selection = select_f1_threshold(target_values, score_values)
    digest = canonical_json_sha256(
        {
            "row_uids": evaluation["row_uid"].astype(str).tolist(),
            "seen_aspects": list(seen_aspects),
            "targets": target_values.tolist(),
            "scores": score_values.tolist(),
        }
    )
    return selection, digest


def _run_record(
    *,
    method: str,
    pseudo_unseen: str,
    partition: int,
    fit: pd.DataFrame,
    evaluation: pd.DataFrame,
    resource: dict[str, object],
    e5: E5FieldSimilarity | None,
) -> dict[str, object]:
    canonical = tuple(str(value) for value in resource["canonical_order"])
    seen = tuple(value for value in canonical if value != pseudo_unseen)
    if method == "tfidf_field_similarity":
        scorer = TfidfFieldSimilarity().fit(
            fit["text"].astype(str).tolist(),
            all_field_texts(seen, resource),
        )

        def score_text(rows: pd.DataFrame, value: str) -> np.ndarray:
            return scorer.score(rows["text"].astype(str).tolist(), value)

        fit_descriptor_sha256 = scorer.fit_candidate_text_sha256
    elif method == "e5_base_v2_field_similarity":
        if e5 is None:
            raise AssertionError("The E5 field cache was not initialised.")

        def score_text(rows: pd.DataFrame, value: str) -> np.ndarray:
            return e5.score_rows(rows, value)

        fit_descriptor_sha256 = canonical_json_sha256(all_field_texts(seen, resource))
    else:
        raise ValueError(f"Unknown development method: {method!r}.")

    selection, seen_score_sha256 = _seen_threshold(
        evaluation, seen, resource, score_text
    )
    fields = rich_aspect_fields(pseudo_unseen, resource)
    interface_scores = score_interfaces(
        evaluation["text"].astype(str).tolist(),
        fields,
        lambda reviews, candidate: score_text(evaluation, candidate),
    )
    target = _targets(evaluation, pseudo_unseen)
    if set(np.unique(target)) != {0, 1}:
        raise ValueError(
            f"Pseudo-unseen evaluation lacks both classes: {pseudo_unseen}, partition={partition}."
        )
    records = []
    for interface in INTERFACES:
        metrics = presence_metrics(
            target,
            interface_scores[interface],
            threshold=float(selection["threshold"]),
        )
        records.append(
            {
                "method_id": method,
                "pseudo_unseen_aspect": pseudo_unseen,
                "row_partition": partition,
                "interface": interface,
                **metrics,
            }
        )
    return {
        "method_id": method,
        "pseudo_unseen_aspect": pseudo_unseen,
        "row_partition": partition,
        "fit_rows": int(len(fit)),
        "evaluation_rows": int(len(evaluation)),
        "seen_aspects": list(seen),
        "fit_descriptor_sha256": fit_descriptor_sha256,
        "pseudo_unseen_descriptor_sha256": canonical_json_sha256(
            [
                fields.base,
                *fields.aliases,
                fields.inclusion,
                fields.exclusion,
                fields.positive_concat,
                fields.full_concat,
            ]
        ),
        "seen_score_sha256": seen_score_sha256,
        "threshold_selection": selection,
        "records": records,
        "failure_count": 0,
        "non_finite_value_count": 0,
        "test_contract_count": 0,
    }


def _aggregate(records: list[dict[str, object]]) -> dict[str, object]:
    frame = pd.DataFrame.from_records(records)
    metrics = (
        "average_precision",
        "f1",
        "precision",
        "recall",
        "false_positive_rows_per_100",
        "score_separation",
    )
    means = (
        frame.groupby(["method_id", "interface"], sort=False)[list(metrics)]
        .mean()
        .reset_index()
    )
    per_aspect = (
        frame.groupby(
            ["method_id", "pseudo_unseen_aspect", "interface"], sort=False
        )[list(metrics)]
        .mean()
        .reset_index()
    )
    baseline = per_aspect[per_aspect["interface"].eq("D")][
        ["method_id", "pseudo_unseen_aspect", "average_precision"]
    ].rename(columns={"average_precision": "D_average_precision"})
    deltas = per_aspect.merge(
        baseline,
        on=["method_id", "pseudo_unseen_aspect"],
        how="left",
        validate="many_to_one",
    )
    deltas["average_precision_delta_vs_D"] = (
        deltas["average_precision"] - deltas["D_average_precision"]
    )
    stability = (
        deltas[deltas["interface"].ne("D")]
        .groupby(["method_id", "interface"], sort=False)[
            "average_precision_delta_vs_D"
        ]
        .agg(
            mean="mean",
            improved_aspects=lambda value: int((value > 0).sum()),
            tied_aspects=lambda value: int((value == 0).sum()),
            worsened_aspects=lambda value: int((value < 0).sum()),
        )
        .reset_index()
    )
    return {
        "condition_means": means.to_dict(orient="records"),
        "per_aspect_means": per_aspect.to_dict(orient="records"),
        "stability": stability.to_dict(orient="records"),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    methods = METHODS if args.method == "all" else (args.method,)
    data_dir = args.data_dir.resolve()
    observed_train_sha256 = _file_sha256(data_dir / "train.csv")
    if observed_train_sha256 != EXPECTED_TRAIN_SHA256:
        raise ValueError(
            "Training data hash mismatch: "
            f"expected={EXPECTED_TRAIN_SHA256}, observed={observed_train_sha256}."
        )
    frame = load_official_fabsa_splits(data_dir, ("train",))
    if set(frame["original_split"].astype(str)) != {"train"}:
        raise AssertionError("Rich development opened a non-training split.")
    frame["_development_partition"] = frame["row_uid"].astype(str).map(
        development_partition
    )
    resource = load_description_bundle(require_approved=True)
    canonical = tuple(str(value) for value in resource["canonical_order"])
    selected_aspects = canonical
    if args.pseudo_unseen_aspect:
        if args.pseudo_unseen_aspect not in canonical:
            raise ValueError("Unknown pseudo-unseen aspect.")
        selected_aspects = (args.pseudo_unseen_aspect,)
    selected_partitions = tuple(range(3))
    if args.row_partition is not None:
        if args.row_partition not in selected_partitions:
            raise ValueError("row-partition must be 0, 1 or 2.")
        selected_partitions = (args.row_partition,)

    output_root = args.output_root.resolve()
    _write_json(output_root / "resource_audit.json", audit_rich_resource(resource))
    all_records: list[dict[str, object]] = []
    completed: list[dict[str, object]] = []
    e5 = (
        E5FieldSimilarity(frame, local_files_only=args.local_files_only)
        if "e5_base_v2_field_similarity" in methods
        else None
    )
    try:
        for method in methods:
            for pseudo_unseen in selected_aspects:
                for partition in selected_partitions:
                    result_path = (
                        output_root
                        / method
                        / f"partition-{partition}"
                        / f"{canonical.index(pseudo_unseen) + 1:02d}.json"
                    )
                    if args.resume and result_path.is_file():
                        value = json.loads(result_path.read_text(encoding="utf-8"))
                        if (
                            value.get("protocol_id") != PROTOCOL_ID
                            or value.get("method_id") != method
                            or value.get("pseudo_unseen_aspect") != pseudo_unseen
                            or value.get("row_partition") != partition
                            or value.get("failure_count") != 0
                            or value.get("test_contract_count") != 0
                        ):
                            raise ValueError(f"Rich-study resume conflict: {result_path}.")
                        completed.append(value)
                        all_records.extend(value["records"])
                        print(
                            f"RESUME {method} partition={partition} aspect={pseudo_unseen}",
                            flush=True,
                        )
                        continue
                    evaluation = frame[
                        frame["_development_partition"].eq(partition)
                    ].copy()
                    fit = frame[
                        ~frame["_development_partition"].eq(partition)
                        & ~frame["labels"].map(
                            lambda labels: _contains_aspect(labels, pseudo_unseen)
                        )
                    ].copy()
                    if fit.empty or evaluation.empty:
                        raise ValueError("A rich development fit or evaluation split is empty.")
                    value = _run_record(
                        method=method,
                        pseudo_unseen=pseudo_unseen,
                        partition=partition,
                        fit=fit,
                        evaluation=evaluation,
                        resource=resource,
                        e5=e5,
                    )
                    value.update(
                        {
                            "schema_version": "taxonomy_rich_description_pseudo_fold_v1",
                            "protocol_id": PROTOCOL_ID,
                            "train_csv_sha256": observed_train_sha256,
                            "description_bundle_sha256": resource["content_sha256"],
                        }
                    )
                    _write_json(result_path, value)
                    completed.append(value)
                    all_records.extend(value["records"])
                    print(
                        f"COMPLETE {method} partition={partition} aspect={pseudo_unseen}",
                        flush=True,
                    )
    finally:
        if e5 is not None:
            e5.close()

    expected_keys = len(methods) * len(selected_aspects) * len(selected_partitions)
    if len(completed) != expected_keys:
        raise AssertionError("Rich study did not complete the requested pseudo-fold grid.")
    summary: dict[str, object] = {
        "schema_version": "taxonomy_rich_description_interface_summary_v1",
        "protocol_id": PROTOCOL_ID,
        "methods": list(methods),
        "pseudo_unseen_aspects": list(selected_aspects),
        "row_partitions": list(selected_partitions),
        "completed_pseudo_folds": len(completed),
        "expected_pseudo_folds": expected_keys,
        "records": all_records,
        "aggregate": _aggregate(all_records),
        "failure_count": 0,
        "non_finite_value_count": 0,
        "resume_conflict_count": 0,
        "test_contract_count": 0,
        "seconds": float(time.perf_counter() - started),
    }
    if set(methods) == set(METHODS) and set(selected_aspects) == set(canonical) and set(selected_partitions) == set(range(3)):
        summary["selection"] = select_global_rich_interface(all_records)
    summary["summary_sha256"] = canonical_json_sha256(
        {key: value for key, value in summary.items() if key != "seconds"}
    )
    _write_json(output_root / "summary.json", summary)
    print(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("all", *METHODS), default="all")
    parser.add_argument("--pseudo-unseen-aspect")
    parser.add_argument("--row-partition", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "experimental"
        / "taxonomy_rich_description_interface_study_v1",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

