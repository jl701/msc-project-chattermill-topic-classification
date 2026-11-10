"""Private seen-only development diagnostics; never reads full/evaluation gold.

Only completed, hash-registered method files are accepted. Without --method the
entire registered matrix must be complete. Partial profiles never choose an
improvement. Outputs include private review text and must remain under outputs/.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msc_project.serving.rag_evaluation import (
    active_presence_threshold, decode_scores, gold_by_id, margin_threshold_candidates,
    presence_space, read_jsonl, records_by_id, score_maps, successful,
    threshold_candidates, validate_record_spaces,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"Existing diagnostic snapshot differs: {path}")
    path.write_text(content, encoding="utf-8")


def write_json(path: Path, value: object) -> None:
    write(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def score_counts(predicted: dict[str, set], gold: dict[str, set]) -> dict:
    pair, presence = Counter(tp=0, fp=0, fn=0), Counter(tp=0, fp=0, fn=0)
    for key, target in gold.items():
        prediction = predicted[key]
        pa, ga = {a for a, _ in prediction}, {a for a, _ in target}
        for counts, actual, expected in ((pair, prediction, target), (presence, pa, ga)):
            counts.update(tp=len(actual & expected), fp=len(actual - expected), fn=len(expected - actual))
    return {"pair": {**pair, "f1": f1(**pair)},
            "presence": {**presence, "f1": f1(**presence)}}


def strip_nonseen(row: dict, seen: set[str]) -> dict:
    """Drop heldout predictions and diagnostics BEFORE calculating any profile."""
    response = row.get("response") or {}
    retrieval = response.get("diagnostics", {}).get("batch_aspect", {}).get("retrieval", [])
    filtered = {"request_id": response.get("request_id", row["request_id"]),
        "presence_score_space": presence_space(response),
        "aspect_scores": [item for item in response.get("aspect_scores", []) if item["aspect"] in seen],
        "full_grid_scores": [item for item in response.get("full_grid_scores", []) if item["aspect"] in seen],
        "retrieval": [item for item in retrieval if item.get("candidate_aspect") in seen],
        "sentiment_truncation_by_aspect": {a: value for a, value in
            response.get("sentiment_truncation_by_aspect", {}).items() if a in seen}}
    return {"request_id": row["request_id"], "status": row.get("status", "ok"),
            "success": successful(row), "response": filtered}


def calibrate_seen(records: dict, gold: dict, aspects: list[str], old: float, runner: float, *,
                   presence_score_space: str = "probability") -> dict:
    maps = {key: score_maps(record, aspects, presence_score_space=presence_score_space)
            for key, record in records.items() if successful(record)}
    generator = margin_threshold_candidates if presence_score_space == "logit_margin" else threshold_candidates
    candidates = generator([p for presence, _ in maps.values() for p in presence.values()], old)
    trials = []
    for threshold in candidates:
        predicted = {key: decode_scores(*maps[key], threshold, runner) if key in maps else set() for key in gold}
        counts = score_counts(predicted, gold)
        calls = sum(len({a for a, _ in pairs}) for pairs in predicted.values())
        trials.append((counts["pair"]["f1"], counts["presence"]["f1"], -calls, threshold))
    best = max(trials)
    result = {"aspect_threshold": best[3], "runner_up_threshold": runner,
            "old_aspect_threshold": old, "candidate_count": len(candidates),
            "seen_pair_f1": best[0], "seen_presence_f1": best[1], "seen_aspect_calls": -best[2],
            "selection_scope": "sanitized_seen_labels_and_seen_scores_only"}
    if presence_score_space == "logit_margin":
        result.update(presence_score_space="logit_margin", presence_margin_threshold=best[3],
                      aspect_threshold=old)
    return result


def decode(records: dict, gold: dict, aspects: list[str], threshold: float, runner: float, *,
           presence_score_space: str = "probability") -> dict:
    return {key: decode_scores(*score_maps(records[key], aspects, presence_score_space=presence_score_space), threshold, runner)
            if successful(records[key]) else set() for key in gold}


def distribution(values: list[float]) -> dict:
    return {"count": len(values), "min": min(values) if values else None,
            "median": float(np.median(values)) if values else None,
            "p95": float(np.quantile(values, .95)) if values else None,
            "max": max(values) if values else None}


def demo_profile(records: dict, predictions: dict, gold: dict) -> dict:
    totals = Counter()
    positive_themes, negative_themes = Counter(), Counter()
    buckets = defaultdict(Counter)
    prompt_tokens, final_tokens = [], []
    for key, row in records.items():
        predicted_aspects, gold_aspects = {a for a, _ in predictions[key]}, {a for a, _ in gold[key]}
        for item in row["response"]["retrieval"]:
            aspect = item["candidate_aspect"]
            ids, answers = item.get("selected_ids", []), item.get("selected_answers", [])
            if len(ids) != len(answers) or any("|" not in identity for identity in ids):
                raise ValueError("Demonstration metadata is incomplete")
            parsed = [identity.rsplit("|", 1) for identity in ids]
            if any(theme not in {score["aspect"] for score in row["response"]["aspect_scores"]}
                   for _, theme in parsed):
                raise ValueError("A demonstration refers to a nonseen aspect")
            aligned_y = sum(theme == aspect and answer == "Y" for (_, theme), answer in zip(parsed, answers))
            aligned_n = sum(theme == aspect and answer == "N" for (_, theme), answer in zip(parsed, answers))
            totals.update(prompts=1, demonstrations=len(ids), aligned_positive=aligned_y,
                          aligned_negative=aligned_n, prompts_with_positive_alignment=aligned_y > 0,
                          prompts_with_negative_alignment=aligned_n > 0,
                          prompts_with_repeated_demo_entry=len(set(ids)) < len(ids),
                          prompts_with_repeated_review=len({uid for uid, _ in parsed}) < len(parsed),
                          non_2Y2N_prompts=answers != list("YYNN"))
            for (_, theme), answer in zip(parsed, answers):
                (positive_themes if answer == "Y" else negative_themes).update([theme])
            bucket = buckets[f"same_theme_positive{aligned_y}_negative{aligned_n}"]
            actual, target = aspect in predicted_aspects, aspect in gold_aspects
            bucket.update(prompts=1, gold_positive=target, gold_negative=not target,
                          fp=actual and not target, fn=target and not actual)
            truncation = item.get("truncation", {})
            if truncation:
                prompt_tokens.append(truncation["original_prompt_tokens"])
                final_tokens.append(truncation["final_prompt_tokens"])
                totals.update(query_trimmed_prompts=truncation.get("query_characters_removed", 0) > 0,
                    severe_query_trimmed_prompts=truncation.get("severe_query_truncation") is True,
                    demo_trimmed_prompts=any(value > 0 for value in truncation.get("demo_characters_removed", [])))
    return {"counts": dict(totals), "positive_demo_themes": dict(positive_themes),
            "negative_demo_themes": dict(negative_themes),
            "alignment_error_buckets": {name: {**counts,
                "false_positive_rate": counts["fp"] / counts["gold_negative"] if counts["gold_negative"] else None,
                "false_negative_rate": counts["fn"] / counts["gold_positive"] if counts["gold_positive"] else None}
                for name, counts in sorted(buckets.items())},
            "original_prompt_tokens": distribution(prompt_tokens), "final_prompt_tokens": distribution(final_tokens),
            "interpretation": "descriptive_alignment_association_not_causal; duplicates_use_review_identity_not_text_similarity"}


def training_conflict_profile(records: dict, predictions: dict, gold: dict,
                              conflict_groups: list[dict], aspects: list[str]) -> tuple[dict, dict]:
    ambiguous, retained = {}, set()
    for group in conflict_groups:
        variants = [set(values) for values in group["seen_aspect_sets"]]
        if not variants or any(values - set(aspects) for values in variants):
            raise ValueError("Training-conflict audit contains an unknown or nonseen aspect")
        changed = set.union(*variants) - set.intersection(*variants)
        for uid in group["retained_rows"]:
            retained.add(uid)
            ambiguous[uid] = changed
    totals = {"unexposed": Counter(), "conflict_group_only": Counter(), "ambiguous_demo_label": Counter()}
    theme_cohorts = defaultdict(lambda: defaultdict(Counter))
    review_cohorts = {"exposed": Counter(), "unexposed": Counter()}
    private_rows = defaultdict(Counter)
    exposed_unique = set()
    for key, record in records.items():
        pa, ga = {a for a, _ in predictions[key]}, {a for a, _ in gold[key]}
        review_exposed = False
        for entry in record["response"]["retrieval"]:
            aspect = entry["candidate_aspect"]
            parsed = [item.rsplit("|", 1) for item in entry["selected_ids"]]
            exposed = {uid for uid, _ in parsed if uid in retained}
            uncertain = {uid for uid, demo_aspect in parsed if demo_aspect in ambiguous.get(uid, set())}
            review_exposed |= bool(exposed)
            exposed_unique.update(exposed)
            category = "ambiguous_demo_label" if uncertain else "conflict_group_only" if exposed else "unexposed"
            actual, target = aspect in pa, aspect in ga
            pset = {s for a, s in predictions[key] if a == aspect}
            gset = {s for a, s in gold[key] if a == aspect}
            counts = {"prompts": 1, "gold_positive": int(target), "gold_negative": int(not target),
                      "presence_fp": int(actual and not target), "presence_fn": int(target and not actual),
                      "presence_errors": int(actual != target), "pair_set_errors": int(pset != gset)}
            totals[category].update(counts)
            theme_cohorts[aspect][category].update(counts)
            for uid in exposed:
                private_rows[uid].update(selected_prompts=1, presence_errors=int(actual != target),
                    pair_set_errors=int(pset != gset), ambiguous_demo_label_prompts=int(uid in uncertain))
        review_cohorts["exposed" if review_exposed else "unexposed"].update(
            reviews=1, seen_set_errors=int(predictions[key] != gold[key]))
    def with_rates(counts):
        return {**{name: counts[name] for name in ("prompts", "gold_positive", "gold_negative",
            "presence_fp", "presence_fn", "presence_errors", "pair_set_errors")},
            "presence_error_rate": counts["presence_errors"] / counts["prompts"] if counts["prompts"] else None,
            "pair_set_error_rate": counts["pair_set_errors"] / counts["prompts"] if counts["prompts"] else None,
            "false_positive_rate": counts["presence_fp"] / counts["gold_negative"] if counts["gold_negative"] else None,
            "false_negative_rate": counts["presence_fn"] / counts["gold_positive"] if counts["gold_positive"] else None}
    return {"retained_conflict_reviews_in_audit": len(retained),
        "unique_conflict_reviews_selected": len(exposed_unique),
        "prompt_cohorts": {name: with_rates(counts) for name, counts in totals.items()},
        "theme_cohorts": {aspect: {name: with_rates(counts) for name, counts in groups.items()}
                          for aspect, groups in theme_cohorts.items()},
        "review_cohorts": {name: {"reviews": counts["reviews"], "seen_set_errors": counts["seen_set_errors"],
                          "seen_set_error_rate": counts["seen_set_errors"] / counts["reviews"]
                          if counts["reviews"] else None} for name, counts in review_cohorts.items()},
        "interpretation": "descriptive_association; themes_and_review_difficulty_confound_exposure; not_a_quarantine_effect_estimate"}, {
        "retained_review_exposures": {uid: {**counts, "ambiguous_seen_aspects": sorted(ambiguous[uid])}
                                      for uid, counts in private_rows.items()}}


def profile_method(records: dict, gold: dict, aspects: list[str], thresholds: dict,
                   text: dict[str, str], max_cases: int) -> tuple[dict, dict, list[dict]]:
    predicted = decode(records, gold, aspects, active_presence_threshold(thresholds), thresholds["runner_up_threshold"],
                       presence_score_space=presence_space(thresholds))
    counts = score_counts(predicted, gold)
    themes = {}
    for aspect in aspects:
        target = {key: {pair for pair in pairs if pair[0] == aspect} for key, pairs in gold.items()}
        actual = {key: {pair for pair in pairs if pair[0] == aspect} for key, pairs in predicted.items()}
        values = score_counts(actual, target)
        values.update(gold_reviews=sum(bool(pairs) for pairs in target.values()),
                      predicted_reviews=sum(bool(pairs) for pairs in actual.values()),
                      sentiment_set_errors_when_both_present=sum(bool(actual[k]) and bool(target[k]) and actual[k] != target[k] for k in target))
        themes[aspect] = values
    diagnostics = Counter()
    cases = []
    for key in gold:
        actual, target = predicted[key], gold[key]
        pa, ga = {a for a, _ in actual}, {a for a, _ in target}
        retr = records[key]["response"]["retrieval"]
        sentiment_trim = records[key]["response"]["sentiment_truncation_by_aspect"]
        selected_trim = sorted(a for a in pa if sentiment_trim.get(a, {}).get("left_truncated") is True)
        any_query_trim = any(item.get("truncation", {}).get("query_characters_removed", 0) > 0 for item in retr)
        severe_query_trim = any(item.get("truncation", {}).get("severe_query_truncation") is True for item in retr)
        diagnostics.update(failed_reviews=not successful(records[key]), seen_empty_outputs=not actual,
            seen_empty_gold_reviews=not target, empty_seen_output_with_nonempty_seen_gold=not actual and bool(target),
            query_trimmed_reviews=any_query_trim, severe_query_trimmed_reviews=severe_query_trim,
            selected_seen_sentiment_truncated_reviews=bool(selected_trim),
            exact_seen_label_set_reviews=actual == target)
        if actual != target or not successful(records[key]):
            affected = {a for a, _ in actual ^ target}
            cases.append({"request_id": key, "text": text[key], "text_characters": len(text[key]),
                "seen_gold": sorted(target), "seen_prediction": sorted(actual),
                "pair_fp": sorted(actual - target), "pair_fn": sorted(target - actual),
                "presence_fp": sorted(pa - ga), "presence_fn": sorted(ga - pa),
                "pair_error_count": len(actual ^ target), "query_trimmed": any_query_trim,
                "severe_query_trimmed": severe_query_trim, "selected_sentiment_truncated": selected_trim,
                "affected_seen_prompts": [{"aspect": item["candidate_aspect"],
                    "selected_ids": item["selected_ids"], "selected_answers": item["selected_answers"],
                    "truncation": item.get("truncation")} for item in retr if item["candidate_aspect"] in affected]})
    cases.sort(key=lambda row: (-row["pair_error_count"], -len(row["presence_fn"]), row["request_id"]))
    summary = {"review_count": len(gold), "thresholds": thresholds, **counts,
               "themes": themes, "diagnostics": dict(diagnostics),
               "demonstrations": demo_profile(records, predicted, gold)}
    return summary, predicted, cases[:max_cases]


def compare_fixed(summary: dict, predictions: dict, fixed_summary: dict, fixed: dict, gold: dict) -> dict:
    return {"seen_pair_f1_difference_points": 100 * (summary["pair"]["f1"] - fixed_summary["pair"]["f1"]),
        "seen_presence_fp_change": summary["presence"]["fp"] - fixed_summary["presence"]["fp"],
        "seen_presence_fn_change": summary["presence"]["fn"] - fixed_summary["presence"]["fn"],
        "reviews_with_fewer_pair_errors": sum(len(predictions[k] ^ gold[k]) < len(fixed[k] ^ gold[k]) for k in gold),
        "reviews_with_more_pair_errors": sum(len(predictions[k] ^ gold[k]) > len(fixed[k] ^ gold[k]) for k in gold),
        "theme_changes": {aspect: {"presence_fp_change": values["presence"]["fp"] - fixed_summary["themes"][aspect]["presence"]["fp"],
            "presence_fn_change": values["presence"]["fn"] - fixed_summary["themes"][aspect]["presence"]["fn"],
            "pair_fp_change": values["pair"]["fp"] - fixed_summary["themes"][aspect]["pair"]["fp"],
            "pair_fn_change": values["pair"]["fn"] - fixed_summary["themes"][aspect]["pair"]["fn"]}
            for aspect, values in summary["themes"].items()}}


def cases_markdown(method: str, cases: list[dict]) -> str:
    lines = [f"# {method} 已见主题开发错例", "", "私有材料，仅按已见主题错误排序；不展示留出主题标签或预测。", ""]
    for i, case in enumerate(cases, 1):
        lines.extend([f"## {i}. {case['request_id']}", "", case["text"], "",
                      f"- 主题误检：{', '.join(case['presence_fp']) or '无'}",
                      f"- 主题漏检：{', '.join(case['presence_fn']) or '无'}",
                      f"- 错误输出的主题／情感：{case['pair_fp']}",
                      f"- 未输出的正确主题／情感：{case['pair_fn']}",
                      f"- 当前评论被裁剪：{case['query_trimmed']}；选中情感输入被裁剪：{case['selected_sentiment_truncated']}", ""])
    return "\n".join(lines) + "\n"


def observations_markdown(result: dict) -> str:
    lines = ["# 已见主题开发诊断", "", "只使用清除留出主题后的标签与分数；所有评论都保留。",
        "七组已全部完成，可据这些诊断讨论最多两轮改进。" if result["matrix_complete"] else
        "七组尚未全部完成。本文件只作已完成方法的阶段画像，不选择改进方案。", "",
        "| 方法 | pair F1 | 主题误检 | 主题漏检 | 完整集合正确 |", "| --- | ---: | ---: | ---: | ---: |"]
    for method, item in result["methods"].items():
        lines.append(f"| {method} | {item['pair']['f1']*100:.2f} | {item['presence']['fp']} | {item['presence']['fn']} | {item['diagnostics']['exact_seen_label_set_reviews']}/{item['review_count']} |")
    for method, item in result["methods"].items():
        themes = sorted(item["themes"].items(), key=lambda pair: -(pair[1]["presence"]["fp"] + pair[1]["presence"]["fn"]))[:5]
        lines.extend(["", f"## {method}", "", "主题错误较多的五项：", ""])
        for aspect, values in themes:
            lines.append(f"- {aspect}：误检 {values['presence']['fp']}，漏检 {values['presence']['fn']}；检出后情感集合错误 {values['sentiment_set_errors_when_both_present']}。")
        demo, diag = item["demonstrations"]["counts"], item["diagnostics"]
        lines.extend(["", f"同主题相关示例至少一条：{demo.get('prompts_with_positive_alignment',0)}/{demo.get('prompts',0)} 个主题提示；同一提示重复评论：{demo.get('prompts_with_repeated_review',0)}。",
            f"评论文本裁剪 {diag.get('query_trimmed_reviews',0)} 条，严重裁剪 {diag.get('severe_query_trimmed_reviews',0)} 条；已输出的已见主题中，情感输入裁剪影响 {diag.get('selected_seen_sentiment_truncated_reviews',0)} 条评论。",
            "示例主题对齐与错误的关联仅为描述，不能据此断言因果；具体文本见单独私有错例文件。"])
        if "training_conflicts" in item:
            conflicts = item["training_conflicts"]
            a = conflicts["prompt_cohorts"]["ambiguous_demo_label"]
            b = conflicts["prompt_cohorts"]["conflict_group_only"]
            lines.append(f"训练冲突样例出现在 {a.get('prompts',0)+b.get('prompts',0)} 个主题提示中，其中 {a.get('prompts',0)} 个提示使用了标签本身存在冲突的示例；两类提示分别产生 {a.get('presence_errors',0)} / {b.get('presence_errors',0)} 次主题错误。")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict:
    # Reject accidental full-gold paths BEFORE reading their content.
    if args.seen_labels.name != "development_seen_labels.jsonl":
        raise ValueError("Only the explicitly sanitized development_seen_labels.jsonl view is accepted")
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / "outputs").resolve()) and not args.synthetic_test:
        raise ValueError("Private diagnostics must stay under ignored local outputs/")
    artifacts = read_json(args.artifacts)
    heldout = set(artifacts["taxonomy"]["heldout_aspects"])
    aspects = [item["aspect"] for item in artifacts["taxonomy"]["candidates"] if item["aspect"] not in heldout]
    gold = gold_by_id(read_jsonl(args.seen_labels), aspects)
    replay = read_jsonl(args.replay)
    if any(set(row) != {"request_id", "text"} for row in replay):
        raise ValueError("Replay must be label-free")
    text = {row["request_id"]: row["text"] for row in replay}
    if len(text) != len(replay) or set(text) != set(gold):
        raise ValueError("Seen gold and replay populations differ or have duplicate IDs")
    identity = read_json(args.development_root / "identity.json")
    completed = read_json(args.development_root / "summary.json")
    state = read_json(args.development_root / "state.json")
    if set(identity["request_ids"]) != set(gold):
        raise ValueError("Registered development population differs")
    if identity["replay"] != sha256(args.replay) or identity["artifacts"] != sha256(args.artifacts):
        raise ValueError("Replay or artifacts differ from the registered scoring inputs")
    matrix_complete = state["status"] in ("complete", "completed") and set(completed["methods"]) == set(identity["methods"])
    if not args.method and not matrix_complete:
        raise RuntimeError("Wait for all registered methods before full-matrix badcase analysis")
    methods = list(dict.fromkeys(["fixed", *(args.method or identity["methods"])]))
    if set(methods) - set(completed["methods"]):
        raise RuntimeError("Only completed registered method files may be profiled")
    conflict_groups = read_json(args.training_conflicts) if args.training_conflicts else None
    summaries, predictions, cases, hashes, private_conflicts = {}, {}, {}, {}, {}
    for method in methods:
        path = args.development_root / f"{method}.jsonl"
        hashes[method] = sha256(path)
        if hashes[method] != completed["methods"][method]["file_sha256"]:
            raise ValueError("Method score file is not the registered completed artifact")
        records = records_by_id([strip_nonseen(row, set(aspects)) for row in read_jsonl(path)])
        validate_record_spaces(records, args.presence_score_space)
        if set(records) != set(gold) or sha256(path) != hashes[method]:
            raise ValueError("Method population is incomplete or changed while reading")
        old = artifacts["variants"]["hybrid"]
        thresholds = calibrate_seen(records, gold, aspects, old["aspect_threshold"], old["runner_up_threshold"],
                                    presence_score_space=args.presence_score_space)
        summaries[method], predictions[method], cases[method] = profile_method(
            records, gold, aspects, thresholds, text, args.max_cases)
        if conflict_groups is not None:
            summaries[method]["training_conflicts"], private_conflicts[method] = training_conflict_profile(
                records, predictions[method], gold, conflict_groups, aspects)
    for method in methods:
        summaries[method]["versus_fixed"] = compare_fixed(summaries[method], predictions[method],
                                                         summaries["fixed"], predictions["fixed"], gold)
    result = {"scope": "historically_exposed_development_seen_only_diagnostics",
        "presence_score_space": args.presence_score_space,
        "matrix_complete": matrix_complete, "all_registered_methods": identity["methods"],
        "profiled_methods": methods, "improvement_selected": False,
        "heldout_predictions_and_labels_used": False,
        "inputs_sha256": {"seen_labels": sha256(args.seen_labels), "replay": sha256(args.replay),
                          "artifacts": sha256(args.artifacts), "completed_scores": hashes},
        "methods": summaries}
    if args.training_conflicts:
        result["inputs_sha256"]["training_seen_conflicts"] = sha256(args.training_conflicts)
    write_json(output / "summary.json", result)
    write(output / "observations.md", observations_markdown(result))
    for method, examples in cases.items():
        write_json(output / f"cases_{method}.json", examples)
        write(output / f"cases_{method}.md", cases_markdown(method, examples))
        if method in private_conflicts:
            write_json(output / f"private_conflict_exposures_{method}.json", private_conflicts[method])
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifacts", type=Path, required=True)
    result.add_argument("--seen-labels", type=Path, default=ROOT / "outputs/rag_20261111/development_seen_labels.jsonl")
    result.add_argument("--replay", type=Path, required=True)
    result.add_argument("--development-root", type=Path, required=True)
    result.add_argument("--method", action="append")
    result.add_argument("--presence-score-space", choices=["probability", "logit_margin"], default="probability")
    result.add_argument("--training-conflicts", type=Path,
                        help="Private training-only conflict group audit with seen labels only")
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--max-cases", type=int, choices=range(5, 11), default=8)
    result.set_defaults(synthetic_test=False)
    return result


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps({"matrix_complete": result["matrix_complete"], "profiled_methods": result["profiled_methods"],
                      "improvement_selected": False}))
