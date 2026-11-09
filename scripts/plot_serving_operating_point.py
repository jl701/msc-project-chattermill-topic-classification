"""Compare completed 70% and exploratory 50% one-hour runs using safe aggregates.

CPU-only plotting. No model imports, request records, labels or new experiments.
The existing six-figure manifest is preserved; this figure has its own manifest.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

from plot_serving_results import (
    complete_campaign, configure_matplotlib, finish, number, require, single,
    windows_checked,
)


SOURCES = {
    "primary": "aggregate_primary/campaign_aggregate.json",
    "original": "aggregate_resilience/campaign_aggregate.json",
    "original_windows": "resilience_analysis/resilience_analysis.json",
    "followup": "aggregate_operating_point_50pct/campaign_aggregate.json",
    "followup_windows": "operating_point_50pct_analysis/resilience_analysis.json",
}
FIGURE_NAME = "07_operating_point_comparison.png"
MANIFEST_NAME = "operating_point_manifest.json"


def prepare(study):
    documents, hashes = {}, {}
    for key, relative in SOURCES.items():
        path = study / relative
        require(path.is_file(), "Required completed aggregate is missing: " + relative)
        raw = path.read_bytes()
        documents[key] = json.loads(raw.decode("utf-8-sig"))
        hashes[relative] = hashlib.sha256(raw).hexdigest()
    primary = complete_campaign(documents["primary"], "primary")
    capacity_runs = [r for r in primary if r.get("variant") == "hybrid"
                     and r.get("purpose") == "closed_load" and r.get("client_concurrency") == 1]
    require(len(capacity_runs) == 3, "Expected three primary Hybrid closed-C1 reference runs")
    capacity = statistics.median(number(r["performance"]["successful_reviews_per_second"], "capacity")
                                 for r in capacity_runs)
    rows, configurations = [], []
    for key, fraction in (("original", .7), ("followup", .5)):
        runs = complete_campaign(documents[key], key)
        analysis = documents[key + "_windows"]
        require(analysis.get("schema_version") == "serving_resilience_analysis_v1"
                and analysis.get("campaign_status") == "completed" and not analysis.get("unfinished_runs"),
                key + " window analysis is incomplete")
        soak = single(analysis["soak"], key + " soak")
        run = single([r for r in runs if r["run"] == soak["run"]], key + " soak run")
        require(run["variant"] == "hybrid" and run["client_mode"] == "open", "Expected Hybrid open load")
        require(run["registered_arrival_duration_s"] == soak["registered_duration_s"] == 3600,
                "Both comparison runs must cover one full hour")
        rate = number(run["offered_reviews_per_second"], "offered rate")
        require(math.isclose(rate / capacity, fraction, rel_tol=0, abs_tol=1e-12),
                "Offered rate does not match the declared capacity fraction")
        cfg = run["server_configuration"]
        configurations.append({k: cfg[k] for k in (
            "variant", "pair_batch_size", "max_batch_tokens", "request_batch_size", "batch_wait_ms",
            "queue_size", "timeout_s", "logits_mode", "prefix_cache")})
        performance = run["performance"]
        require(performance["attempts"] == soak["summary"]["attempts"], "Aggregate attempt counts differ")
        windows = windows_checked(soak["windows"], 300, duration=3600)
        require(sum(r["attempts"] for r in windows) == performance["attempts"]
                and sum(r["failed"] for r in windows) == performance["failed"],
                "Five-minute cohorts do not cover every attempt")
        budget = performance["response_budgets_seconds"]["2"]
        proportion = number(budget["fraction_of_all_attempts"], "two-second delivery fraction", upper=1)
        require(math.isclose(proportion, budget["met"] / performance["attempts"], rel_tol=0, abs_tol=1e-12),
                "Two-second numerator and denominator disagree")
        rows.append({
            "capacity_fraction": fraction, "offered_reviews_per_second": rate,
            "attempts": performance["attempts"], "failed": performance["failed"],
            "successful": performance["successful"], "met_2s": budget["met"],
            "within_2s_percent": proportion * 100,
            "http_p95_s": number(performance["successful_latency"]["p95_ms"], "full-hour P95") / 1000,
            "windows": [{"start_min": r["window_start_s"] / 60, "end_min": r["window_end_s"] / 60,
                         "http_p95_s": number(r["successful_latency"]["p95_ms"], "window P95") / 1000,
                         "attempts": r["attempts"], "failed": r["failed"]} for r in windows],
        })
    require(configurations[0] == configurations[1], "The two one-hour server configurations differ")
    return {"rows": rows, "configuration": configurations[0], "reference_capacity": capacity}, hashes


def draw(plt, data):
    fig = plt.figure(figsize=(7, 6.6))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.2])
    latency = fig.add_subplot(grid[0, 0])
    delivery = fig.add_subplot(grid[0, 1])
    windows = fig.add_subplot(grid[1, :])
    colors = ("#697B91", "#23876A")
    labels = [f"{row['capacity_fraction']:.0%} 容量" for row in data["rows"]]
    for index, (row, color) in enumerate(zip(data["rows"], colors)):
        latency.bar(index, row["http_p95_s"], width=.55, color=color)
        latency.annotate(f"{row['http_p95_s']:.3f}", (index, row["http_p95_s"]),
                         xytext=(0, 5), textcoords="offset points", ha="center", fontsize=9)
        delivery.bar(index, row["within_2s_percent"], width=.55, color=color)
        delivery.annotate(f"{row['within_2s_percent']:.2f}%\n{row['met_2s']}/{row['attempts']}",
                          (index, row["within_2s_percent"]), xytext=(0, 4), textcoords="offset points",
                          ha="center", fontsize=8)
        centers = [(r["start_min"] + r["end_min"]) / 2 for r in row["windows"]]
        windows.plot(centers, [r["http_p95_s"] for r in row["windows"]],
                     color=color, marker=("o", "s")[index], markersize=4,
                     label=f"{labels[index]}（{row['offered_reviews_per_second']:.3f} 评论/s）")
    latency.set_title("全小时 HTTP P95")
    latency.set_ylabel("成功请求延迟（秒）")
    latency.set_ylim(0, max(r["http_p95_s"] for r in data["rows"]) * 1.22)
    delivery.set_title("全小时两秒内成功交付")
    delivery.set_ylabel("全部计划尝试中的比例（%）")
    delivery.set_ylim(0, 115)
    delivery.set_yticks(range(0, 101, 20))
    delivery.axhline(95, linestyle="--", linewidth=1, color="#AA4A41")
    delivery.text(.02, .055, "虚线：95% 观察目标", transform=delivery.transAxes,
                  fontsize=8, color="#AA4A41")
    for axis in (latency, delivery):
        axis.set_xticks(range(2), labels)
        axis.set_xlim(-.6, 1.6)
    windows.set_title("各自运行过程中的 5 分钟 HTTP P95")
    windows.set_ylabel("成功请求延迟（秒）")
    windows.set_xlabel("各自启动压测后的时间（分钟）")
    windows.set_xlim(0, 60)
    windows.set_xticks(range(0, 61, 10))
    windows.set_ylim(bottom=0)
    windows.axhline(2, linestyle="--", linewidth=1, color="#AA4A41")
    windows.legend(frameon=False, ncols=2, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.26))
    finish(fig, "Hybrid 一小时运行：70% 与 50% 容量对照",
           "两档各测 1 小时；50% 是查看 70% 结果后追加的探索验证，顺序运行，非随机重复试验。\n"
           "2 秒从计划到达时刻计时，全部尝试均在分母中；P95 仅包含成功请求。\n"
           "5 分钟按计划到达分组；同一批开发评论循环使用，不能当作新的独立业务样本。",
           bottom=.25, top=.87)
    fig.subplots_adjust(hspace=.58, wspace=.46)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=Path("outputs/serving_20261109"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        require(not (args.output_dir / FIGURE_NAME).exists()
                and not (args.output_dir / MANIFEST_NAME).exists(), "Preserve the existing comparison figure")
        data, hashes = prepare(args.study_root)
        plt, font = configure_matplotlib()
        fig = draw(plt, data)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / FIGURE_NAME
        fig.savefig(path, dpi=200, metadata={"Software": "Chattermill aggregate report plotting"})
        plt.close(fig)
        manifest = {"schema_version": "serving_operating_point_figure_v1", "status": "completed",
                    "generated_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": hashes,
                    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "figure": path.name, "figure_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "font": font, "dpi": 200, "plot_data": data,
                    "interpretation": "Sequential single-hour observations. 50% was selected after the 70% result; exploratory operational validation, not a randomized causal comparison or a production SLA."}
        (args.output_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print("Saved " + FIGURE_NAME, flush=True)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Plotting refused: {exc}\n")


if __name__ == "__main__":
    main()
