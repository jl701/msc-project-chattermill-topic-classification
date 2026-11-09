"""Plot six Chinese report figures from completed, aggregate-only serving evidence.

This script never imports model code, reads request records, or runs experiments.
All five aggregate inputs must exist and pass completeness checks before output.
Only PNG images and a provenance/plot-data manifest are written, to a new folder.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import unicodedata


SOURCES = {
    "primary": "aggregate_primary/campaign_aggregate.json",
    "comparison": "aggregate_comparison/campaign_aggregate.json",
    "intervals": "quality_intervals.json",
    "resilience": "resilience_analysis/resilience_analysis.json",
    "resilience_campaign": "aggregate_resilience/campaign_aggregate.json",
}
VARIANTS = ("few_shot", "qlora", "hybrid")
LABELS = {"few_shot": "Few-shot", "qlora": "QLoRA", "hybrid": "Hybrid"}
COLORS = {"few_shot": "#3378A8", "qlora": "#C47828", "hybrid": "#23876A"}
MARKERS = {"few_shot": "o", "qlora": "s", "hybrid": "^"}
SCOPES = ("overall", "seen", "heldout")
CONCURRENCIES = (1, 2, 4, 8)
FRACTIONS = (.25, .5, .75, 1., 1.25)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value, context, *, lower=0, upper=None):
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value), f"Missing/nonfinite numeric value: {context}")
    require(value >= lower and (upper is None or value <= upper),
            f"Out-of-range numeric value: {context}")
    return float(value)


def single(rows, context):
    require(len(rows) == 1, f"Expected exactly one {context}; found {len(rows)}")
    return rows[0]


def complete_campaign(document, context):
    require(document.get("schema_version") == "serving_campaign_aggregate_v1",
            f"Unsupported {context} schema")
    require(document.get("campaign_status") == "completed", f"{context} is not completed")
    rows = document.get("runs", [])
    require(bool(rows) and all(r.get("status") == "completed" for r in rows),
            f"{context} contains incomplete runs")
    require(document.get("registered_run_count") == len(rows)
            == document.get("completed_run_count"), f"{context} completion counts disagree")
    require(len({r["run"] for r in rows}) == len(rows), f"{context} has duplicate run names")
    return rows


def quality_run(rows, variant, cache):
    return single([r for r in rows if r.get("quality_evaluation") is True
                   and r.get("variant") == variant
                   and r.get("server_configuration", {}).get("prefix_cache") == cache],
                  f"{variant}/{cache} quality run")


def distribution(values):
    return {"median": statistics.median(values), "min": min(values), "max": max(values)}


def performance_number(row, field):
    return number(row["performance"][field], field)


def latency_seconds(row, quantile="p95_ms"):
    return number(row["performance"]["successful_latency"][quantile], quantile) / 1000


def pressure_data(rows, mode):
    result = []
    grid = CONCURRENCIES if mode == "closed" else FRACTIONS
    for variant in VARIANTS:
        series = []
        for x in grid:
            field = "client_concurrency" if mode == "closed" else "registered_rate_fraction"
            group = [r for r in rows if r.get("variant") == variant
                     and r.get("client_mode") == mode and r.get(field) == x
                     and r.get("purpose") == ("closed_load" if mode == "closed"
                                                else "independent_arrival_load")]
            require(len(group) == 3, f"Expected three repeats for {variant}/{mode}/{x}")
            point = {"x": x, "repeats": len(group),
                     "throughput": distribution([performance_number(r, "successful_reviews_per_second")
                                                 for r in group]),
                     "http_p95_s": distribution([latency_seconds(r) for r in group]),
                     "failure_percent": distribution([
                         number(r["performance"]["error_rate"], "error_rate", upper=1) * 100
                         for r in group])}
            if mode == "open":
                capacities = [number(r["reference_capacity_reviews_per_second"], "capacity") for r in group]
                rates = [number(r["offered_reviews_per_second"], "offered rate") for r in group]
                require(all(math.isclose(rate, x * cap, rel_tol=1e-9, abs_tol=1e-12)
                            for rate, cap in zip(rates, capacities)),
                        "Offered rate differs from its registered capacity fraction")
                point["capacity_reviews_per_second"] = distribution(capacities)
                point["offered_reviews_per_second"] = distribution(rates)
            series.append(point)
        result.append({"variant": variant, "points": series})
    return result


def windows_checked(rows, width, *, duration=None):
    require(bool(rows), "Missing time windows")
    ordered = sorted(rows, key=lambda row: row["window_start_s"])
    end = 0
    for row in ordered:
        begin = number(row["window_start_s"], "window_start_s")
        next_end = number(row["window_end_s"], "window_end_s")
        require(begin == end and next_end - begin == width,
                "Time windows must be complete, contiguous and match the registered width")
        end = next_end
    if duration is not None:
        require(end == duration, "Time windows do not cover the full registered duration")
    return ordered


def telemetry_statistics(row, field):
    require(str(row.get("telemetry_scope", "")).startswith("Approximately run-aligned by UTC"),
            "Soak plot needs window-aligned telemetry, not repeated session summaries")
    stats = row["telemetry"]["metrics"][field]
    require(number(stats.get("n"), field + ".n") > 0, "Missing device telemetry samples")
    out = {key: number(stats.get(key), field + "." + key) for key in ("mean", "min", "max")}
    require(out["min"] <= out["mean"] <= out["max"], "Invalid telemetry range")
    return out


def prepare(study_root):
    documents, hashes = {}, {}
    for key, relative in SOURCES.items():
        source = study_root / relative
        require(source.is_file(), f"Required aggregate is missing: {relative}")
        raw = source.read_bytes()
        documents[key] = json.loads(raw.decode("utf-8-sig"))
        hashes[relative] = hashlib.sha256(raw).hexdigest()
    primary = complete_campaign(documents["primary"], "primary campaign")
    comparison = complete_campaign(documents["comparison"], "comparison campaign")
    resilience_runs = complete_campaign(documents["resilience_campaign"], "resilience campaign")
    intervals = documents["intervals"]
    require(intervals.get("schema_version") == "chattermill_serving_quality_intervals_v1",
            "Unsupported quality interval schema")
    require(intervals.get("f1_scale") == "0-1", "Quality interval scale must be 0-1")
    coverage = intervals["coverage"]
    require(coverage.get("complete_evaluation_coverage") is True
            and coverage["unattempted_evaluation_reviews"] == 0,
            "Quality intervals do not cover the complete frozen evaluation replay")
    review_count = coverage["frozen_evaluation_reviews"]
    require(review_count == coverage["common_unique_attempted_reviews"] and review_count > 0,
            "Quality coverage counts disagree")
    configurations = [(v, "on") for v in VARIANTS] + [(v, "off") for v in ("few_shot", "hybrid")]
    require(set(intervals["systems"]) == {v + "_" + cache for v, cache in configurations},
            "Quality systems must be the five declared VARIANT_on/off names")
    quality, http = [], []
    settings = []
    for variant, cache in configurations:
        run = quality_run(primary if cache == "on" else comparison, variant, cache)
        system = intervals["systems"][variant + "_" + cache]
        require(run["fold_id"] == intervals["fold_id"], "Quality fold identities differ")
        require(system["variant"] == variant, "Quality system name/variant mismatch")
        require(system["unique_attempted_reviews"] == review_count
                == run["independent_attempted_review_count"], "Incomplete quality replay")
        require(system["attempts"] == run["performance"]["attempts"]
                and system["successful_attempts"] == run["performance"]["successful"],
                "Quality interval and campaign attempt counts differ")
        config = run["server_configuration"]
        settings.append({k: config.get(k) for k in ("pair_batch_size", "max_batch_tokens",
                                                  "request_batch_size", "logits_mode")})
        require(run["client_mode"] == "closed" and run["client_concurrency"] == 1,
                "Quality latency comparison must use closed concurrency 1")
        scores = []
        for scope in SCOPES:
            estimate = system["views"]["failure_as_empty"]["scopes"][scope]["pair"]
            point = number(estimate["micro_f1"], "pair micro F1", upper=1)
            reference = run["quality"]["delivered_quality_failure_as_empty"][scope]["pair"]["micro_f1"]
            require(math.isclose(point, reference, rel_tol=0, abs_tol=1e-12),
                    "Interval point estimate differs from the campaign quality aggregate")
            bounds = estimate["percentile_95_interval"]
            require(isinstance(bounds, list) and len(bounds) == 2, "Missing descriptive 95% interval")
            lo, hi = [number(v, "pair F1 interval", upper=1) for v in bounds]
            require(lo <= hi, "Reversed quality interval")
            scores.append({"scope": scope, "pair_f1": point * 100,
                           "interval95": [lo * 100, hi * 100]})
        if cache == "on":
            quality.append({"variant": variant, "scores": scores})
        http.append({"variant": variant, "cache": cache,
                     "p50_s": latency_seconds(run, "p50_ms"), "p95_s": latency_seconds(run),
                     "attempts": run["performance"]["attempts"],
                     "failed": run["performance"]["failed"]})
    require(all(s == settings[0] for s in settings), "Quality inference settings are not matched")
    require(settings[0] == {"pair_batch_size": 4, "max_batch_tokens": 4096,
                            "request_batch_size": 1, "logits_mode": "last_token"},
            "Report captions require the registered pair-4/request-1 runtime")
    for row in primary:
        if row.get("purpose") in ("closed_load", "independent_arrival_load"):
            config = row["server_configuration"]
            require(config.get("prefix_cache") == "on"
                    and all(config.get(key) == value for key, value in settings[0].items()),
                    "Load curves must use the common registered primary runtime")
    resilience = documents["resilience"]
    require(resilience.get("schema_version") == "serving_resilience_analysis_v1",
            "Unsupported resilience schema")
    require(resilience.get("campaign_status") == "completed" and not resilience.get("unfinished_runs"),
            "Resilience analysis is incomplete")
    rules = resilience["rules"]
    require(rules["soak_window_seconds"] == 300 and rules["shift_window_seconds"] == 60,
            "Unexpected resilience window widths")
    soak = single(resilience["soak"], "soak run")
    require(soak["run"].startswith("hybrid_resilience__"), "Soak figure requires the Hybrid campaign")
    require(soak["registered_duration_s"] == 3600, "Soak must cover the registered 60 minutes")
    soak_run = single([row for row in resilience_runs if row["run"] == soak["run"]], "soak aggregate run")
    require(soak_run["variant"] == "hybrid" and soak_run["client_mode"] == "open", "Wrong soak identity")
    capacity_rows = [row for row in primary if row.get("variant") == "hybrid"
                     and row.get("purpose") == "closed_load" and row.get("client_concurrency") == 1]
    require(len(capacity_rows) == 3, "Soak capacity reference requires three closed-C1 repeats")
    capacity = statistics.median(performance_number(row, "successful_reviews_per_second") for row in capacity_rows)
    rate = number(soak_run["offered_reviews_per_second"], "soak offered rate")
    fraction = rate / capacity
    require(math.isclose(fraction, .7, rel_tol=0, abs_tol=1e-12), "Original soak must use 70% capacity")
    require(soak_run["performance"]["attempts"] == soak["summary"]["attempts"], "Soak aggregates disagree")
    soak_data = []
    for row in windows_checked(soak["windows"], 300, duration=3600):
        soak_data.append({"start_min": row["window_start_s"] / 60,
                          "end_min": row["window_end_s"] / 60,
                          "http_p95_s": number(row["successful_latency"]["p95_ms"], "soak P95") / 1000,
                          "queue_p95_s": number(row["successful_queue_wait"]["p95_ms"], "queue P95") / 1000,
                          "gpu_memory_mib": telemetry_statistics(row, "gpu_memory_used_mib"),
                          "gpu_temperature_c": telemetry_statistics(row, "gpu_temperature_c"),
                          "attempts": row["attempts"], "failed": row["failed"]})
    shift = single(resilience["input_shift"], "input shift run")
    require(shift["run"].startswith("hybrid_resilience__"), "Shift figure requires the Hybrid campaign")
    threshold = number(rules["input_length_js_threshold_bits"], "JS threshold", upper=1)
    require(threshold == .1, "The declared JS threshold must remain 0.1 bits")
    shift_data = []
    for row in windows_checked(shift["windows"], 60):
        require(row["signal_threshold_bits"] == threshold, "Window and global JS thresholds differ")
        require(row["minimum_sample_warning"] is (row["attempts"] < 20),
                "Small-sample warning differs from the declared window count")
        shift_data.append({"start_min": row["window_start_s"] / 60,
                           "end_min": row["window_end_s"] / 60,
                           "js_bits": number(row["input_length_js_bits"], "JS divergence", upper=1),
                           "http_p95_s": number(row["successful_latency"]["p95_ms"], "shift P95") / 1000,
                           "attempts": row["attempts"], "small_sample": row["minimum_sample_warning"],
                           "failed": row["failed"]})
    return {"review_count": review_count, "bootstrap_replicates": intervals["bootstrap_replicates"],
            "settings": settings[0], "quality": quality, "quality_http": http,
            "closed": pressure_data(primary, "closed"), "open": pressure_data(primary, "open"),
            "soak": soak_data, "soak_capacity_fraction": fraction, "soak_offered_rate": rate,
            "shift": shift_data, "js_threshold_bits": threshold}, hashes


def configure_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager, pyplot as plt
    candidates = ("Microsoft YaHei", "Noto Sans CJK SC", "Source Han Sans SC", "SimHei")
    fonts = {font.name for font in font_manager.fontManager.ttflist}
    font = next((name for name in candidates if name in fonts), None)
    require(font is not None, "A Chinese font is required; Microsoft YaHei was not found")
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": [font],
                         "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
                         "axes.unicode_minus": False, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "white",
                         "savefig.facecolor": "white", "lines.linewidth": 1.7})
    return plt, font


def wrap_note(note, columns=109):
    """Wrap CJK text by approximate display width while keeping explicit breaks."""
    output = []
    for paragraph in note.splitlines():
        current, width = "", 0
        for character in paragraph:
            next_width = 2 if unicodedata.east_asian_width(character) in ("W", "F") else 1
            if width + next_width > columns:
                output.append(current)
                current, width = "", 0
            current += character
            width += next_width
        output.append(current)
    return "\n".join(output)


def finish(fig, title, note, *, bottom=.17, top=.84):
    fig.suptitle(title, y=.98, fontsize=13, weight="bold")
    fig.text(.075, .025, wrap_note(note), ha="left", va="bottom", fontsize=8, linespacing=1.5)
    fig.subplots_adjust(left=.12, right=.97, bottom=bottom, top=top, hspace=.43, wspace=.33)
    for ax in fig.axes:
        ax.grid(axis="y", color="#D5DADF", linewidth=.6, alpha=.7)
        ax.set_axisbelow(True)


def quality_figure(plt, data):
    fig, ax = plt.subplots(figsize=(7, 4.7))
    for i, series in enumerate(data["quality"]):
        variant = series["variant"]
        for j, point in enumerate(series["scores"]):
            x = j + (i - 1) * .23
            low, high = point["interval95"]
            ax.vlines(x, low, high, color=COLORS[variant], linewidth=1.8)
            ax.hlines([low, high], x - .035, x + .035, color=COLORS[variant], linewidth=1.5)
            ax.plot(x, point["pair_f1"], MARKERS[variant], color=COLORS[variant], markersize=7,
                    label=LABELS[variant] if j == 0 else None)
            ax.annotate(f"{point['pair_f1']:.1f}", (x, high), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=8, color=COLORS[variant])
    ax.set_xticks(range(3), ["整体\nOverall", "训练见过的方面\nSeen", "训练未见的方面\nHeld-out"])
    ax.set_ylabel("方面-情感 pair micro F1（0-100 标度）")
    ax.set_ylim(0, 108)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlim(-.5, 2.5)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.15), ncols=3, frameon=False)
    finish(fig, "三种推理方案的质量：点估计与描述性 95% 区间",
           f"{data['review_count']} 条历史验证评论，单一冻结划分；前缀缓存开（QLoRA 阶段不使用缓存）。\n"
           f"整条评论为 bootstrap 单位，{data['bootstrap_replicates']} 次重采样；未成功交付记为空预测。\n"
           "区间为描述性百分位区间，不表示显著性，也不是新的盲测结果。", bottom=.24, top=.80)
    return fig


def http_figure(plt, data):
    fig, ax = plt.subplots(figsize=(7, 4.8))
    order = [("few_shot", "on"), ("few_shot", "off"), ("qlora", "on"),
             ("hybrid", "on"), ("hybrid", "off")]
    rows = [single([r for r in data["quality_http"] if (r["variant"], r["cache"]) == key],
                   "quality HTTP point") for key in order]
    maximum = max(r["p95_s"] for r in rows)
    for i, row in enumerate(rows):
        for offset, field, color, label in [(-.17, "p50_s", "#3C779B", "P50"),
                                            (.17, "p95_s", "#96BBCC", "P95")]:
            ax.barh(i + offset, row[field], height=.30, color=color, label=label if i == 0 else None)
            ax.text(row[field] + maximum * .015, i + offset, f"{row[field]:.2f}", va="center", fontsize=8)
    ax.set_yticks(range(len(rows)), [LABELS[r["variant"]] + ("\n无示例前缀复用" if r["variant"] == "qlora" else "\n缓存开" if r["cache"] == "on" else "\n缓存关")
                                    for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, maximum * 1.18)
    ax.set_xlabel("成功请求 HTTP 端到端延迟（秒 / 完整评论）")
    ax.legend(ncols=2, frameon=False, loc="upper right")
    failures = sum(r["failed"] for r in rows)
    attempts = sum(r["attempts"] for r in rows)
    finish(fig, "五种配置的质量回放延迟",
           f"各配置回放同一组 {data['review_count']} 条评论；闭环并发 1，候选对批量 4，请求批量 1。\n"
           f"P50/P95 仅统计成功请求；五配置共 {failures}/{attempts} 次失败。QLoRA 不应用前缀缓存。\n"
           "同机 HTTP 客户端计时；各配置顺序运行，不能排除温度、频率及运行顺序影响。", bottom=.25, top=.88)
    fig.subplots_adjust(left=.22)
    return fig


def line_ranges(ax, series, measure, xlabel, ylabel, *, log_x=False):
    for group in series:
        variant = group["variant"]
        points = group["points"]
        x = [p["x"] for p in points]
        ax.plot(x, [p[measure]["median"] for p in points], marker=MARKERS[variant],
                color=COLORS[variant], label=LABELS[variant], markersize=5)
        ax.fill_between(x, [p[measure]["min"] for p in points], [p[measure]["max"] for p in points],
                        color=COLORS[variant], alpha=.14, linewidth=0)
    if log_x:
        ax.set_xscale("log", base=2)
    ticks = [p["x"] for p in series[0]["points"]]
    ax.set_xticks(ticks, [f"{value:g}" for value in ticks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)


def closed_figure(plt, data):
    fig, axes = plt.subplots(1, 2, figsize=(7, 4.0))
    line_ranges(axes[0], data["closed"], "throughput", "闭环并发请求数", "成功吞吐（评论 / 秒）", log_x=True)
    line_ranges(axes[1], data["closed"], "http_p95_s", "闭环并发请求数", "成功请求 HTTP P95（秒）", log_x=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .90), ncols=3, frameon=False)
    finish(fig, "并发增加时的吞吐与响应时间",
           "每点为 3 次重复的中位数，阴影为最小–最大范围，不是置信区间。\n"
           "三方案采用同一缓存开策略、候选对批量 4、请求批量 1；吞吐计入结束后的排空时间。\n"
           "延迟仅统计成功请求；开发评论可重复出现，重复调用不构成新的独立质量样本。",
           bottom=.27, top=.77)
    return fig


def open_figure(plt, data):
    fig, axes = plt.subplots(1, 2, figsize=(7, 4.3))
    line_ranges(axes[0], data["open"], "http_p95_s", "各自 C1 容量的倍数", "成功请求 HTTP P95（秒）")
    line_ranges(axes[1], data["open"], "failure_percent", "各自 C1 容量的倍数", "失败比例（全部计划请求，%）")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .90), ncols=3, frameon=False)
    capacity = "；".join(LABELS[g["variant"]] + f" {g['points'][0]['capacity_reviews_per_second']['median']:.3f}"
                          for g in data["open"])
    finish(fig, "独立到达流量：按各方案自身容量施压",
           "横轴是各方案自身 C1 吞吐的倍数，不是相同绝对到达率。\n"
           f"1× 容量（评论 / 秒）：{capacity}。\n"
           "每点 3 次重复，中线为中位数，阴影为范围；P95 仅成功请求，失败率包含客户端丢弃。\n"
           "HTTP 延迟从实际发起请求计时，不含客户端等待计划到达时刻后的调度延迟。",
           bottom=.30, top=.77)
    return fig


def soak_figure(plt, data):
    fig, axes = plt.subplots(3, 1, figsize=(7, 6.4), sharex=True)
    rows = data["soak"]
    x = [(r["start_min"] + r["end_min"]) / 2 for r in rows]
    axes[0].plot(x, [r["http_p95_s"] for r in rows], "o-", color="#3378A8", label="HTTP P95", markersize=4)
    axes[0].plot(x, [r["queue_p95_s"] for r in rows], "s-", color="#C47828", label="排队 P95", markersize=4)
    axes[0].set_ylabel("延迟（秒）")
    axes[0].legend(ncols=2, frameon=False, loc="lower right", bbox_to_anchor=(1, 1.01))
    for ax, field, color, ylabel in [(axes[1], "gpu_memory_mib", "#23876A", "设备显存（MiB）"),
                                     (axes[2], "gpu_temperature_c", "#9E5E87", "设备温度（°C）")]:
        ax.plot(x, [r[field]["mean"] for r in rows], "o-", color=color, markersize=4)
        ax.fill_between(x, [r[field]["min"] for r in rows], [r[field]["max"] for r in rows],
                        color=color, alpha=.15, linewidth=0)
        ax.set_ylabel(ylabel)
    axes[0].set_ylim(bottom=0)
    axes[-1].set_xlim(0, 60)
    axes[-1].set_xticks(range(0, 61, 10))
    axes[-1].set_xlabel("持续运行时间（分钟；点位于各 5 分钟窗口中心）")
    finish(fig, f"Hybrid：{data['soak_capacity_fraction']:.0%} 容量连续运行 60 分钟",
           "延迟和排队时间按计划到达窗口分组，末尾窗口包含结束后的排空结果；只统计成功请求。\n"
           "设备指标为窗口内采样均值，阴影为采样最小–最大值；采样峰值不是连续真实峰值。\n"
           "显存与温度由设备驱动报告，覆盖整块 GPU，不等于服务独占显存或模型内存。",
           bottom=.19, top=.91)
    return fig


def shift_figure(plt, data):
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.8), sharex=True)
    rows = data["shift"]
    x = [(r["start_min"] + r["end_min"]) / 2 for r in rows]
    axes[0].plot(x, [r["js_bits"] for r in rows], color="#23876A", label="输入长度 JS")
    for small, fill, label in [(False, "#23876A", "窗口 ≥20 次计划请求"),
                               (True, "white", "窗口 <20 次计划请求")]:
        selected = [i for i, r in enumerate(rows) if r["small_sample"] is small]
        if selected:
            axes[0].scatter([x[i] for i in selected], [rows[i]["js_bits"] for i in selected],
                            s=25, facecolor=fill, edgecolor="#23876A", label=label, zorder=3)
    axes[0].axhline(data["js_threshold_bits"], linestyle="--", color="#AA4A41", linewidth=1.3,
                   label="描述性阈值 0.1 bits")
    axes[0].set_ylabel("输入长度 JS 散度\n（bits）", labelpad=13)
    axes[0].set_ylim(bottom=0)
    axes[0].legend(frameon=False, ncols=2, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.44))
    axes[1].plot(x, [r["http_p95_s"] for r in rows], "o-", color="#3378A8", markersize=4)
    axes[1].set_ylabel("成功请求 HTTP P95\n（秒）", labelpad=13)
    axes[1].set_ylim(bottom=0)
    axes[1].set_xlim(0, rows[-1]["end_min"])
    axes[1].set_xlabel("运行时间（分钟；每点覆盖 60 秒计划到达窗口）")
    finish(fig, "输入长度变化监控与响应延迟",
           "JS 使用固定长度分箱及开发集参考分布；超过 0.1 bits 仅产生分布变化信号。\n"
           "空心点表示不足 20 次计划请求；短窗口信号与 P95 可能不稳定。\n"
           "输入变化不等于预测质量下降；本图不能证明 F1 变化或因果关系。",
           bottom=.25, top=.76)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-root", type=Path, default=Path("outputs/serving_20261109"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        require(not args.output_dir.exists(), "Output directory already exists; choose a new plot directory")
        data, hashes = prepare(args.study_root)
        plt, font = configure_matplotlib()
        figures = [("01_quality_pair_f1", quality_figure), ("02_quality_http_latency", http_figure),
                   ("03_closed_load", closed_figure), ("04_open_load", open_figure),
                   ("05_hybrid_soak", soak_figure), ("06_input_shift", shift_figure)]
        args.output_dir.mkdir(parents=True)
        files = []
        for name, draw in figures:
            fig = draw(plt, data)
            output = args.output_dir / (name + ".png")
            fig.savefig(output, dpi=200, metadata={"Software": "Chattermill aggregate report plotting"})
            plt.close(fig)
            files.append({"file": output.name, "sha256": hashlib.sha256(output.read_bytes()).hexdigest()})
            print("Saved " + output.name, flush=True)
        manifest = {
            "schema_version": "serving_report_figures_v1", "status": "completed",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "source_sha256": hashes, "plot_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "font": font, "dpi": 200, "width_inches": 7, "figures": files,
            "definitions": {
                "few_shot": "Frozen Qwen few-shot presence and sentiment scoring",
                "qlora": "Same Qwen base with the frozen QLoRA adapter for both stages",
                "hybrid": "Frozen Qwen few-shot presence followed by QLoRA sentiment",
                "prefix_cache": "Fixed system/train-demo KV prefix only; no query or result cache",
                "quality": "Aspect-sentiment pair micro F1 x100; unique historical validation reviews; failure-as-empty",
                "intervals": "Whole-review cluster bootstrap percentile 95% intervals, descriptive only",
                "latency": "Successful-request same-host HTTP latency, not model-only forward time",
                "ranges": "Minimum/maximum of three repeated runs, not confidence intervals",
                "telemetry": "Window-aligned sampled device-wide statistics, not server-exclusive memory",
            },
            "plot_data": data,
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(2, f"Plotting refused: {exc}\n")


if __name__ == "__main__":
    main()
