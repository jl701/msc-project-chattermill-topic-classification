from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.baselines.candidate_label import candidate_pair_labels
from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import DEFAULT_HELDOUT_ASPECTS, build_heldout_aspect_split, load_all_fabsa
from msc_project.evaluation.metrics import evaluate_pair_and_aspect
from msc_project.llm.candidate_label import (
    PROMPT_VARIANTS,
    CostRates,
    aggregate_llm_diagnostics,
    build_candidate_messages,
    candidate_json_schema,
    diagnostic_dict,
    estimate_cost,
    extract_usage,
    parse_candidate_output,
    usage_dict,
)


DEFAULT_GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
RESPONSE_FORMATS = ("none", "json_object", "json_schema")


class ApiRequestError(RuntimeError):
    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_aspect_descriptions(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("aspect_descriptions"), dict):
        payload = payload["aspect_descriptions"]
    if not isinstance(payload, dict):
        raise ValueError("Aspect descriptions JSON must be a mapping or contain an aspect_descriptions mapping.")

    descriptions = {str(aspect): str(description).strip() for aspect, description in payload.items()}
    missing = [aspect for aspect in DEFAULT_HELDOUT_ASPECTS if not descriptions.get(aspect)]
    if missing:
        raise ValueError(f"Missing descriptions for held-out aspects: {missing}")
    return {aspect: descriptions[aspect] for aspect in DEFAULT_HELDOUT_ASPECTS}


def variant_requires_aspect_descriptions(variant: str) -> bool:
    return variant.endswith("_generated_descriptions")


def select_eval_rows(frame, limit: int | None, sample: bool, seed: int):
    if limit is None or limit >= len(frame):
        return frame.copy()
    if not sample:
        return frame.head(limit).copy()
    rng = random.Random(seed)
    selected = sorted(rng.sample(list(range(len(frame))), limit))
    return frame.iloc[selected].copy()


def first_set_env(names: list[str]) -> tuple[str | None, str | None]:
    for name in names:
        value = os.environ.get(name)
        if value:
            return name, value
    return None, None


def resolve_api_config(args) -> tuple[str, str, str]:
    api_key_envs = [args.api_key_env] if args.api_key_env else ["OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]
    api_key_env, api_key = first_set_env(api_key_envs)
    if not api_key:
        raise SystemExit(
            "Missing API key. Set OPENAI_API_KEY for the Chattermill OpenAI-compatible endpoint, "
            "or GEMINI_API_KEY/GOOGLE_API_KEY for the Google Gemini OpenAI-compatible API."
        )

    base_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    if not base_url and api_key_env in {"GEMINI_API_KEY", "GOOGLE_API_KEY"}:
        base_url = DEFAULT_GOOGLE_OPENAI_BASE_URL
    if not base_url:
        raise SystemExit("Missing base URL. Set OPENAI_BASE_URL or pass --base-url.")

    return base_url.rstrip("/"), api_key, str(api_key_env)


def response_format_payload(response_format: str) -> dict[str, Any] | None:
    if response_format == "none":
        return None
    if response_format == "json_object":
        return {"type": "json_object"}
    if response_format == "json_schema":
        return candidate_json_schema()
    raise ValueError(f"Unknown response format: {response_format}")


def output_container_for(response_format: str, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    return "object" if response_format in {"json_object", "json_schema"} else "array"


def build_payload(messages: list[dict[str, str]], args, response_format: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": messages,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    format_payload = response_format_payload(response_format)
    if format_payload is not None:
        payload["response_format"] = format_payload
    if args.seed is not None:
        payload["seed"] = int(args.seed)

    extra_body = {}
    if args.extra_body_json:
        extra_body.update(json.loads(args.extra_body_json))
    if args.thinking_budget is not None:
        google_config = extra_body.setdefault("google", {})
        thinking_config = google_config.setdefault("thinking_config", {})
        thinking_config["thinking_budget"] = int(args.thinking_budget)
    payload.update(extra_body)
    return payload


def post_chat_completion(
    base_url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    url = f"{base_url}/chat/completions"
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ApiRequestError(exc.code, body) from exc
    except urllib.error.URLError as exc:
        raise ApiRequestError(None, str(exc)) from exc


def completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "")


def call_with_retries(
    base_url: str,
    api_key: str,
    messages: list[dict[str, str]],
    fallback_messages: list[dict[str, str]],
    args,
    response_format: str,
) -> tuple[dict[str, Any], str, bool]:
    attempted_format = response_format
    for attempt in range(args.max_retries + 1):
        try:
            payload_messages = fallback_messages if attempted_format == "none" and attempted_format != response_format else messages
            payload = build_payload(payload_messages, args, attempted_format)
            response = post_chat_completion(base_url, api_key, payload, timeout=args.request_timeout)
            return response, attempted_format, attempted_format != response_format
        except ApiRequestError as exc:
            can_fallback = (
                args.response_format_fallback
                and attempted_format != "none"
                and exc.status in {400, 404, 422}
            )
            if can_fallback:
                attempted_format = "none"
                continue
            if attempt >= args.max_retries:
                raise
            sleep_seconds = args.retry_sleep * (attempt + 1)
            time.sleep(sleep_seconds)
    raise RuntimeError("Unreachable retry state.")


def run_variant_split(
    eval_df,
    split_name: str,
    aspects: list[str],
    pair_classes: list[str],
    variant: str,
    args,
    output_dir: Path,
    aspect_descriptions: dict[str, str] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    response_format = args.response_format
    output_container = output_container_for(response_format, args.output_container)
    fallback_output_container = output_container_for("none", args.output_container)
    rates = CostRates(
        input_per_1m=args.input_cost_per_1m,
        output_per_1m=args.output_cost_per_1m,
        reasoning_per_1m=args.reasoning_cost_per_1m,
    )

    request_rows = []
    start = time.time()
    for row_number, (_, row) in enumerate(eval_df.iterrows(), start=1):
        messages = build_candidate_messages(
            text=str(row["text"]),
            aspects=aspects,
            prompt_variant=variant,
            output_container=output_container,
            aspect_descriptions=aspect_descriptions,
        )
        fallback_messages = build_candidate_messages(
            text=str(row["text"]),
            aspects=aspects,
            prompt_variant=variant,
            output_container=fallback_output_container,
            aspect_descriptions=aspect_descriptions,
        )
        request_rows.append(
            {
                "row_index": row_number - 1,
                "id": str(row["id"]),
                "messages": messages,
                "fallback_messages": fallback_messages if args.response_format_fallback else None,
                "response_format": response_format,
                "output_container": output_container,
                "fallback_output_container": fallback_output_container,
            }
        )
        if args.dry_run:
            continue

        example_start = time.time()
        response, response_format_used, response_format_fallback = call_with_retries(
            base_url=str(base_url),
            api_key=str(api_key),
            messages=messages,
            fallback_messages=fallback_messages,
            args=args,
            response_format=response_format,
        )
        seconds = time.time() - example_start
        raw_output = completion_text(response)
        parsed = parse_candidate_output(
            raw_output,
            aspects,
            require_aspect_id=variant.startswith("indexed"),
            allow_object_wrapper=True,
        )
        usage = extract_usage(response)
        estimated_cost = estimate_cost(usage, rates)
        diagnostics = diagnostic_dict(parsed)
        usage_data = usage_dict(usage)
        rows.append(
            {
                "row_index": row_number - 1,
                "id": str(row["id"]),
                "row_uid": str(row.get("row_uid", "")),
                "original_split": str(row.get("original_split", "")),
                "org_index": int(row["org_index"]),
                "text": str(row["text"]),
                "gold_pair_labels": row["supervision_pair_labels"],
                "raw_output": raw_output,
                "pred_pair_labels": parsed.pair_labels,
                **diagnostics,
                "seconds": seconds,
                **{key: value for key, value in usage_data.items() if key != "raw_usage"},
                "raw_usage": usage_data["raw_usage"],
                "estimated_cost_usd": estimated_cost,
                "response_format_requested": response_format,
                "response_format_used": response_format_used,
                "response_format_fallback": response_format_fallback,
            }
        )
        print(
            f"[{split_name} {variant}] {row_number}/{len(eval_df)} "
            f"valid_json={parsed.valid_json} schema_valid={parsed.schema_valid} "
            f"predicted={len(parsed.pair_labels)} seconds={seconds:.2f}",
            flush=True,
        )

    write_jsonl(request_rows, output_dir / f"requests_{split_name}_{variant}.jsonl")
    if args.dry_run:
        return {
            "split": split_name,
            "prompt_variant": variant,
            "dry_run": True,
            "examples": int(len(eval_df)),
            "response_format": response_format,
            "output_container": output_container,
            "aspect_descriptions_used": bool(aspect_descriptions),
            "requests_file": str(output_dir / f"requests_{split_name}_{variant}.jsonl"),
        }

    pred_labels = [row["pred_pair_labels"] for row in rows]
    metrics = evaluate_pair_and_aspect(eval_df["supervision_pair_labels"].tolist(), pred_labels, pair_classes)
    diagnostics = aggregate_llm_diagnostics(rows)
    elapsed = time.time() - start
    result = {
        "split": split_name,
        "prompt_variant": variant,
        "dry_run": False,
        "model": args.model,
        "examples": int(len(eval_df)),
        "response_format": response_format,
        "output_container": output_container,
        "aspect_descriptions_used": bool(aspect_descriptions),
        "seconds": elapsed,
        "seconds_per_example": elapsed / max(1, len(rows)),
        **metrics,
        **diagnostics,
    }
    write_jsonl(rows, output_dir / f"predictions_{split_name}_{variant}.jsonl")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Gemini hosted held-out-aspect candidate-label evaluation.")
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--model", default="vertex_ai/gemini-2.5-flash")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--strategy", choices=["label_masked", "example_filtered"], default="example_filtered")
    parser.add_argument("--split", choices=["validation", "test", "both"], default="validation")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--prompt-variant", action="append", choices=PROMPT_VARIANTS, default=[])
    parser.add_argument(
        "--aspect-descriptions-json",
        type=Path,
        default=None,
        help="JSON mapping used by *_generated_descriptions prompt variants.",
    )
    parser.add_argument("--response-format", choices=RESPONSE_FORMATS, default="json_schema")
    parser.add_argument("--response-format-fallback", action="store_true")
    parser.add_argument("--output-container", choices=["auto", "array", "object"], default="auto")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--extra-body-json", default=None)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=2.0)
    parser.add_argument("--input-cost-per-1m", type=float, default=None)
    parser.add_argument("--output-cost-per-1m", type=float, default=None)
    parser.add_argument("--reasoning-cost-per-1m", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Write request JSONL files without calling the API.")
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "outputs" / "llm" / f"gemini_candidate_label_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key_env = None
    base_url = None
    api_key = None
    if not args.dry_run:
        base_url, api_key, api_key_env = resolve_api_config(args)

    frame = load_all_fabsa(args.data_dir)
    splits = build_heldout_aspect_split(
        frame,
        DEFAULT_HELDOUT_ASPECTS,
        strategy=args.strategy,
        eval_label_scope="heldout",
    )
    pair_classes = candidate_pair_labels(DEFAULT_HELDOUT_ASPECTS)
    variants = args.prompt_variant or ["indexed"]
    requires_descriptions = any(variant_requires_aspect_descriptions(variant) for variant in variants)
    if requires_descriptions and args.aspect_descriptions_json is None:
        raise SystemExit("--aspect-descriptions-json is required for *_generated_descriptions prompt variants.")

    aspect_descriptions = None
    if args.aspect_descriptions_json is not None:
        try:
            aspect_descriptions = load_aspect_descriptions(args.aspect_descriptions_json)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise SystemExit(f"Could not load aspect descriptions: {exc}") from exc
    split_names = ["validation", "test"] if args.split == "both" else [args.split]

    summary: dict[str, Any] = {
        "task": "Gemini hosted candidate-label held-out-aspect baseline",
        "model": args.model,
        "strategy": args.strategy,
        "heldout_aspects": DEFAULT_HELDOUT_ASPECTS,
        "pair_classes": pair_classes,
        "response_format": args.response_format,
        "response_format_fallback": bool(args.response_format_fallback),
        "output_container": args.output_container,
        "temperature": float(args.temperature),
        "max_tokens": int(args.max_tokens),
        "thinking_budget": args.thinking_budget,
        "aspect_descriptions_json": str(args.aspect_descriptions_json) if args.aspect_descriptions_json else None,
        "aspect_descriptions": aspect_descriptions,
        "limit": args.limit,
        "sample": bool(args.sample),
        "seed": int(args.seed),
        "dry_run": bool(args.dry_run),
        "api_key_env": api_key_env,
        "base_url": base_url,
        "cost_rates": asdict(
            CostRates(
                input_per_1m=args.input_cost_per_1m,
                output_per_1m=args.output_cost_per_1m,
                reasoning_per_1m=args.reasoning_cost_per_1m,
            )
        ),
        "results": {},
    }

    for split_name in split_names:
        eval_df = select_eval_rows(splits[split_name], args.limit, args.sample, args.seed)
        summary["results"][split_name] = {}
        for variant in variants:
            result = run_variant_split(
                eval_df=eval_df,
                split_name=split_name,
                aspects=DEFAULT_HELDOUT_ASPECTS,
                pair_classes=pair_classes,
                variant=variant,
                args=args,
                output_dir=output_dir,
                aspect_descriptions=aspect_descriptions if variant_requires_aspect_descriptions(variant) else None,
                base_url=base_url,
                api_key=api_key,
            )
            summary["results"][split_name][variant] = result
            print(json.dumps(result, indent=2), flush=True)

    write_json(summary, output_dir / "summary.json")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved Gemini candidate-label outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
