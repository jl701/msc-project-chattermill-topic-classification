"""Read public review feeds; never authenticate, bypass restrictions or make labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from msc_project.serving.review_data import parse_feed, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/rag_public_collection_20261111.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/rag_20261111/new_reviews")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "collection_manifest.json").exists():
        raise SystemExit("Snapshot already exists; use a new output directory")
    raw = output / "raw_feeds"
    raw.mkdir(exist_ok=True)
    all_rows, pages, failures = {}, [], []
    for source in config["sources"]:
        for page in range(1, config["maximum_pages_per_source"] + 1):
            url = config["endpoint"].format(**source, page=page)
            body = None
            for attempt in range(config["maximum_attempts_per_page"]):
                try:
                    request = urllib.request.Request(url, headers={"User-Agent": "ChattermillResearchReviewStudy/1.0"})
                    with urllib.request.urlopen(request, timeout=config["timeout_seconds"]) as response:
                        body = response.read()
                    break
                except (urllib.error.URLError, TimeoutError) as error:
                    failures.append({"app": source["app"], "page": page, "attempt": attempt + 1,
                                     "error_type": type(error).__name__, "status": getattr(error, "code", None)})
                    if getattr(error, "code", None) in (401, 403, 429):
                        break
                    time.sleep(config["request_delay_seconds"])
            if body is None:
                break
            stamp = datetime.now(timezone.utc).isoformat()
            payload = json.loads(body)
            rows = parse_feed(payload, **source, source_url=url)
            name = f"{source['app'].lower()}_{page:02d}.json"
            (raw / name).write_bytes(body)
            new = 0
            for row in rows:
                if row["request_id"] not in all_rows:
                    all_rows[row["request_id"]] = dict(row, collected_at=stamp)
                    new += 1
            pages.append({"app": source["app"], "page": page, "url": url, "collected_at": stamp,
                          "file": f"raw_feeds/{name}", "sha256": hashlib.sha256(body).hexdigest(),
                          "entry_count": len(rows), "new_ids": new})
            print(json.dumps({"app": source["app"], "page": page, "count": len(rows), "new": new}), flush=True)
            if not rows or new == 0:
                break
            time.sleep(config["request_delay_seconds"])
    write_jsonl(output / "collected_reviews.jsonl", list(all_rows.values()))
    manifest = {"config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
                "created_at": datetime.now(timezone.utc).isoformat(), "unique_review_ids": len(all_rows),
                "pages": pages, "failures": failures, "labels_generated": False}
    (output / "collection_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"total": len(all_rows), "pages": len(pages), "failures": len(failures)}))


if __name__ == "__main__":
    main()
