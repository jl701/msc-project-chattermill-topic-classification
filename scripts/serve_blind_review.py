"""Serve a separately prepared blind-review queue without changing the main UI."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import msc_project.serving.annotation as annotation
from msc_project.serving.annotation_workflow import load_queue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    load_queue(args.queue)
    # This process-local page variant never reloads or edits the live first pass.
    annotation.PAGE = annotation.PAGE.replace("首批 50 条", "盲复核队列").replace(
        "先完成首批 50 条，确定标注口径。", "本批为随机抽取的20%盲复核。请独立判断，不查看第一次标签。")
    annotation.PAGE = annotation.PAGE.replace("先完成盲复核队列，确定标注口径。", "本批为随机抽取的20%盲复核。请独立判断，不查看第一次标签。")
    store = annotation.AnnotationStore(args.queue / "annotation_reviews.jsonl", args.queue / "annotation_taxonomy.json", args.queue / "human_labels")
    server = annotation.make_server(store, args.port)
    print(f"Blind review ready: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
