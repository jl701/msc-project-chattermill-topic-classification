"""Start the local human annotation page. No GPU, external services or prelabels."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from msc_project.serving.annotation import AnnotationStore, make_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "outputs/rag_20261111/new_reviews")
    parser.add_argument("--port", type=int, default=8769)
    args = parser.parse_args()
    store = AnnotationStore(args.data / "annotation_reviews.jsonl", args.data / "annotation_taxonomy.json", args.data / "human_labels")
    server = make_server(store, args.port)
    print(f"Annotation ready: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
