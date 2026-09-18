from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from msc_project.data.fabsa import default_data_dir
from msc_project.experiments.taxonomy_reuse import audit_precloud_exact_reuse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/experiments/taxonomy_two_stage_precloud_v2.json",
    )
    parser.add_argument("--write", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_precloud_exact_reuse(
        project_root=PROJECT_ROOT,
        data_dir=(args.data_dir or default_data_dir()).resolve(),
        config_path=args.config.resolve(),
    )
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.write is not None:
        path = args.write.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(payload, encoding="utf-8", newline="\n")
        temporary.replace(path)
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
