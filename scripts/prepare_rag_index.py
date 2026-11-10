"""Build a private train-only retrieval index without opening official test data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from msc_project.data.fabsa import default_data_dir
from msc_project.data.splits import load_official_fabsa_splits
from msc_project.serving.retrieval import build_pool, digest, E5Encoder, E5_REVISION, encode_pool


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-replay", type=Path, action="append", required=True)
    parser.add_argument("--encoder-path", type=Path)
    parser.add_argument("--encoder-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--encode", action="store_true")
    args = parser.parse_args(argv)
    import numpy as np
    artifacts = json.loads(args.artifacts.read_text(encoding="utf-8"))
    frame = load_official_fabsa_splits(args.data_dir, ("train",))
    heldout = "Account management: Account access"
    excluded = []
    source_hashes = {}
    for path in args.exclude_replay:
        source_hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        excluded.extend(json.loads(s)["text"] for s in path.read_text(encoding="utf-8").splitlines() if s.strip())
    pool, audit = build_pool(frame.to_dict("records"), artifacts["taxonomy"]["candidates"], heldout, excluded)
    fixed = artifacts["few_shot"]["demonstrations"]["aspect"]
    legal = {digest(r) for r in pool}
    invalid = [r["row_uid"] + "|" + r["candidate_aspect"] for r in fixed if digest(r) not in legal]
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    pool_hash = digest(pool)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous["pool_sha256"] != pool_hash:
            raise ValueError("Refusing to overwrite an index with a different pool")
    (args.output / "pool.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in pool), encoding="utf-8")
    manifest = {"schema_version": "chattermill_demo_index_v1", **audit,
                "pool_sha256": pool_hash, "exclusion_source_hashes": source_hashes,
                "train_csv_sha256": hashlib.sha256((args.data_dir/"train.csv").read_bytes()).hexdigest(),
                "frozen_fixed_ineligible_ids": invalid,
                "encoder_path": str(args.encoder_path) if args.encoder_path else None,
                "e5_revision": E5_REVISION, "encoded": False,
                "index_encoding_device": args.encoder_device if args.encode else None,
                "index_encoding_batch_size": 32 if args.encode else None}
    if args.encode:
        if not args.encoder_path:
            raise ValueError("--encode requires --encoder-path")
        encoder = E5Encoder(args.encoder_path, device=args.encoder_device)
        embeddings = encode_pool(pool, encoder)
        np.save(args.output / "embeddings.npy", embeddings)
        manifest.update(encoded=True, embedding_dimension=int(embeddings.shape[1]),
                        embeddings_sha256=hashlib.sha256((args.output/"embeddings.npy").read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:v for k,v in manifest.items() if k not in {"encoder_path", "exclusion_source_hashes"}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
