"""Final identity, data, CUDA and gate-evidence check on the formal RunPod."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
DEFAULT_PARENT = (
    PROJECT_ROOT / "configs/experiments/taxonomy_two_stage_precloud_v2.json"
)
DEFAULT_EXECUTION = (
    PROJECT_ROOT
    / "configs/experiments/taxonomy_two_stage_cloud_execution_safety_v1.json"
)
DEFAULT_GATE_ROOT = (
    PROJECT_ROOT / "outputs/experimental/taxonomy_two_stage_4090_gate_v1"
)

from msc_project.experiments.taxonomy_two_stage_formal import (  # noqa: E402
    formal_storage_budget_report,
)
from msc_project.experiments.taxonomy_methods import resolve_method_spec  # noqa: E402
from msc_project.experiments.taxonomy_post_supervisor_cloud import (  # noqa: E402
    PARALLEL_PLAN_SCHEMA as POST_SUPERVISOR_PARALLEL_PLAN_SCHEMA,
    build_worker_manifest as build_post_supervisor_worker_manifest,
    load_parallel_plan as load_post_supervisor_parallel_plan,
)
from msc_project.experiments.taxonomy_two_stage_parallel import (  # noqa: E402
    build_worker_manifest as build_legacy_worker_manifest,
    load_parallel_plan as load_legacy_parallel_plan,
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise FileNotFoundError(path)
        candidate = parent
    if not candidate.is_dir():
        candidate = candidate.parent
    return candidate


def _build_parallel_worker_manifest(
    path: Path,
    worker_id: str,
    *,
    output_root: str,
    data_dir: str,
) -> dict[str, object]:
    """Build the worker manifest with the parser registered by the plan schema."""

    raw = _read(path)
    if raw.get("schema_version") == POST_SUPERVISOR_PARALLEL_PLAN_SCHEMA:
        plan = load_post_supervisor_parallel_plan(path)
        builder = build_post_supervisor_worker_manifest
    else:
        plan = load_legacy_parallel_plan(path)
        builder = build_legacy_worker_manifest
    return builder(
        plan,
        worker_id,
        output_root=output_root,
        data_dir=data_dir,
    )


def _command(*values: str) -> str:
    return subprocess.run(
        list(values),
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def _gpu_snapshot() -> dict[str, object]:
    raw = _command(
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,temperature.gpu,power.draw,power.limit,utilization.gpu",
        "--format=csv,noheader,nounits",
    )
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("Formal executor requires exactly one visible GPU.")
    fields = [value.strip() for value in lines[0].split(",")]
    if len(fields) != 7:
        raise RuntimeError("Unexpected nvidia-smi startup schema.")
    return {
        "name": fields[0],
        "memory_total_mib": float(fields[1]),
        "memory_used_mib": float(fields[2]),
        "temperature_c": float(fields[3]),
        "power_w": float(fields[4]),
        "power_limit_w": float(fields[5]),
        "utilization_pct": float(fields[6]),
    }


def _pass_artifact(path: Path) -> dict[str, Any]:
    value = _read(path)
    if value.get("status") != "pass" or value.get("test_contract_count") != 0:
        raise ValueError(f"Required RunPod gate artifact did not pass: {path}")
    return value


def _snapshot_weight_files(snapshot: Path) -> list[Path]:
    """Return every weight shard required by a cached model snapshot."""

    index_paths = sorted(snapshot.glob("*.safetensors.index.json"))
    if index_paths:
        required: set[str] = set()
        for index_path in index_paths:
            payload = _read(index_path)
            weight_map = payload.get("weight_map")
            if not isinstance(weight_map, Mapping) or not weight_map:
                raise ValueError(f"Invalid weight index: {index_path}")
            required.update(str(value) for value in weight_map.values())
        paths = [snapshot / value for value in sorted(required)]
    else:
        paths = sorted(snapshot.glob("*.safetensors"))
        if not paths:
            paths = [
                path
                for name in ("pytorch_model.bin", "model.bin")
                if (path := snapshot / name).is_file()
            ]
    missing = [
        path.as_posix()
        for path in paths
        if not path.is_file() or path.stat().st_size == 0
    ]
    if not paths or missing:
        raise FileNotFoundError(
            f"Cached model weights are absent or incomplete: {missing or snapshot}"
        )
    return paths


def _model_cache_snapshot(hf_home: Path) -> dict[str, object]:
    """Prove both formal model revisions are complete and usable offline."""

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer

    resolved_home = hf_home.resolve()
    hub_cache = resolved_home / "hub"
    if not resolved_home.is_dir() or not hub_cache.is_dir():
        raise FileNotFoundError(
            f"Persistent Hugging Face cache is missing: {resolved_home}"
        )
    models: dict[str, object] = {}
    method_ids = (
        "distilbert_review_candidate_cross_encoder",
        "qwen_candidate_pair_qlora",
    )
    for method_id in method_ids:
        spec = resolve_method_spec(method_id)
        if not spec.model_id or not spec.model_revision:
            raise ValueError(f"Formal model is not revision-pinned: {method_id}")
        snapshot = Path(
            snapshot_download(
                repo_id=spec.model_id,
                revision=spec.model_revision,
                cache_dir=hub_cache,
                local_files_only=True,
            )
        )
        if snapshot.name != spec.model_revision:
            raise ValueError(
                f"Resolved cache revision differs from the registry for {method_id}."
            )
        broken_links = [
            path.as_posix()
            for path in snapshot.rglob("*")
            if path.is_symlink() and not path.exists()
        ]
        if broken_links:
            raise FileNotFoundError(
                f"Cached snapshot contains broken links for {method_id}: {broken_links}"
            )
        AutoConfig.from_pretrained(
            spec.model_id,
            revision=spec.model_revision,
            cache_dir=hub_cache,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            spec.model_id,
            revision=spec.model_revision,
            cache_dir=hub_cache,
            local_files_only=True,
        )
        weight_files = _snapshot_weight_files(snapshot)
        for weight_path in weight_files:
            if weight_path.suffix == ".safetensors":
                with safe_open(weight_path, framework="pt", device="cpu") as tensors:
                    if not list(tensors.keys()):
                        raise ValueError(f"Empty safetensors shard: {weight_path}")
        models[method_id] = {
            "status": "pass",
            "model_id": spec.model_id,
            "model_revision": spec.model_revision,
            "snapshot": str(snapshot),
            "tokenizer_class": tokenizer.__class__.__name__,
            "weight_file_count": len(weight_files),
            "weight_bytes": sum(path.stat().st_size for path in weight_files),
            "broken_link_count": 0,
        }
    return {
        "status": "pass",
        "hf_home": str(resolved_home),
        "hub_cache": str(hub_cache),
        "offline_only": True,
        "models": models,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    failures: list[str] = []
    checks: dict[str, bool] = {}
    head = _command("git", "rev-parse", "HEAD")
    checks["source_commit"] = head == args.expected_commit
    if not checks["source_commit"]:
        failures.append("source_commit")
    source_status = _command("git", "status", "--porcelain", "--untracked-files=no")
    checks["tracked_source_clean"] = not source_status
    if not checks["tracked_source_clean"]:
        failures.append("tracked_source_clean")

    parent = _read(args.parent_config)
    execution = _read(args.execution_config)
    expected = parent["exact_input_hashes"]
    train = args.data_dir / "train.csv"
    validation = args.data_dir / "validation.csv"
    checks["data_hashes"] = bool(
        train.is_file()
        and validation.is_file()
        and _hash(train) == expected["train_csv_sha256"]
        and _hash(validation) == expected["validation_csv_sha256"]
    )
    if not checks["data_hashes"]:
        failures.append("data_hashes")
    checks["official_test_absent"] = not (args.data_dir / "test.csv").exists()
    if not checks["official_test_absent"]:
        failures.append("official_test_absent")

    try:
        worker_manifest = _build_parallel_worker_manifest(
            args.parallel_plan,
            args.worker_id,
            output_root=str(args.output_root.resolve()),
            data_dir=str(args.data_dir.resolve()),
        )
        checks["parallel_worker_manifest"] = bool(
            worker_manifest.get("failure_count") == 0
            and worker_manifest.get("test_contract_count") == 0
        )
    except Exception as error:
        worker_manifest = {"status": "fail", "error": repr(error)}
        checks["parallel_worker_manifest"] = False
    if not checks["parallel_worker_manifest"]:
        failures.append("parallel_worker_manifest")

    try:
        import torch

        cuda = {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
    except Exception as error:
        cuda = {"available": False, "error": repr(error)}
    checks["cuda"] = bool(
        cuda.get("available")
        and cuda.get("device_count") == 1
        and cuda.get("device_name") == "NVIDIA GeForce RTX 4090"
    )
    if not checks["cuda"]:
        failures.append("cuda")

    try:
        gpu = _gpu_snapshot()
    except Exception as error:
        gpu = {"error": repr(error)}
    checks["gpu_identity_and_idle_thermal"] = bool(
        gpu.get("name") == "NVIDIA GeForce RTX 4090"
        and float(gpu.get("temperature_c", 999)) < 85
        and float(gpu.get("memory_total_mib", 0)) >= 24000
    )
    if not checks["gpu_identity_and_idle_thermal"]:
        failures.append("gpu_identity_and_idle_thermal")

    smoke_paths = [
        args.gate_root / "gate/frozen_qwen_few_shot.json",
        args.gate_root / "gate/distilbert_true_two_stage.json",
        args.gate_root / "gate/qwen_true_two_stage_qlora.json",
    ]
    try:
        smokes = {path.name: _pass_artifact(path) for path in smoke_paths}
        soak = _pass_artifact(args.gate_root / "sustained_hardware_soak_audit.json")
        checks["real_model_gate_evidence"] = True
    except Exception as error:
        smokes = {"error": repr(error)}
        soak = {}
        checks["real_model_gate_evidence"] = False
        failures.append("real_model_gate_evidence")

    backend_free_bytes = shutil.disk_usage(
        _nearest_existing_parent(args.output_root.parent)
    ).free
    try:
        storage_budget = formal_storage_budget_report(
            execution,
            observed_used_gib=args.observed_volume_used_gib,
        )
        checks["disk_capacity"] = storage_budget["status"] == "pass"
    except Exception as error:
        storage_budget = {"status": "fail", "error": repr(error)}
        checks["disk_capacity"] = False
    if not checks["disk_capacity"]:
        failures.append("disk_capacity")
    stale = (
        [path.as_posix() for path in args.output_root.rglob("*.tmp-*")]
        if args.output_root.exists()
        else []
    )
    checks["no_stale_temporary_artifacts"] = not stale
    if stale:
        failures.append("no_stale_temporary_artifacts")

    packages = {}
    for name in ("transformers", "peft", "bitsandbytes", "pandas", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    checks["required_packages"] = all(packages.values())
    if not checks["required_packages"]:
        failures.append("required_packages")

    try:
        model_cache = _model_cache_snapshot(args.hf_home)
        checks["pinned_model_cache"] = model_cache.get("status") == "pass"
    except Exception as error:
        model_cache = {
            "status": "fail",
            "hf_home": str(args.hf_home.resolve()),
            "error": repr(error),
        }
        checks["pinned_model_cache"] = False
    if not checks["pinned_model_cache"]:
        failures.append("pinned_model_cache")

    return {
        "schema_version": "taxonomy_two_stage_runpod_startup_v2",
        "status": "pass" if not failures else "fail",
        "formal_release_ready": not failures,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "expected_commit": args.expected_commit,
        "observed_commit": head,
        "tracked_source_status": source_status,
        "worker_id": args.worker_id,
        "worker_manifest": worker_manifest,
        "checks": checks,
        "cuda": cuda,
        "gpu": gpu,
        "package_versions": packages,
        "model_cache": model_cache,
        "storage_budget": storage_budget,
        "backend_reported_free_disk_gib_informational_only": backend_free_bytes
        / 1024**3,
        "stale_temporary_artifacts": stale,
        "smoke_artifacts": {
            key: value.get("schema_version") if isinstance(value, Mapping) else value
            for key, value in smokes.items()
        },
        "soak_schema": soak.get("schema_version"),
        "failure_count": len(failures),
        "failures": failures,
        "test_contract_count": 0,
        "include_official_test": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the formal RunPod immediately before launch.")
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--worker-id",
        required=True,
        choices=("worker-qlora-a", "worker-qlora-b", "worker-distil-frozen"),
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--hf-home",
        type=Path,
        required=True,
        help="Persistent Hugging Face home whose pinned snapshots must load offline.",
    )
    parser.add_argument("--parent-config", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--execution-config", type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument(
        "--parallel-plan",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs/experiments/taxonomy_two_stage_three_gpu_parallel_v1.json"
        ),
    )
    parser.add_argument("--gate-root", type=Path, default=DEFAULT_GATE_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/workspace/taxonomy_two_stage_formal_v1"),
    )
    parser.add_argument(
        "--observed-volume-used-gib",
        type=float,
        required=True,
        help=(
            "Conservative upper bound from the RunPod volume quota display; "
            "shared-backend df output is not accepted as quota evidence."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    value = run(args)
    _atomic_json(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0 if value["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
