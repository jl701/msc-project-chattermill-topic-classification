"""Verify a fixed fallback's immutable artifact source before model loading."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping
from msc_project.serving.presence_policy import validated_presence_policy


def validate_fallback_policy(artifacts: Mapping[str, Any], variant: str) -> dict | None:
    policy = artifacts.get("retrieval_fallback_policy")
    if policy is None:
        return None
    if not isinstance(policy, dict) or policy.get("method") != "fixed" or policy.get("variant") != variant:
        raise ValueError("Fallback must explicitly identify the fixed method and current variant")
    path_value = policy.get("source_artifacts_path")
    if not isinstance(path_value, str) or not Path(path_value).is_absolute():
        raise ValueError("Fixed fallback requires an absolute source-artifact path")
    path = Path(path_value)
    try:
        raw = path.read_bytes()
        source = json.loads(raw.decode("utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise ValueError("Fixed fallback source artifact cannot be read") from exc
    source_sha = hashlib.sha256(raw).hexdigest()
    if source_sha != policy.get("source_artifacts_sha256"):
        raise ValueError("Fixed fallback source artifact SHA mismatch")
    if source.get("service_calibration", {}).get("method") != "fixed" or source.get("retrieval_fallback_policy") is not None:
        raise ValueError("Fallback source must be the independent calibrated fixed comparator")
    for key in ("fold_id", "training_scope_id", "condition", "model", "adapter", "taxonomy", "few_shot", "qlora"):
        if artifacts.get(key) != source.get(key):
            raise ValueError(f"Fallback changes frozen inference resource: {key}")
    thresholds = source.get("variants", {}).get(variant, {})
    validated = {"method": "fixed", "variant": variant, "source_artifacts_path": str(path.resolve()),
                 "source_artifacts_sha256": source_sha}
    source_presence = validated_presence_policy(thresholds)
    if validated_presence_policy(policy) != source_presence:
        raise ValueError("Fallback score space or margin threshold differs from verified fixed source")
    if validated_presence_policy(artifacts["variants"][variant])["presence_score_space"] != source_presence["presence_score_space"]:
        raise ValueError("Dynamic and fixed fallback presence score spaces must match")
    validated.update(source_presence)
    for key in ("aspect_threshold", "runner_up_threshold"):
        value = policy.get(key)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or not 0 <= value <= 1 or thresholds.get(key) != value):
            raise ValueError(f"Fallback {key} differs from its verified fixed source")
        validated[key] = float(value)
    return validated
