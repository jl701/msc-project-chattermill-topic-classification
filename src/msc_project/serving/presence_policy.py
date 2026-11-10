"""Shared presence-score policy, independent of Torch, files and datasets."""
from __future__ import annotations

import math
from typing import Any, Mapping

PROBABILITY = "probability"
LOGIT_MARGIN = "logit_margin"


def presence_space(settings: Mapping[str, Any]) -> str:
    value = settings.get("presence_score_space", PROBABILITY)
    if value not in (PROBABILITY, LOGIT_MARGIN):
        raise ValueError("Unknown presence_score_space")
    return value


def finite_margin(value: Any) -> float:
    """Return a finite binary64 Python float, never a float32 scalar."""
    if isinstance(value, bool):
        raise ValueError("Presence numeric values cannot be boolean")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Presence numeric value must be finite") from exc
    if not math.isfinite(result):
        raise ValueError("Presence numeric value must be finite")
    return result


def active_presence_threshold(settings: Mapping[str, Any]) -> float:
    key = "presence_margin_threshold" if presence_space(settings) == LOGIT_MARGIN else "aspect_threshold"
    if key not in settings:
        raise ValueError(f"Presence policy requires {key}")
    return finite_margin(settings[key])


def validated_presence_policy(settings: Mapping[str, Any]) -> dict[str, Any]:
    if "aspect_threshold" not in settings:
        raise ValueError("The preserved probability aspect_threshold is required")
    result = {"presence_score_space": presence_space(settings),
              "aspect_threshold": finite_margin(settings["aspect_threshold"])}
    if "presence_margin_threshold" in settings:
        result["presence_margin_threshold"] = finite_margin(settings["presence_margin_threshold"])
    active_presence_threshold(result)
    return result


def presence_value(aspect_row: Mapping[str, Any], score_space: str) -> float:
    field = "presence_logit_margin" if presence_space({"presence_score_space": score_space}) == LOGIT_MARGIN else "score"
    if field not in aspect_row:
        raise ValueError(f"Aspect score requires {field}")
    return finite_margin(aspect_row[field])


def presence_selected(aspect_row: Mapping[str, Any], settings: Mapping[str, Any]) -> bool:
    return presence_value(aspect_row, presence_space(settings)) >= active_presence_threshold(settings)
