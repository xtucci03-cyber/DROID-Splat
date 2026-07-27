#!/usr/bin/env python3
"""Extract and validate MappingActivityObserver events from a run log."""

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PREFIX = "[MappingActivityObserver]"
SCHEMA = 1
ITERATION_EVENT = "iteration_activity"
UPDATE_EVENT = "update_summary"
FINAL_EVENT_TYPES = {"finalize_summary", "process_summary"}
COMMON_FIELDS = (
    "schema",
    "event_id",
    "event_type",
    "timestamp_utc",
    "pid",
    "process_role",
    "update_id",
    "iteration_id",
    "sampled",
    "status",
    "reason",
)
COUNT_FIELDS = (
    "visible_union_count",
    "touched_union_count",
    "gradient_active_any_count",
    "gradient_active_render_proxy_count",
    "scaling_grad_active_count",
)
RATIO_PAIRS = (
    ("visible_union_count", "visible_union_ratio"),
    ("touched_union_count", "touched_union_ratio"),
    ("gradient_active_any_count", "gradient_active_any_ratio"),
    (
        "gradient_active_render_proxy_count",
        "gradient_active_render_proxy_ratio",
    ),
    ("scaling_grad_active_count", "scaling_grad_active_ratio"),
)
MEMORY_FIELDS = (
    "memory_allocated_bytes",
    "memory_reserved_bytes",
    "max_memory_allocated_bytes",
    "max_memory_reserved_bytes",
)
STAGE_NAMES = (
    "iteration_total",
    "render_forward",
    "loss_computation",
    "backward",
    "densification_stats",
    "densify_and_prune",
    "optimizer_step",
    "zero_grad_housekeeping",
)
SUMMARY_MEAN_FIELDS = {
    "visible_union_count_mean": "visible_union_count",
    "visible_union_ratio_mean": "visible_union_ratio",
    "touched_union_count_mean": "touched_union_count",
    "touched_union_ratio_mean": "touched_union_ratio",
    "gradient_active_any_count_mean": "gradient_active_any_count",
    "gradient_active_any_ratio_mean": "gradient_active_any_ratio",
    "gradient_active_render_proxy_count_mean": (
        "gradient_active_render_proxy_count"
    ),
    "gradient_active_render_proxy_ratio_mean": (
        "gradient_active_render_proxy_ratio"
    ),
    "scaling_grad_active_count_mean": "scaling_grad_active_count",
}


class ValidationError(RuntimeError):
    pass


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_keys(
    event: Dict[str, Any],
    required: Sequence[str],
    location: str,
) -> List[str]:
    return [
        f"{location}: missing required field {key!r}"
        for key in required
        if key not in event
    ]


def extract_events(log_path: Path) -> List[Dict[str, Any]]:
    """Recover every prefixed JSON object, including multiple objects per line."""
    decoder = json.JSONDecoder()
    events: List[Dict[str, Any]] = []
    with log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            cursor = 0
            while True:
                marker = line.find(PREFIX, cursor)
                if marker < 0:
                    break
                json_start = line.find("{", marker + len(PREFIX))
                if json_start < 0:
                    raise ValidationError(
                        f"line {line_number}: marker has no JSON object"
                    )
                try:
                    event, consumed = decoder.raw_decode(line[json_start:])
                except json.JSONDecodeError as error:
                    raise ValidationError(
                        f"line {line_number}: malformed observer JSON: {error}"
                    ) from error
                if not isinstance(event, dict):
                    raise ValidationError(
                        f"line {line_number}: observer payload is not an object"
                    )
                event["_source_line"] = line_number
                event["_source_offset"] = marker
                events.append(event)
                cursor = json_start + consumed
    if not events:
        raise ValidationError(
            f"No {PREFIX} events found in {log_path}."
        )
    return events


def _validate_common(event: Dict[str, Any], location: str) -> List[str]:
    errors = _require_keys(
        event,
        COMMON_FIELDS,
        location,
    )
    if errors:
        return errors
    if event["schema"] != SCHEMA:
        errors.append(f"{location}: unsupported schema {event['schema']!r}")
    if not isinstance(event["event_id"], str) or not event["event_id"]:
        errors.append(f"{location}: event_id must be a non-empty string")
    if event["process_role"] != "mapper":
        errors.append(f"{location}: process_role must be 'mapper'")
    if not _is_int(event["pid"]) or event["pid"] <= 0:
        errors.append(f"{location}: pid must be a positive integer")
    if event["status"] not in {"ok", "error", "skipped"}:
        errors.append(f"{location}: invalid status {event['status']!r}")
    if not isinstance(event["sampled"], bool):
        errors.append(f"{location}: sampled must be boolean")
    return errors


def _validate_count(
    value: Any,
    n_gaussians: int,
    name: str,
    location: str,
) -> List[str]:
    if value is None:
        return []
    if not _is_int(value) or not 0 <= value <= n_gaussians:
        return [
            f"{location}: {name}={value!r} is outside [0, {n_gaussians}]"
        ]
    return []


def _validate_ratio(
    count: Any,
    ratio: Any,
    n_gaussians: int,
    name: str,
    location: str,
) -> List[str]:
    if count is None:
        if ratio is not None:
            return [f"{location}: {name} must be null when count is null"]
        return []
    expected = None if n_gaussians == 0 else count / n_gaussians
    if expected is None:
        if ratio is not None:
            return [f"{location}: {name} must be null when N is zero"]
        return []
    if (
        isinstance(ratio, bool)
        or not isinstance(ratio, (int, float))
        or not math.isfinite(float(ratio))
        or abs(float(ratio) - expected) > 1e-9
    ):
        return [
            f"{location}: {name}={ratio!r} does not match {count}/{n_gaussians}"
        ]
    return []


def _validate_nonnegative_tree(
    values: Any,
    location: str,
    field_name: str,
) -> List[str]:
    errors: List[str] = []
    if isinstance(values, dict):
        for key, child in values.items():
            errors.extend(
                _validate_nonnegative_tree(
                    child, location, f"{field_name}.{key}"
                )
            )
    elif values is not None:
        if (
            isinstance(values, bool)
            or not isinstance(values, (int, float))
            or not math.isfinite(float(values))
            or values < 0
        ):
            errors.append(
                f"{location}: {field_name} must contain non-negative numbers or null"
            )
    return errors


def _validate_memory_tree(values: Any, location: str) -> List[str]:
    errors: List[str] = []
    if not isinstance(values, dict):
        return [f"{location}: memory data must be an object"]
    for key, child in values.items():
        if key in MEMORY_FIELDS:
            if (
                child is not None
                and (
                    isinstance(child, bool)
                    or not isinstance(child, int)
                    or child < 0
                )
            ):
                errors.append(
                    f"{location}: {key} must be a non-negative integer or null"
                )
        elif isinstance(child, dict):
            errors.extend(_validate_memory_tree(child, location))
    return errors


def _validate_iteration(
    event: Dict[str, Any], location: str
) -> List[str]:
    errors = _require_keys(
        event,
        (
            "camera_uids",
            "global_gaussian_count",
            "global_gaussian_count_at_backward",
            "global_gaussian_count_after_iteration",
            "visible_count_per_camera",
            "visible_union_count",
            "visible_union_ratio",
            "touched_positive_count_per_camera",
            "touched_union_count",
            "touched_union_ratio",
            "touched_proxy_status",
            "touched_proxy_reason",
            "gradient_active_any_count",
            "gradient_active_any_ratio",
            "gradient_active_render_proxy_count",
            "gradient_active_render_proxy_ratio",
            "scaling_grad_active_count",
            "scaling_grad_active_ratio",
            "grad_none_parameter_names",
            "cpu_wall_ms_by_stage",
            "cuda_ms_by_stage",
            "memory_snapshots",
            "invariant_failures",
        ),
        location,
    )
    if errors:
        return errors

    if not _is_int(event["update_id"]) or event["update_id"] < 0:
        errors.append(f"{location}: update_id must be an integer >= 0")
    if not _is_int(event["iteration_id"]) or event["iteration_id"] < 0:
        errors.append(f"{location}: iteration_id must be an integer >= 0")
    if event["sampled"] is not True:
        errors.append(f"{location}: iteration event must be sampled")
    if event["status"] != "ok":
        errors.append(f"{location}: iteration status is not ok")

    n_gaussians = event["global_gaussian_count"]
    if not _is_int(n_gaussians) or n_gaussians < 0:
        errors.append(
            f"{location}: global_gaussian_count must be an integer >= 0"
        )
        return errors
    for key in (
        "global_gaussian_count_at_backward",
        "global_gaussian_count_after_iteration",
    ):
        if not _is_int(event[key]) or event[key] < 0:
            errors.append(f"{location}: {key} must be an integer >= 0")

    for key in COUNT_FIELDS:
        errors.extend(
            _validate_count(event[key], n_gaussians, key, location)
        )
    for count_key, ratio_key in RATIO_PAIRS:
        errors.extend(
            _validate_ratio(
                event[count_key],
                event[ratio_key],
                n_gaussians,
                ratio_key,
                location,
            )
        )

    camera_uids = event["camera_uids"]
    visible_per_camera = event["visible_count_per_camera"]
    touched_per_camera = event["touched_positive_count_per_camera"]
    if not isinstance(camera_uids, list) or not all(
        _is_int(uid) for uid in camera_uids
    ):
        errors.append(f"{location}: camera_uids must be an integer list")
    if not isinstance(visible_per_camera, list):
        errors.append(
            f"{location}: visible_count_per_camera must be a list"
        )
        visible_per_camera = []
    if not isinstance(touched_per_camera, list):
        errors.append(
            f"{location}: touched_positive_count_per_camera must be a list"
        )
        touched_per_camera = []
    if isinstance(camera_uids, list):
        if visible_per_camera and len(visible_per_camera) != len(camera_uids):
            errors.append(
                f"{location}: visible per-camera length does not match camera_uids"
            )
        if touched_per_camera and len(touched_per_camera) != len(camera_uids):
            errors.append(
                f"{location}: touched per-camera length does not match camera_uids"
            )
    for index, count in enumerate(visible_per_camera):
        errors.extend(
            _validate_count(
                count,
                n_gaussians,
                f"visible_count_per_camera[{index}]",
                location,
            )
        )
    for index, count in enumerate(touched_per_camera):
        errors.extend(
            _validate_count(
                count,
                n_gaussians,
                f"touched_positive_count_per_camera[{index}]",
                location,
            )
        )
    for index, (touched, visible) in enumerate(
        zip(touched_per_camera, visible_per_camera)
    ):
        if (
            touched is not None
            and visible is not None
            and touched > visible
        ):
            errors.append(
                f"{location}: touched proxy exceeds visible count for camera index {index}"
            )
    if (
        event["touched_union_count"] is not None
        and event["visible_union_count"] is not None
        and event["touched_union_count"] > event["visible_union_count"]
    ):
        errors.append(f"{location}: touched union exceeds visible union")
    if event["touched_proxy_status"] not in {
        "ok",
        "unavailable",
        "partial_unavailable",
        "not_collected",
    }:
        errors.append(f"{location}: invalid touched_proxy_status")
    if event["touched_proxy_status"] in {
        "unavailable",
        "partial_unavailable",
        "not_collected",
    } and not event["touched_proxy_reason"]:
        errors.append(
            f"{location}: unavailable touched proxy requires a reason"
        )
    if event["touched_proxy_status"] == "ok" and event[
        "touched_proxy_reason"
    ] is not None:
        errors.append(
            f"{location}: ok touched proxy must have a null reason"
        )

    if not isinstance(event["invariant_failures"], list):
        errors.append(f"{location}: invariant_failures must be a list")
    elif event["invariant_failures"]:
        errors.append(
            f"{location}: observer reported invariant failures: "
            f"{event['invariant_failures']}"
        )
    if not isinstance(event["grad_none_parameter_names"], list) or not all(
        isinstance(name, str)
        for name in event["grad_none_parameter_names"]
    ):
        errors.append(
            f"{location}: grad_none_parameter_names must be a string list"
        )
    for timing_field in ("cpu_wall_ms_by_stage", "cuda_ms_by_stage"):
        timing = event[timing_field]
        if not isinstance(timing, dict):
            errors.append(f"{location}: {timing_field} must be an object")
        else:
            missing_stages = [
                stage for stage in STAGE_NAMES if stage not in timing
            ]
            if missing_stages:
                errors.append(
                    f"{location}: {timing_field} is missing stages "
                    f"{missing_stages}"
                )
    errors.extend(
        _validate_nonnegative_tree(
            event["cpu_wall_ms_by_stage"],
            location,
            "cpu_wall_ms_by_stage",
        )
    )
    errors.extend(
        _validate_nonnegative_tree(
            event["cuda_ms_by_stage"],
            location,
            "cuda_ms_by_stage",
        )
    )
    errors.extend(_validate_memory_tree(event["memory_snapshots"], location))
    return errors


def _validate_update(
    event: Dict[str, Any], location: str
) -> List[str]:
    errors = _require_keys(
        event,
        (
            "update_kind",
            "sample_every",
            "gradient_eps",
            "configured_iteration_total",
            "iteration_total",
            "sampled_iteration_count",
            "global_count_start",
            "global_count_end",
            "global_count_min",
            "global_count_max",
            "explicit_observer_sync_count",
            "invariant_failures",
            "memory_maxima",
            "memory_boundary_maxima",
            *SUMMARY_MEAN_FIELDS.keys(),
        ),
        location,
    )
    if errors:
        return errors
    if not _is_int(event["update_id"]) or event["update_id"] < 0:
        errors.append(f"{location}: update_id must be an integer >= 0")
    if event["iteration_id"] is not None:
        errors.append(f"{location}: update summary iteration_id must be null")
    if event["sampled"] is not False:
        errors.append(f"{location}: update summary sampled must be false")
    if event["status"] == "error":
        errors.append(f"{location}: update summary status is error")
    elif event["status"] == "skipped" and not event["reason"]:
        errors.append(
            f"{location}: skipped update summary requires a reason"
        )
    if (
        not _is_int(event["sample_every"])
        or event["sample_every"] < 1
    ):
        errors.append(f"{location}: sample_every must be an integer >= 1")
    if (
        isinstance(event["gradient_eps"], bool)
        or not isinstance(event["gradient_eps"], (int, float))
        or not math.isfinite(float(event["gradient_eps"]))
        or event["gradient_eps"] < 0
    ):
        errors.append(
            f"{location}: gradient_eps must be a finite number >= 0"
        )
    for key in (
        "configured_iteration_total",
        "iteration_total",
        "sampled_iteration_count",
        "global_count_start",
        "global_count_end",
    ):
        if not _is_int(event[key]) or event[key] < 0:
            errors.append(f"{location}: {key} must be an integer >= 0")
    for key in ("global_count_min", "global_count_max"):
        value = event[key]
        if value is not None and (not _is_int(value) or value < 0):
            errors.append(
                f"{location}: {key} must be an integer >= 0 or null"
            )
    if (
        event["global_count_min"] is not None
        and event["global_count_max"] is not None
        and event["global_count_min"] > event["global_count_max"]
    ):
        errors.append(f"{location}: global count min exceeds max")
    if (
        _is_int(event["iteration_total"])
        and _is_int(event["sampled_iteration_count"])
        and event["sampled_iteration_count"] > event["iteration_total"]
    ):
        errors.append(
            f"{location}: sampled iteration count exceeds iteration total"
        )
    if event["status"] == "skipped" and (
        event["iteration_total"] != 0
        or event["sampled_iteration_count"] != 0
    ):
        errors.append(
            f"{location}: skipped update must not report completed iterations"
        )
    if event["explicit_observer_sync_count"] not in (0, 1):
        errors.append(
            f"{location}: explicit_observer_sync_count must be 0 or 1"
        )
    if not isinstance(event["invariant_failures"], list):
        errors.append(f"{location}: invariant_failures must be a list")
    elif event["invariant_failures"]:
        errors.append(
            f"{location}: update reported invariant failures: "
            f"{event['invariant_failures']}"
        )
    errors.extend(_validate_memory_tree(event["memory_maxima"], location))
    errors.extend(
        _validate_memory_tree(event["memory_boundary_maxima"], location)
    )
    for key, value in event.items():
        if key.endswith("_cuda_ms_total") and value is not None:
            errors.extend(
                _validate_nonnegative_tree(value, location, key)
            )
    return errors


def _validate_finalize(
    event: Dict[str, Any], location: str
) -> List[str]:
    errors = _require_keys(
        event,
        (
            "update_count",
            "iteration_event_count",
            "final_gaussian_count",
            "observer_errors",
            "pending_update",
            "pending_cuda_event_count",
        ),
        location,
    )
    if errors:
        return errors
    if event["update_id"] is not None or event["iteration_id"] is not None:
        errors.append(f"{location}: finalize IDs must be null")
    if event["sampled"] is not False:
        errors.append(f"{location}: finalize sampled must be false")
    if event["status"] != "ok":
        errors.append(f"{location}: finalize status is not ok")
    for key in (
        "update_count",
        "iteration_event_count",
        "final_gaussian_count",
        "pending_cuda_event_count",
    ):
        if not _is_int(event[key]) or event[key] < 0:
            errors.append(f"{location}: {key} must be an integer >= 0")
    if event["pending_update"] is not False:
        errors.append(f"{location}: finalize reports a pending update")
    if event["pending_cuda_event_count"] != 0:
        errors.append(f"{location}: finalize reports pending CUDA events")
    if not isinstance(event["observer_errors"], list):
        errors.append(f"{location}: observer_errors must be a list")
    elif event["observer_errors"]:
        errors.append(f"{location}: observer_errors is not empty")
    return errors


def validate_events(events: Sequence[Dict[str, Any]]) -> None:
    errors: List[str] = []
    event_ids = set()
    iteration_keys = set()
    update_summaries: Dict[int, Dict[str, Any]] = {}
    iterations_by_update: Dict[int, List[Dict[str, Any]]] = {}
    finalize_events: List[Dict[str, Any]] = []

    for index, event in enumerate(events):
        location = (
            f"event {index} (line {event.get('_source_line', '?')})"
        )
        common_errors = _validate_common(event, location)
        errors.extend(common_errors)
        if any(key not in event for key in COMMON_FIELDS):
            continue
        event_id = event.get("event_id")
        if event_id in event_ids:
            errors.append(f"{location}: duplicate event_id {event_id!r}")
        event_ids.add(event_id)

        event_type = event.get("event_type")
        if event_type == ITERATION_EVENT:
            errors.extend(_validate_iteration(event, location))
            key = (event.get("update_id"), event.get("iteration_id"))
            if key in iteration_keys:
                errors.append(f"{location}: duplicate iteration event {key}")
            iteration_keys.add(key)
            if _is_int(event.get("update_id")):
                iterations_by_update.setdefault(
                    event["update_id"], []
                ).append(event)
        elif event_type == UPDATE_EVENT:
            errors.extend(_validate_update(event, location))
            update_id = event.get("update_id")
            if update_id in update_summaries:
                errors.append(
                    f"{location}: duplicate update summary {update_id}"
                )
            elif _is_int(update_id):
                update_summaries[update_id] = event
        elif event_type in FINAL_EVENT_TYPES:
            errors.extend(_validate_finalize(event, location))
            finalize_events.append(event)
        else:
            errors.append(
                f"{location}: unsupported event_type {event_type!r}"
            )

    if len(finalize_events) != 1:
        errors.append(
            f"expected exactly one finalize/process summary, found {len(finalize_events)}"
        )

    all_update_ids = set(iterations_by_update) | set(update_summaries)
    for update_id in sorted(all_update_ids):
        if update_id not in update_summaries:
            errors.append(f"update {update_id}: missing update_summary")
            continue
        actual = len(iterations_by_update.get(update_id, []))
        expected = update_summaries[update_id].get(
            "sampled_iteration_count"
        )
        if expected != actual:
            errors.append(
                f"update {update_id}: sampled count mismatch "
                f"summary={expected}, events={actual}"
            )
        summary_event = update_summaries[update_id]
        if (
            _is_int(summary_event.get("iteration_total"))
            and _is_int(summary_event.get("sample_every"))
        ):
            actual_ids = sorted(
                event["iteration_id"]
                for event in iterations_by_update.get(update_id, [])
            )
            expected_ids = list(
                range(
                    0,
                    summary_event["iteration_total"],
                    summary_event["sample_every"],
                )
            )
            if actual_ids != expected_ids:
                errors.append(
                    f"update {update_id}: sampled iteration IDs do not "
                    f"match sample_every={summary_event['sample_every']}"
                )
        update_iterations = iterations_by_update.get(update_id, [])
        sampled_globals = [
            event["global_gaussian_count"]
            for event in update_iterations
            if _is_int(event.get("global_gaussian_count"))
        ]
        expected_min = min(sampled_globals) if sampled_globals else None
        expected_max = max(sampled_globals) if sampled_globals else None
        if summary_event.get("global_count_min") != expected_min:
            errors.append(
                f"update {update_id}: global_count_min does not match "
                "sampled iteration events"
            )
        if summary_event.get("global_count_max") != expected_max:
            errors.append(
                f"update {update_id}: global_count_max does not match "
                "sampled iteration events"
            )
        for summary_key, iteration_key in SUMMARY_MEAN_FIELDS.items():
            values = [
                float(event[iteration_key])
                for event in update_iterations
                if isinstance(event.get(iteration_key), (int, float))
                and not isinstance(event.get(iteration_key), bool)
            ]
            expected_mean = mean(values) if values else None
            actual_mean = summary_event.get(summary_key)
            if expected_mean is None:
                matches = actual_mean is None
            else:
                matches = (
                    isinstance(actual_mean, (int, float))
                    and not isinstance(actual_mean, bool)
                    and math.isfinite(float(actual_mean))
                    and abs(float(actual_mean) - expected_mean) <= 1e-9
                )
            if not matches:
                errors.append(
                    f"update {update_id}: {summary_key} does not match "
                    f"sampled {iteration_key}"
                )

    if update_summaries:
        update_ids = sorted(update_summaries)
        if update_ids != list(range(len(update_ids))):
            errors.append(
                f"update IDs are not contiguous from zero: {update_ids}"
            )

    if finalize_events:
        finalize = finalize_events[0]
        if finalize.get("update_count") != len(update_summaries):
            errors.append(
                "finalize update_count does not match update summaries"
            )
        if finalize.get("iteration_event_count") != len(iteration_keys):
            errors.append(
                "finalize iteration_event_count does not match iteration events"
            )

    if errors:
        raise ValidationError("\n".join(errors))


def _numeric_values(
    events: Iterable[Dict[str, Any]], key: str
) -> List[float]:
    values = []
    for event in events:
        value = event.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "mean": mean(values) if values else None,
        "max": max(values) if values else None,
    }


def build_summary(events: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    iterations = [
        event for event in events if event["event_type"] == ITERATION_EVENT
    ]
    updates = [
        event for event in events if event["event_type"] == UPDATE_EVENT
    ]
    finalize = [
        event
        for event in events
        if event["event_type"] in FINAL_EVENT_TYPES
    ][0]

    cuda_totals: Dict[str, float] = {}
    memory_maxima = {key: None for key in MEMORY_FIELDS}
    invariant_failures: List[str] = []
    for update in updates:
        invariant_failures.extend(update["invariant_failures"])
        for key, value in update.items():
            if key.endswith("_cuda_ms_total") and value is not None:
                cuda_totals[key] = cuda_totals.get(key, 0.0) + float(
                    value
                )
        for key in MEMORY_FIELDS:
            value = update["memory_maxima"].get(key)
            if value is not None:
                memory_maxima[key] = (
                    int(value)
                    if memory_maxima[key] is None
                    else max(memory_maxima[key], int(value))
                )

    return {
        "schema": SCHEMA,
        "validation_pass": True,
        "event_count": len(events),
        "update_count": len(updates),
        "sampled_iteration_count": len(iterations),
        "global_gaussian_count": _stats(
            _numeric_values(iterations, "global_gaussian_count")
        ),
        "visible_union_count": _stats(
            _numeric_values(iterations, "visible_union_count")
        ),
        "visible_union_ratio": _stats(
            _numeric_values(iterations, "visible_union_ratio")
        ),
        "touched_proxy_union_count": _stats(
            _numeric_values(iterations, "touched_union_count")
        ),
        "touched_proxy_union_ratio": _stats(
            _numeric_values(iterations, "touched_union_ratio")
        ),
        "gradient_active_any_count": _stats(
            _numeric_values(iterations, "gradient_active_any_count")
        ),
        "gradient_active_any_ratio": _stats(
            _numeric_values(iterations, "gradient_active_any_ratio")
        ),
        "gradient_active_render_proxy_count": _stats(
            _numeric_values(
                iterations, "gradient_active_render_proxy_count"
            )
        ),
        "gradient_active_render_proxy_ratio": _stats(
            _numeric_values(
                iterations, "gradient_active_render_proxy_ratio"
            )
        ),
        "scaling_grad_active_count": _stats(
            _numeric_values(iterations, "scaling_grad_active_count")
        ),
        "cuda_time_totals_ms": cuda_totals,
        "memory_maxima_bytes": memory_maxima,
        "explicit_observer_sync_count_total": sum(
            update["explicit_observer_sync_count"] for update in updates
        ),
        "invariant_failures": invariant_failures,
        "finalize_complete": True,
        "final_gaussian_count": finalize["final_gaussian_count"],
    }


def _clean_event(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in event.items()
        if not key.startswith("_source_")
    }


def write_outputs(
    events: Sequence[Dict[str, Any]],
    summary: Dict[str, Any],
    output_jsonl: Path,
    summary_json: Path,
) -> None:
    for path in (output_jsonl, summary_json):
        if path.exists():
            raise ValidationError(f"Refusing to overwrite existing file: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    created: List[Path] = []
    try:
        with output_jsonl.open("x", encoding="utf-8", newline="\n") as handle:
            created.append(output_jsonl)
            for event in events:
                handle.write(
                    json.dumps(
                        _clean_event(event),
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                )
        with summary_json.open("x", encoding="utf-8", newline="\n") as handle:
            created.append(summary_json)
            json.dump(
                summary,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
    except Exception:
        for path in created:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract, validate, and summarize MappingActivityObserver "
            "events without modifying the input log."
        )
    )
    parser.add_argument("run_log", type=Path, help="DROID-Splat run log")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("mapping_activity_events.jsonl"),
        help="validated event JSONL output",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("mapping_activity_summary.json"),
        help="validated summary JSON output",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        events = extract_events(args.run_log)
        validate_events(events)
        summary = build_summary(events)
        write_outputs(
            events=events,
            summary=summary,
            output_jsonl=args.output_jsonl,
            summary_json=args.summary_json,
        )
    except (OSError, TypeError, ValidationError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "validation_pass": True,
                "event_count": len(events),
                "update_count": summary["update_count"],
                "sampled_iteration_count": summary[
                    "sampled_iteration_count"
                ],
                "output_jsonl": str(args.output_jsonl.resolve()),
                "summary_json": str(args.summary_json.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
