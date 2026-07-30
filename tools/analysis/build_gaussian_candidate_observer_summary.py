#!/usr/bin/env python3
"""Extract and validate GaussianCandidateObserver v0 JSON events."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Optional


PREFIX = "[GaussianCandidateObserver]"
SCHEMA_VERSION = 1
COVERAGE_METHOD = "voxel_occupancy"
DEPTH_SOURCES = frozenset(
    {
        "explicit_depthmap",
        "estimated_clean_depth",
        "depth_prior",
        "random_initialization",
        "neighbor_scale_fallback",
        "unknown",
    }
)
REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "event_type",
        "event_id",
        "status",
        "reason",
        "mapper_update_id",
        "source_camera_id",
        "init",
        "observer_mode",
        "gpu_timing_enabled",
        "memory_observation_enabled",
        "depth_source",
        "depth_pixel_count",
        "valid_depth_count",
        "valid_depth_ratio",
        "pre_downsample_point_count",
        "post_downsample_point_count",
        "candidate_3d_count",
        "candidate_finite_count",
        "candidate_nonfinite_count",
        "existing_gaussian_count",
        "coverage_method",
        "coverage_voxel_size",
        "candidate_unique_voxel_count",
        "candidate_intra_voxel_duplicate_count",
        "occupied_candidate_count",
        "novel_candidate_count",
        "occupied_ratio",
        "novel_ratio",
        "spatial_extent_x",
        "spatial_extent_y",
        "spatial_extent_z",
        "observer_cpu_ms",
        "observer_gpu_ms",
        "allocated_before_bytes",
        "allocated_after_bytes",
        "reserved_before_bytes",
        "reserved_after_bytes",
        "gaussian_before",
        "gaussian_after_extend",
        "admitted_candidate_count",
        "dropped_candidate_count",
        "empty_candidate",
        "all_candidates_admitted",
        "conservation_pass",
        "error",
    }
)
COUNT_FIELDS = (
    "mapper_update_id",
    "source_camera_id",
    "depth_pixel_count",
    "valid_depth_count",
    "pre_downsample_point_count",
    "post_downsample_point_count",
    "candidate_3d_count",
    "candidate_finite_count",
    "candidate_nonfinite_count",
    "existing_gaussian_count",
    "candidate_unique_voxel_count",
    "candidate_intra_voxel_duplicate_count",
    "occupied_candidate_count",
    "novel_candidate_count",
    "gaussian_before",
    "gaussian_after_extend",
    "admitted_candidate_count",
    "dropped_candidate_count",
)
NULLABLE_FLOAT_FIELDS = (
    "valid_depth_ratio",
    "occupied_ratio",
    "novel_ratio",
    "spatial_extent_x",
    "spatial_extent_y",
    "spatial_extent_z",
    "observer_gpu_ms",
)
NULLABLE_MEMORY_FIELDS = (
    "allocated_before_bytes",
    "allocated_after_bytes",
    "reserved_before_bytes",
    "reserved_after_bytes",
)
OUTPUT_FILENAMES = (
    "candidate_observer_events.jsonl",
    "candidate_observer_summary.json",
    "candidate_observer_summary.csv",
    "CANDIDATE_OBSERVER_SUMMARY_中文.md",
)


class ValidationError(RuntimeError):
    pass


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _assert_no_nonfinite(value: Any, path: str = "event") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationError(f"{path} contains a non-finite number.")
    if isinstance(value, dict):
        for key, item in value.items():
            _assert_no_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_nonfinite(item, f"{path}[{index}]")


def _require_count(event: dict[str, Any], field: str) -> int:
    value = event[field]
    if not _is_int(value) or value < 0:
        raise ValidationError(
            f"{event.get('event_id', '<unknown>')}: {field} must be a "
            f"non-negative integer, got {value!r}."
        )
    return value


def _require_bool(event: dict[str, Any], field: str) -> bool:
    value = event[field]
    if not isinstance(value, bool):
        raise ValidationError(
            f"{event.get('event_id', '<unknown>')}: {field} must be bool, "
            f"got {value!r}."
        )
    return value


def _require_optional_float(
    event: dict[str, Any],
    field: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[float]:
    value = event[field]
    if value is None:
        return None
    if not _is_finite_number(value):
        raise ValidationError(
            f"{event.get('event_id', '<unknown>')}: {field} must be null "
            f"or finite, got {value!r}."
        )
    normalized = float(value)
    if minimum is not None and normalized < minimum:
        raise ValidationError(
            f"{event.get('event_id', '<unknown>')}: {field} must be >= "
            f"{minimum}, got {normalized}."
        )
    if maximum is not None and normalized > maximum:
        raise ValidationError(
            f"{event.get('event_id', '<unknown>')}: {field} must be <= "
            f"{maximum}, got {normalized}."
        )
    return normalized


def _assert_close(
    event_id: str,
    field: str,
    actual: float,
    expected: float,
) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValidationError(
            f"{event_id}: {field}={actual} does not match {expected}."
        )


def extract_events_from_lines(
    lines: Iterable[str],
) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        cursor = 0
        while True:
            marker = line.find(PREFIX, cursor)
            if marker < 0:
                break
            brace = line.find("{", marker + len(PREFIX))
            if brace < 0:
                raise ValidationError(
                    f"Line {line_number}: marker has no following JSON object."
                )
            try:
                event, consumed = decoder.raw_decode(line[brace:])
            except json.JSONDecodeError as error:
                raise ValidationError(
                    f"Line {line_number}: invalid candidate observer JSON: "
                    f"{error}."
                ) from error
            if not isinstance(event, dict):
                raise ValidationError(
                    f"Line {line_number}: observer payload must be an object."
                )
            _assert_no_nonfinite(event, f"line[{line_number}]")
            events.append(event)
            cursor = brace + consumed
    if not events:
        raise ValidationError("No GaussianCandidateObserver events were found.")
    return events


def extract_events(run_log: Path) -> list[dict[str, Any]]:
    if not run_log.is_file():
        raise ValidationError(f"Input run log does not exist: {run_log}.")
    with run_log.open("r", encoding="utf-8", errors="replace") as handle:
        return extract_events_from_lines(handle)


def validate_event(event: dict[str, Any]) -> None:
    event_id = event.get("event_id", "<unknown>")
    fields = set(event)
    if fields != REQUIRED_FIELDS:
        raise ValidationError(
            f"{event_id}: field mismatch; "
            f"missing={sorted(REQUIRED_FIELDS - fields)}, "
            f"extra={sorted(fields - REQUIRED_FIELDS)}."
        )
    _assert_no_nonfinite(event, str(event_id))

    if event["schema"] != SCHEMA_VERSION:
        raise ValidationError(
            f"{event_id}: schema must be {SCHEMA_VERSION}."
        )
    if event["event_type"] != "candidate_observation":
        raise ValidationError(
            f"{event_id}: invalid event_type {event['event_type']!r}."
        )
    if not isinstance(event_id, str) or not event_id:
        raise ValidationError("event_id must be a non-empty string.")
    if event["status"] not in {"ok", "error"}:
        raise ValidationError(
            f"{event_id}: status must be 'ok' or 'error'."
        )
    if not isinstance(event["reason"], str) or not event["reason"]:
        raise ValidationError(f"{event_id}: reason must be non-empty.")
    if event["observer_mode"] != "observe":
        raise ValidationError(
            f"{event_id}: observer_mode must be 'observe'."
        )
    if event["depth_source"] not in DEPTH_SOURCES:
        raise ValidationError(
            f"{event_id}: invalid depth_source {event['depth_source']!r}."
        )
    if event["coverage_method"] != COVERAGE_METHOD:
        raise ValidationError(
            f"{event_id}: coverage_method must be {COVERAGE_METHOD!r}."
        )
    if (
        not _is_finite_number(event["coverage_voxel_size"])
        or float(event["coverage_voxel_size"]) <= 0.0
    ):
        raise ValidationError(
            f"{event_id}: coverage_voxel_size must be positive and finite."
        )

    for field in COUNT_FIELDS:
        _require_count(event, field)
    for field in (
        "init",
        "empty_candidate",
        "all_candidates_admitted",
        "conservation_pass",
        "gpu_timing_enabled",
        "memory_observation_enabled",
    ):
        _require_bool(event, field)
    if (
        not _is_finite_number(event["observer_cpu_ms"])
        or float(event["observer_cpu_ms"]) < 0.0
    ):
        raise ValidationError(
            f"{event_id}: observer_cpu_ms must be finite and non-negative."
        )
    for field in NULLABLE_FLOAT_FIELDS:
        minimum = 0.0
        maximum = 1.0 if field.endswith("_ratio") else None
        _require_optional_float(
            event,
            field,
            minimum=minimum,
            maximum=maximum,
        )
    for field in NULLABLE_MEMORY_FIELDS:
        value = event[field]
        if value is not None and (not _is_int(value) or value < 0):
            raise ValidationError(
                f"{event_id}: {field} must be null or a non-negative int."
            )
    if not event["gpu_timing_enabled"] and event["observer_gpu_ms"] is not None:
        raise ValidationError(
            f"{event_id}: observer_gpu_ms must be null when GPU timing is disabled."
        )
    if not event["memory_observation_enabled"] and any(
        event[field] is not None for field in NULLABLE_MEMORY_FIELDS
    ):
        raise ValidationError(
            f"{event_id}: memory fields must be null when memory observation "
            "is disabled."
        )

    expected_id = (
        f"gco-v0:{event['mapper_update_id']}:{event['source_camera_id']}"
    )
    if event_id != expected_id:
        raise ValidationError(
            f"{event_id}: expected stable event_id {expected_id!r}."
        )

    if event["valid_depth_count"] > event["depth_pixel_count"]:
        raise ValidationError(
            f"{event_id}: valid_depth_count exceeds depth_pixel_count."
        )
    if event["depth_pixel_count"] == 0:
        if event["valid_depth_ratio"] is not None:
            raise ValidationError(
                f"{event_id}: valid_depth_ratio must be null for zero pixels."
            )
    else:
        if event["valid_depth_ratio"] is None:
            raise ValidationError(
                f"{event_id}: valid_depth_ratio must be present."
            )
        _assert_close(
            event_id,
            "valid_depth_ratio",
            float(event["valid_depth_ratio"]),
            event["valid_depth_count"] / event["depth_pixel_count"],
        )

    if (
        event["post_downsample_point_count"]
        > event["pre_downsample_point_count"]
    ):
        raise ValidationError(
            f"{event_id}: post-downsample count exceeds pre-downsample count."
        )
    if (
        event["candidate_finite_count"]
        + event["candidate_nonfinite_count"]
        != event["candidate_3d_count"]
    ):
        raise ValidationError(
            f"{event_id}: finite/nonfinite candidate conservation failed."
        )
    if (
        event["candidate_unique_voxel_count"]
        > event["candidate_finite_count"]
        or event["candidate_intra_voxel_duplicate_count"]
        != event["candidate_finite_count"]
        - event["candidate_unique_voxel_count"]
    ):
        raise ValidationError(
            f"{event_id}: candidate voxel uniqueness contract failed."
        )
    if (
        event["occupied_candidate_count"]
        + event["novel_candidate_count"]
        != event["candidate_finite_count"]
    ):
        raise ValidationError(
            f"{event_id}: occupied/novel candidate conservation failed."
        )

    finite_count = event["candidate_finite_count"]
    if finite_count == 0:
        for field in (
            "occupied_ratio",
            "novel_ratio",
            "spatial_extent_x",
            "spatial_extent_y",
            "spatial_extent_z",
        ):
            if event[field] is not None:
                raise ValidationError(
                    f"{event_id}: {field} must be null with no finite candidates."
                )
    else:
        for field in (
            "occupied_ratio",
            "novel_ratio",
            "spatial_extent_x",
            "spatial_extent_y",
            "spatial_extent_z",
        ):
            if event[field] is None and event["status"] == "ok":
                raise ValidationError(
                    f"{event_id}: {field} must be present for finite candidates."
                )
        if event["occupied_ratio"] is not None:
            _assert_close(
                event_id,
                "occupied_ratio",
                float(event["occupied_ratio"]),
                event["occupied_candidate_count"] / finite_count,
            )
        if event["novel_ratio"] is not None:
            _assert_close(
                event_id,
                "novel_ratio",
                float(event["novel_ratio"]),
                event["novel_candidate_count"] / finite_count,
            )

    if event["candidate_3d_count"] == 0:
        if not event["empty_candidate"]:
            raise ValidationError(
                f"{event_id}: empty_candidate must be true for zero candidates."
            )
        if event["post_downsample_point_count"] > 5:
            raise ValidationError(
                f"{event_id}: empty candidate event has more than five "
                "post-downsample points."
            )
    else:
        if event["empty_candidate"]:
            raise ValidationError(
                f"{event_id}: empty_candidate cannot be true."
            )
        if (
            event["post_downsample_point_count"]
            != event["candidate_3d_count"]
        ):
            raise ValidationError(
                f"{event_id}: candidate count does not match post-downsample "
                "point count."
            )

    if event["existing_gaussian_count"] != event["gaussian_before"]:
        raise ValidationError(
            f"{event_id}: existing_gaussian_count != gaussian_before."
        )
    if (
        event["candidate_3d_count"]
        != event["admitted_candidate_count"]
        or event["dropped_candidate_count"] != 0
        or not event["all_candidates_admitted"]
    ):
        raise ValidationError(
            f"{event_id}: observe-only admission contract failed."
        )
    if (
        event["gaussian_after_extend"]
        != event["gaussian_before"] + event["admitted_candidate_count"]
        or not event["conservation_pass"]
    ):
        raise ValidationError(
            f"{event_id}: Gaussian count conservation failed."
        )

    if event["status"] == "ok":
        if event["error"] is not None:
            raise ValidationError(
                f"{event_id}: ok event must have error=null."
            )
    else:
        error = event["error"]
        if (
            not isinstance(error, dict)
            or set(error) != {"type", "message"}
            or not all(isinstance(value, str) and value for value in error.values())
        ):
            raise ValidationError(
                f"{event_id}: error event needs structured error details."
            )


def validate_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    event_ids: set[str] = set()
    camera_keys: set[tuple[int, int]] = set()
    voxel_size: Optional[float] = None
    error_events: list[str] = []
    for event in events:
        validate_event(event)
        event_id = event["event_id"]
        if event_id in event_ids:
            raise ValidationError(f"Duplicate event_id: {event_id}.")
        event_ids.add(event_id)
        camera_key = (
            event["mapper_update_id"],
            event["source_camera_id"],
        )
        if camera_key in camera_keys:
            raise ValidationError(
                "Duplicate Camera observation key: "
                f"{camera_key}."
            )
        camera_keys.add(camera_key)
        current_voxel_size = float(event["coverage_voxel_size"])
        if voxel_size is None:
            voxel_size = current_voxel_size
        elif not math.isclose(
            voxel_size,
            current_voxel_size,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValidationError(
                "All events in one run must use the same voxel size."
            )
        if event["status"] == "error":
            error_events.append(event_id)
    if error_events:
        raise ValidationError(
            "Observer error events make the formal summary invalid: "
            f"{error_events}."
        )
    return sorted(
        events,
        key=lambda event: (
            event["mapper_update_id"],
            event["source_camera_id"],
            event["event_id"],
        ),
    )


def _stats(values: list[float]) -> dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "total": None,
            "mean": None,
            "min": None,
            "max": None,
        }
    return {
        "count": len(values),
        "total": float(sum(values)),
        "mean": float(statistics.fmean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _config_fingerprint(events: list[dict[str, Any]]) -> str:
    payload = {
        "schema": SCHEMA_VERSION,
        "observer_mode": "observe",
        "coverage_method": COVERAGE_METHOD,
        "coverage_voxel_size": events[0]["coverage_voxel_size"],
        "gpu_timing_enabled": events[0]["gpu_timing_enabled"],
        "memory_observation_enabled": events[0][
            "memory_observation_enabled"
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def build_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    events = validate_events(events)
    gpu_timing_enabled = events[0]["gpu_timing_enabled"]
    memory_observation_enabled = events[0]["memory_observation_enabled"]
    if any(
        event["gpu_timing_enabled"] != gpu_timing_enabled
        or event["memory_observation_enabled"]
        != memory_observation_enabled
        for event in events
    ):
        raise ValidationError(
            "All events in one run must use the same timing and memory "
            "observation configuration."
        )
    candidate_total = sum(event["candidate_3d_count"] for event in events)
    finite_total = sum(event["candidate_finite_count"] for event in events)
    nonfinite_total = sum(
        event["candidate_nonfinite_count"] for event in events
    )
    unique_voxel_total = sum(
        event["candidate_unique_voxel_count"] for event in events
    )
    intra_duplicate_total = sum(
        event["candidate_intra_voxel_duplicate_count"] for event in events
    )
    occupied_total = sum(
        event["occupied_candidate_count"] for event in events
    )
    novel_total = sum(event["novel_candidate_count"] for event in events)
    depth_pixel_total = sum(event["depth_pixel_count"] for event in events)
    valid_depth_total = sum(event["valid_depth_count"] for event in events)
    allocated_deltas = [
        event["allocated_after_bytes"] - event["allocated_before_bytes"]
        for event in events
        if event["allocated_before_bytes"] is not None
        and event["allocated_after_bytes"] is not None
    ]
    reserved_deltas = [
        event["reserved_after_bytes"] - event["reserved_before_bytes"]
        for event in events
        if event["reserved_before_bytes"] is not None
        and event["reserved_after_bytes"] is not None
    ]
    return {
        "schema": SCHEMA_VERSION,
        "validation_pass": True,
        "errors": [],
        "event_count": len(events),
        "mapper_update_ids": sorted(
            {event["mapper_update_id"] for event in events}
        ),
        "camera_count": len(
            {event["source_camera_id"] for event in events}
        ),
        "init_event_count": sum(bool(event["init"]) for event in events),
        "empty_event_count": sum(
            bool(event["empty_candidate"]) for event in events
        ),
        "error_event_count": 0,
        "candidate_total": candidate_total,
        "candidate_finite_total": finite_total,
        "candidate_nonfinite_total": nonfinite_total,
        "candidate_unique_voxel_total": unique_voxel_total,
        "candidate_intra_voxel_duplicate_total": intra_duplicate_total,
        "occupied_candidate_total": occupied_total,
        "novel_candidate_total": novel_total,
        "occupied_ratio_weighted": (
            occupied_total / finite_total if finite_total else None
        ),
        "novel_ratio_weighted": (
            novel_total / finite_total if finite_total else None
        ),
        "depth_pixel_total": depth_pixel_total,
        "valid_depth_total": valid_depth_total,
        "valid_depth_ratio_weighted": (
            valid_depth_total / depth_pixel_total
            if depth_pixel_total
            else None
        ),
        "observer_cpu_ms": _stats(
            [float(event["observer_cpu_ms"]) for event in events]
        ),
        "observer_gpu_ms": _stats(
            [
                float(event["observer_gpu_ms"])
                for event in events
                if event["observer_gpu_ms"] is not None
            ]
        ),
        "allocated_delta_bytes": _stats(
            [float(value) for value in allocated_deltas]
        ),
        "reserved_delta_bytes": _stats(
            [float(value) for value in reserved_deltas]
        ),
        "conservation_failure_count": 0,
        "duplicate_event_count": 0,
        "coverage_method": COVERAGE_METHOD,
        "coverage_voxel_size": events[0]["coverage_voxel_size"],
        "gpu_timing_enabled": gpu_timing_enabled,
        "memory_observation_enabled": memory_observation_enabled,
        "config_fingerprint": _config_fingerprint(events),
    }


def _summary_csv(summary: dict[str, Any]) -> str:
    flat = {
        key: (
            json.dumps(value, ensure_ascii=False, sort_keys=True)
            if isinstance(value, (dict, list))
            else value
        )
        for key, value in summary.items()
    }
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(flat.keys()))
    writer.writeheader()
    writer.writerow(flat)
    return output.getvalue()


def _summary_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Gaussian Candidate Observer v0 摘要",
            "",
            f"- 验证通过：`{str(summary['validation_pass']).lower()}`",
            f"- 事件数：{summary['event_count']}",
            f"- Camera数：{summary['camera_count']}",
            f"- 初始化事件：{summary['init_event_count']}",
            f"- 空候选事件：{summary['empty_event_count']}",
            f"- 候选总数：{summary['candidate_total']}",
            f"- 有限候选：{summary['candidate_finite_total']}",
            f"- 非有限候选：{summary['candidate_nonfinite_total']}",
            f"- 唯一候选体素累计：{summary['candidate_unique_voxel_total']}",
            (
                "- 候选体素内重复累计："
                f"{summary['candidate_intra_voxel_duplicate_total']}"
            ),
            f"- 已占用候选：{summary['occupied_candidate_total']}",
            f"- 新颖候选：{summary['novel_candidate_total']}",
            (
                "- 加权已占用比例："
                f"{summary['occupied_ratio_weighted']}"
            ),
            f"- 加权新颖比例：{summary['novel_ratio_weighted']}",
            f"- 有效深度累计：{summary['valid_depth_total']}",
            (
                "- 加权有效深度比例："
                f"{summary['valid_depth_ratio_weighted']}"
            ),
            f"- 覆盖方法：`{summary['coverage_method']}`",
            f"- 体素尺寸：{summary['coverage_voxel_size']}",
            f"- 守恒失败：{summary['conservation_failure_count']}",
            f"- 重复事件：{summary['duplicate_event_count']}",
            f"- 配置指纹：`{summary['config_fingerprint']}`",
            "",
            "> 本摘要只用于候选诊断，不能替代Observer关闭的clean性能结果。",
            "",
        ]
    )


def write_outputs(
    *,
    output_dir: Path,
    events: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    if output_dir.exists():
        raise ValidationError(
            f"Output directory already exists; refusing overwrite: {output_dir}."
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    sorted_events = sorted(
        events,
        key=lambda event: (
            event["mapper_update_id"],
            event["source_camera_id"],
            event["event_id"],
        ),
    )
    jsonl = "".join(
        json.dumps(
            event,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for event in sorted_events
    )
    summary_json = (
        json.dumps(
            summary,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    contents = {
        "candidate_observer_events.jsonl": jsonl,
        "candidate_observer_summary.json": summary_json,
        "candidate_observer_summary.csv": _summary_csv(summary),
        "CANDIDATE_OBSERVER_SUMMARY_中文.md": _summary_markdown(summary),
    }
    for filename in OUTPUT_FILENAMES:
        (output_dir / filename).write_text(
            contents[filename],
            encoding="utf-8",
        )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract, validate, and summarize GaussianCandidateObserver v0 "
            "events without modifying the input log."
        )
    )
    parser.add_argument("run_log", type=Path, help="Input run.log path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory; existing paths are rejected.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    try:
        events = extract_events(args.run_log)
        validated_events = validate_events(events)
        summary = build_summary(validated_events)
        write_outputs(
            output_dir=args.output_dir,
            events=validated_events,
            summary=summary,
        )
    except (OSError, ValidationError, ValueError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "validation_pass": True,
                "event_count": len(validated_events),
                "output_dir": str(args.output_dir),
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
